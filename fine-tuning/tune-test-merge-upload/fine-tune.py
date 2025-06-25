# Fine-tuning Script for Mac (Apple Silicon) - No CUDA Required
# Uses native PyTorch MPS backend and LoRA for efficient training

import pandas as pd
import json
import random
import re
import torch
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import login
import os
import time

class MacFineTuner:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize fine-tuner for Mac with Apple Silicon
        
        Args:
            model_name: Base model to fine-tune
        """
        self.model_name = model_name
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
    def _get_device(self):
        """Get the best available device for Mac"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

class MedicalQADatasetCreator:
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
        self.qa_pairs = []
        
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from abstract"""
        sections = {}
        text_lower = text.lower()
        
        # Section patterns - more robust regex
        patterns = {
            'background': r'background[:\s]+(.*?)(?=methods[:\s]+|results[:\s]+|conclusions[:\s]+|$)',
            'methods': r'methods[:\s]+(.*?)(?=results[:\s]+|conclusions[:\s]+|background[:\s]+|$)',
            'results': r'results[:\s]+(.*?)(?=conclusions[:\s]+|methods[:\s]+|background[:\s]+|$)',
            'conclusions': r'conclusions[:\s]+(.*?)(?=methods[:\s]+|results[:\s]+|background[:\s]+|$)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Clean up content
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                if len(content) > 50:  # Only keep substantial sections
                    sections[section] = content[:500]  # Limit length
        
        return sections
    
    def generate_qa_pairs(self) -> List[Dict]:
        """Generate Q&A pairs from abstracts"""
        qa_pairs = []
        
        print("Generating Q&A pairs from abstracts...")
        for idx, row in self.df.iterrows():
            if idx % 20 == 0:
                print(f"Processing abstract {idx+1}/100...")
                
            abstract = row['abstract_text']
            
            # Skip rows with None/null abstract text
            if abstract is None or pd.isna(abstract):
                continue
                
            sections = self.extract_sections(abstract)
            
            # Generate questions for this abstract
            questions = self._generate_questions_for_abstract(abstract, sections, idx)
            qa_pairs.extend(questions)
        
        print(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def _generate_questions_for_abstract(self, abstract: str, sections: Dict, idx: int) -> List[Dict]:
        """Generate various question types for each abstract"""
        questions = []
        
        # Basic section-based questions
        if 'background' in sections:
            questions.extend([
                {
                    'question': "What is the background of this study?",
                    'answer': sections['background'],
                    'type': 'background'
                },
                {
                    'question': "What problem does this research address?",
                    'answer': sections['background'],
                    'type': 'background'
                }
            ])
        
        if 'methods' in sections:
            questions.extend([
                {
                    'question': "What methods were used in this study?",
                    'answer': sections['methods'],
                    'type': 'methods'
                },
                {
                    'question': "How was this research conducted?",
                    'answer': sections['methods'],
                    'type': 'methods'
                }
            ])
        
        if 'results' in sections:
            questions.extend([
                {
                    'question': "What were the main results of this study?",
                    'answer': sections['results'],
                    'type': 'results'
                },
                {
                    'question': "What did this study find?",
                    'answer': sections['results'],
                    'type': 'results'
                }
            ])
        
        if 'conclusions' in sections:
            questions.extend([
                {
                    'question': "What are the conclusions of this study?",
                    'answer': sections['conclusions'],
                    'type': 'conclusions'
                },
                {
                    'question': "What do the authors conclude?",
                    'answer': sections['conclusions'],
                    'type': 'conclusions'
                }
            ])
        
        # Clinical trial specific questions
        if 'clinical trial' in abstract.lower():
            trial_info = self._extract_trial_info(abstract)
            if trial_info:
                questions.append({
                    'question': "What type of clinical trial is this?",
                    'answer': trial_info,
                    'type': 'trial_type'
                })
        
        # Drug/treatment questions
        drugs = self._extract_drug_names(abstract)
        if drugs:
            questions.append({
                'question': "What drugs or treatments are mentioned in this study?",
                'answer': f"The following treatments are mentioned: {', '.join(drugs)}. {sections.get('background', '')[:200]}",
                'type': 'treatments'
            })
        
        # Summary question
        summary = self._create_summary(sections)
        if summary:
            questions.append({
                'question': "Can you summarize this research study?",
                'answer': summary,
                'type': 'summary'
            })
        
        # Add metadata
        for q in questions:
            q['abstract_id'] = f"abstract_{idx}"
            q['source'] = 'asco_abstract'
        
        return questions
    
    def _extract_trial_info(self, text: str) -> str:
        """Extract clinical trial information"""
        trial_types = ['phase 1', 'phase 2', 'phase 3', 'phase i', 'phase ii', 'phase iii', 
                      'randomized', 'controlled', 'double-blind', 'open-label', 'multicenter']
        
        found_types = [t for t in trial_types if t in text.lower()]
        if found_types:
            return f"This is a {', '.join(found_types)} clinical trial."
        return ""
    
    def _extract_drug_names(self, text: str) -> List[str]:
        """Extract drug/treatment names"""
        # Common oncology drugs and immunotherapy agents
        drugs = [
            'pembrolizumab', 'nivolumab', 'ipilimumab', 'atezolizumab', 'durvalumab', 
            'avelumab', 'cemiplimab', 'dostarlimab', 'toripalimab', 'sintilimab',
            'rituximab', 'trastuzumab', 'bevacizumab', 'cetuximab', 'panitumumab',
            'carboplatin', 'cisplatin', 'oxaliplatin', 'docetaxel', 'paclitaxel',
            'gemcitabine', 'fluorouracil', 'capecitabine', 'temozolomide', 'dabrafenib',
            'vemurafenib', 'trametinib', 'cobimetinib', 'binimetinib', 'encorafenib'
        ]
        
        found_drugs = []
        text_lower = text.lower()
        for drug in drugs:
            if drug in text_lower:
                found_drugs.append(drug.title())
        
        return list(set(found_drugs))
    
    def _create_summary(self, sections: Dict) -> str:
        """Create a summary from available sections"""
        summary_parts = []
        
        if 'background' in sections:
            summary_parts.append(f"Background: {sections['background'][:150]}...")
        if 'methods' in sections:
            summary_parts.append(f"Methods: {sections['methods'][:100]}...")
        if 'results' in sections:
            summary_parts.append(f"Results: {sections['results'][:150]}...")
        if 'conclusions' in sections:
            summary_parts.append(f"Conclusions: {sections['conclusions'][:100]}...")
        
        return " ".join(summary_parts) if summary_parts else ""

def create_training_dataset(csv_file: str) -> Dataset:
    """Create training dataset in ChatML format for Phi-3"""
    creator = MedicalQADatasetCreator(csv_file)
    qa_pairs = creator.generate_qa_pairs()
    
    print(f"Converting {len(qa_pairs)} Q&A pairs to training format...")
    
    # Format for Phi-3 ChatML format
    formatted_data = []
    for qa in qa_pairs:
        # Phi-3 uses a specific chat format
        chat_format = f"""<|user|>
{qa['question']}<|end|>
<|assistant|>
{qa['answer']}<|end|>"""
        
        formatted_data.append({
            'text': chat_format,
            'question_type': qa.get('type', 'general'),
            'abstract_id': qa.get('abstract_id', 'unknown')
        })
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer with LoRA configuration"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization for Mac (optional, can help with memory)
    # Note: BitsAndBytesConfig may not work on Mac, so we'll use regular loading
    try:
        # Try 4-bit quantization if available
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Loaded model with 4-bit quantization")
        
    except Exception as e:
        print(f"Quantization failed ({e}), loading in float16...")
        # Fallback to regular loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if device.type != "mps" else None
        )
        
        # Move to device if not using device_map
        if device.type == "mps":
            model = model.to(device)
    
    # Prepare model for training
    model.gradient_checkpointing_enable()
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print(f"Model loaded successfully. Trainable parameters: {model.num_parameters()}")
    model.print_trainable_parameters()
    
    return model, tokenizer

def fine_tune_model(csv_file: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Fine-tune model on Mac"""
    
    # Initialize
    fine_tuner = MacFineTuner(model_name)
    
    # Create dataset
    print("Creating training dataset...")
    dataset = create_training_dataset(csv_file)
    print(f"Dataset created with {len(dataset)} examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, fine_tuner.device)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding=False, 
            max_length=1024,  # Reduced for Mac memory
            return_tensors=None
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Training arguments optimized for Mac
    training_args = TrainingArguments(
        output_dir="./medical_qa_phi3_mac",
        overwrite_output_dir=True,
        
        # Batch size and accumulation
        per_device_train_batch_size=1,  # Small batch for Mac
        gradient_accumulation_steps=8,   # Effective batch size of 8
        
        # Learning and steps
        learning_rate=2e-4,
        num_train_epochs=3,
        max_steps=200,  # Limit steps for demo
        
        # Memory optimization
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # Logging and saving
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        
        # Optimization
        optim="adamw_torch",  # Use PyTorch optimizer
        lr_scheduler_type="cosine",
        warmup_steps=20,
        
        # Disable features that might cause issues on Mac
        fp16=False,  # Disable mixed precision on Mac
        bf16=False,
        remove_unused_columns=False,
        
        # Evaluation
        eval_strategy="no",  # Disable evaluation for simplicity
        
        # Misc
        seed=42,
        data_seed=42,
        report_to=None,  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    start_time = time.time()
    
    # Train the model
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    return model, tokenizer, trainer

def test_model(model, tokenizer, test_questions: List[str]):
    """Test the fine-tuned model"""
    print("\n" + "="*50)
    print("Testing Fine-tuned Model")
    print("="*50)
    
    model.eval()
    
    for question in test_questions:
        prompt = f"""<|user|>
{question}<|end|>
<|assistant|>
"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False  # Disable cache to avoid compatibility issues
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split('<|assistant|>')[-1].strip()
        
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print("-" * 30)

def push_to_huggingface(model, tokenizer, repo_name: str, hf_token: str):
    """Push model to Hugging Face Hub"""
    print(f"\nPushing model to Hugging Face: {repo_name}")
    
    # Login
    login(token=hf_token)
    
    # Save locally first
    local_path = "./final_medical_qa_model"
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    
    # Create model card
    model_card = f"""---
library_name: peft
base_model: microsoft/Phi-3-mini-4k-instruct
tags:
- medical
- qa
- fine-tuned
- lora
---

# Medical Q&A Model (Fine-tuned Phi-3 Mini)

This model is fine-tuned on medical research abstracts from ASCO 2025 for question-answering tasks.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base_model, "{repo_name}")

# Ask a question
question = "What is pembrolizumab used for?"
prompt = f"<|user|>\\n{{question}}<|end|>\\n<|assistant|>\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

- Base model: microsoft/Phi-3-mini-4k-instruct
- Training data: 100 ASCO medical research abstracts
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Training platform: Mac with Apple Silicon
"""

    with open(f"{local_path}/README.md", "w") as f:
        f.write(model_card)
    
    # Push to Hub
    try:
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)
        print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"❌ Error pushing to Hub: {e}")

def main():
    """Main training pipeline"""
    # Configuration
    CSV_FILE = "HundredASCOAbstract2025.csv"
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    HF_TOKEN = "hf_12345"  # Replace with your token
    REPO_NAME = "goaiguru/medical-qa-phi3-mini-mac"  # Replace with your repo
    
    # Test questions
    test_questions = [
        "What is pembrolizumab used for?",
        "What are common adverse events in immunotherapy trials?",
        "How do clinical trials measure treatment effectiveness?",
        "What biomarkers are used for patient selection in cancer treatment?",
        "What is the difference between Phase I and Phase III trials?"
    ]
    
    print("🏥 Medical Q&A Fine-tuning for Mac")
    print("="*50)
    
    try:
        # Fine-tune model
        model, tokenizer, trainer = fine_tune_model(CSV_FILE, MODEL_NAME)
        
        # Test model
        test_model(model, tokenizer, test_questions)
        
        # Optionally push to Hugging Face (uncomment when ready)
        # if HF_TOKEN != "your_huggingface_token_here":
        #     push_to_huggingface(model, tokenizer, REPO_NAME, HF_TOKEN)
        
        print("\n✅ Fine-tuning completed successfully!")
        print(f"Model saved to: ./medical_qa_phi3_mac")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Mac-specific requirements:
# pip install torch torchvision torchaudio
# pip install transformers datasets peft accelerate
# pip install huggingface_hub pandas
# 
# Note: No CUDA or Unsloth dependencies needed!

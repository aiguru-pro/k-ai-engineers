#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def push_to_huggingface(model_path, repo_name, hf_token):
    """Push already trained model to Hugging Face"""
    print("🚀 Uploading Model to Hugging Face")
    print("=" * 50)
    
    # Load the trained model
    print("Loading trained model...")
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Load the fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Create local directory for upload
    local_path = "./temp_upload"
    os.makedirs(local_path, exist_ok=True)
    
    # Save model locally first
    print("Preparing model for upload...")
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
- asco
- oncology
---

# Medical Q&A Model (Fine-tuned Phi-3 Mini)

This model is fine-tuned on medical research abstracts from ASCO 2025 for question-answering tasks.

## Model Details

- **Base model**: microsoft/Phi-3-mini-4k-instruct
- **Training data**: 100 ASCO medical research abstracts
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) 
- **Training platform**: Mac with Apple Silicon
- **Total Q&A pairs**: 924
- **Training epochs**: 3
- **Training steps**: 200

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

# Load fine-tuned model
model = PeftModel.from_pretrained(base_model, "{repo_name}")

# Ask a medical question
question = "What are the main findings about pembrolizumab in melanoma treatment?"
prompt = f"<|user|>\\n{{question}}<|end|>\\n<|assistant|>\\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split('<|assistant|>')[-1].strip()
print(answer)
```

## Training Details

The model was fine-tuned on medical research abstracts focusing on:
- Immunotherapy treatments
- Clinical trial results
- Adverse events reporting
- Biomarker identification
- Treatment efficacy measurements

## Intended Use

This model is designed for educational and research purposes in medical question-answering. It should not be used for clinical decision-making or patient care without proper medical supervision.

## Limitations

- Trained on a limited dataset of 100 abstracts
- Focused primarily on oncology/immunotherapy research
- May not generalize to all medical domains
- Should not replace professional medical advice
"""

    with open(f"{local_path}/README.md", "w") as f:
        f.write(model_card)
    
    # Push to Hub
    try:
        print(f"Uploading to: {repo_name}")
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)
        print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(local_path)
        
    except Exception as e:
        print(f"❌ Error pushing to Hub: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Configuration
    MODEL_PATH = "./medical_qa_phi3_mac/checkpoint-200"  # Your trained model
    HF_TOKEN = "hf_12345"  # Your HF token
    REPO_NAME = "goaiguru/medical-qa-phi3-mini-mac"  # Your repo name
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please ensure the model was trained successfully.")
        return
    
    try:
        push_to_huggingface(MODEL_PATH, REPO_NAME, HF_TOKEN)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

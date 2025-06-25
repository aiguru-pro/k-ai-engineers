#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_models():
    """Load both base model and fine-tuned model"""
    print("🏥 Loading Base and Fine-tuned Medical Q&A Models")
    print("=" * 50)
    
    # Load base model and tokenizer
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load fine-tuned adapter
    adapter_path = "./medical_qa_phi3_mac/checkpoint-200"
    print(f"Loading fine-tuned adapter from: {adapter_path}")
    
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    finetuned_model.eval()
    
    print("✅ Both models loaded successfully!")
    return base_model, finetuned_model, tokenizer

def get_model_response(model, tokenizer, question):
    """Get response from a model for a given question"""
    # Format prompt
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
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
            use_cache=False
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split('<|assistant|>')[-1].strip()
    return answer

def compare_models(base_model, finetuned_model, tokenizer, questions):
    """Compare responses from base model vs fine-tuned model"""
    print("\n" + "=" * 80)
    print("🔍 COMPARISON: Base Model vs Fine-tuned Model")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("=" * 60)
        
        # Get base model response
        print("🤖 BASE MODEL Response:")
        print("-" * 30)
        base_answer = get_model_response(base_model, tokenizer, question)
        print(base_answer)
        
        print("\n🎯 FINE-TUNED MODEL Response:")
        print("-" * 30)
        finetuned_answer = get_model_response(finetuned_model, tokenizer, question)
        print(finetuned_answer)
        
        print("\n" + "=" * 60)

def main():
    # Test questions
    test_questions = [
        "What are the main findings about pembrolizumab in melanoma treatment?",
        "What are the common adverse events reported in immunotherapy trials?",
        "Which biomarkers are mentioned for patient selection?"
    ]
    
    try:
        # Load both models
        base_model, finetuned_model, tokenizer = load_models()
        
        # Compare the models
        compare_models(base_model, finetuned_model, tokenizer, test_questions)
        
        print("\n" + "=" * 80)
        print("✅ Comparison completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

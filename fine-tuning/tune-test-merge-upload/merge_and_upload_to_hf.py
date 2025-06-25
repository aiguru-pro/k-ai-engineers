#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

def merge_and_upload_model(adapter_path, repo_name, hf_token):
    """Merge LoRA adapter with base model and upload merged version"""
    print("🔄 Merging LoRA Adapter with Base Model")
    print("=" * 60)
    
    # Load base model and tokenizer
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter weights into base model
    print("🔗 Merging adapter weights into base model...")
    merged_model = model_with_adapter.merge_and_unload()
    
    # Save merged model locally
    local_path = "./merged_model_temp"
    print(f"Saving merged model to: {local_path}")
    merged_model.save_pretrained(local_path, safe_serialization=True)
    tokenizer.save_pretrained(local_path)
    
    # Create model card for merged version
    model_card = f"""---
library_name: transformers
base_model: microsoft/Phi-3-mini-4k-instruct
tags:
- medical
- qa
- fine-tuned
- merged
- asco
- oncology
license: mit
---

# Medical Q&A Model - MERGED (Fine-tuned Phi-3 Mini)

🚀 **This is the MERGED version** - Ready for production use!

This model contains the fine-tuned weights merged directly into the base Phi-3 Mini model. No adapter loading required.

## Model Details

- **Base model**: microsoft/Phi-3-mini-4k-instruct
- **Training data**: 100 ASCO medical research abstracts
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) - now merged
- **Training platform**: Mac with Apple Silicon
- **Total Q&A pairs**: 924
- **Training epochs**: 3
- **Training steps**: 200

## 🎯 Perfect for Production & SageMaker

This merged model works seamlessly with:
- SageMaker inference endpoints
- Standard transformers pipelines
- Any deployment platform
- No special loading code needed

## Usage

### Simple Loading (Recommended)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Direct loading - no adapter needed!
tokenizer = AutoTokenizer.from_pretrained("{repo_name}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{repo_name}", trust_remote_code=True)

# Ask a medical question
question = "What are the main findings about pembrolizumab in melanoma treatment?"
prompt = f"<|user|>\\n{{question}}<|end|>\\n<|assistant|>\\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split('<|assistant|>')[-1].strip()
print(answer)
```

### SageMaker Inference
```python
# inference.py for SageMaker
from transformers import AutoTokenizer, AutoModelForCausalLM

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer

def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    question = input_data["question"]
    prompt = f"<|user|>\\n{{question}}<|end|>\\n<|assistant|>\\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {{"answer": response.split('<|assistant|>')[-1].strip()}}
```

## Training Details

The model was fine-tuned on medical research abstracts focusing on:
- Immunotherapy treatments (pembrolizumab, nivolumab, etc.)
- Clinical trial results and endpoints
- Adverse events reporting and safety profiles
- Biomarker identification and patient selection
- Treatment efficacy measurements and response rates

## Model Versions

- **Adapter version**: `goaiguru/medical-qa-phi3-mini-mac` (9MB - LoRA adapter only)
- **Merged version**: `{repo_name}` (7GB - Full model, this repo)

## Intended Use

This model is designed for educational and research purposes in medical question-answering. It should not be used for clinical decision-making or patient care without proper medical supervision.

## Limitations

- Trained on a limited dataset of 100 ASCO abstracts
- Focused primarily on oncology/immunotherapy research
- May not generalize to all medical domains
- Should not replace professional medical advice

## Citation

If you use this model, please cite:
```
@misc{{medical-qa-phi3-mini-mac-merged,
  title={{Medical Q&A Model - Fine-tuned Phi-3 Mini (Merged)}},
  author={{GoAI Guru}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""

    with open(f"{local_path}/README.md", "w") as f:
        f.write(model_card)
    
    # Push merged model to Hub
    try:
        print(f"\n🚀 Uploading merged model to: {repo_name}")
        print("This may take several minutes (uploading ~7GB)...")
        
        merged_model.push_to_hub(
            repo_name, 
            token=hf_token,
            safe_serialization=True,
            max_shard_size="2GB"  # Split into smaller shards
        )
        tokenizer.push_to_hub(repo_name, token=hf_token)
        
        print(f"✅ Merged model successfully pushed to: https://huggingface.co/{repo_name}")
        print("\n🎯 Ready for production use!")
        print("   - SageMaker inference ✅")
        print("   - Standard transformers ✅") 
        print("   - No adapter loading needed ✅")
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(local_path)
        print(f"\n🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Error pushing merged model to Hub: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Configuration
    ADAPTER_PATH = "./medical_qa_phi3_mac/checkpoint-200"  # Your trained adapter
    HF_TOKEN = "hf_12345"  # Your HF token
    MERGED_REPO_NAME = "goaiguru/medical-qa-phi3-mini-mac-merged"  # Merged model repo
    
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter model not found at: {ADAPTER_PATH}")
        print("Please ensure the model was trained successfully.")
        return
    
    print("🏥 Medical Q&A Model - Merge & Upload to HuggingFace")
    print("=" * 60)
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Target repo: {MERGED_REPO_NAME}")
    print(f"Expected size: ~7GB")
    print("=" * 60)
    
    # Confirm before proceeding (large upload)
    confirm = input("Proceed with merge and upload? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return
    
    try:
        merge_and_upload_model(ADAPTER_PATH, MERGED_REPO_NAME, HF_TOKEN)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! You now have both versions:")
        print("   📦 Adapter: goaiguru/medical-qa-phi3-mini-mac")
        print("   🚀 Merged:  goaiguru/medical-qa-phi3-mini-mac-merged")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

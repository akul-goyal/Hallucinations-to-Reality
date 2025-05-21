from huggingface_hub import snapshot_download
import os

# Download Mistral 7B Instruct
print("Downloading Mistral 7B Instruct model from Hugging Face...")
model_path = snapshot_download(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    local_dir='./models/mistral-7b-instruct'
)
print(f"Model downloaded successfully to {model_path}!")
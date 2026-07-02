import os
import sys

# Set local cache path
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

def download():
    repo = "Lykon/dreamshaper-xl-v2-turbo"
    print(f"1. Downloading config files, tokenizers, and schedulers for {repo}...")
    snapshot_download(
        repo_id=repo,
        ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
        max_workers=1
    )
    
    print(f"2. Downloading only the FP16 model weights for {repo} (saves 50% bandwidth and RAM)...")
    snapshot_download(
        repo_id=repo,
        allow_patterns=[
            "*.fp16.safetensors",
            "model.fp16.safetensors",
            "diffusion_pytorch_model.fp16.safetensors"
        ],
        max_workers=1
    )
    print("Download completed successfully!")

if __name__ == "__main__":
    download()

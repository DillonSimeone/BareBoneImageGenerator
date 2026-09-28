import os
import sys
import shutil

# Set local cache path
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            if "=" in _line and not _line.strip().startswith("#"):
                _k, _v = _line.strip().split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from huggingface_hub import snapshot_download

MODEL_REPOS = {
    "dreamshaper_turbo": "Lykon/dreamshaper-xl-v2-turbo",
    "flux_schnell": "black-forest-labs/FLUX.1-schnell",
    "flux_dev": "black-forest-labs/FLUX.1-dev",
    "z_image_turbo": "Tongyi-MAI/Z-Image-Turbo",
    "realvis_xl": "SG161222/RealVisXL_V4.0",
    "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9",
    "pony_xl": "Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime",
    "animagine_xl": "cagliostrolab/animagine-xl-3.1",
    "qwen_image_edit_rapid_aio": "Phr00t/Qwen-Image-Edit-Rapid-AIO"
}

def download(model_key="dreamshaper_turbo", variant="v23/Qwen-Rapid-AIO-SFW-v23.safetensors"):
    repo = MODEL_REPOS.get(model_key, model_key)
    
    if "qwen-image-edit-rapid-aio" in repo.lower():
        print(f"Downloading base Qwen-Image-Edit-2511 configs, tokenizer, and processor...")
        snapshot_download(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.onnx", "*.pt"],
            max_workers=2
        )
        # Also download VAE and text encoder configs
        snapshot_download(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            allow_patterns=["vae/*", "text_encoder/config.json", "transformer/config.json"],
            ignore_patterns=["text_encoder/*.safetensors", "transformer/*.safetensors"],
            max_workers=2
        )
        print(f"Downloading Qwen-Image-Edit-Rapid-AIO checkpoint: {variant}...")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
            filename=variant
        )
        # Link or copy into checkpoints directory
        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        dest_filename = os.path.basename(variant)
        dest_path = os.path.join(checkpoints_dir, dest_filename)
        if not os.path.exists(dest_path):
            print(f"Linking checkpoint to {dest_path}...")
            try:
                os.link(ckpt_path, dest_path)
            except Exception:
                shutil.copyfile(ckpt_path, dest_path)
        print(f"Qwen-Image-Edit-Rapid-AIO ready at {dest_path}!")
        return

    print(f"1. Downloading config files, tokenizers, and schedulers for {repo}...")
    snapshot_download(
        repo_id=repo,
        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.onnx"],
        max_workers=2
    )
    
    print(f"2. Downloading model weights for {repo}...")
    snapshot_download(
        repo_id=repo,
        allow_patterns=["*.safetensors"],
        max_workers=2
    )
    print(f"Download for {repo} completed successfully!")

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "dreamshaper_turbo"
    var = sys.argv[2] if len(sys.argv) > 2 else "v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
    download(key, var)

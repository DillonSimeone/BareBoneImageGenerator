import os
# Configure Hugging Face cache directory locally to avoid using C: drive
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import uuid
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import numpy as np

app = FastAPI(title="Antigravity Image Gen WebUI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for generated images
OUTPUT_DIR = os.path.join("static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for active pipeline and model ID
current_pipe = None
current_model_id = None

MODELS = {
    "dreamshaper_turbo": {
        "id": "Lykon/dreamshaper-xl-v2-turbo",
        "name": "DreamShaper XL Turbo (Fast & Stylized)",
        "description": "High-quality, fast SDXL turbo model. Recommended 4-8 steps, CFG scale 1.5 - 2.5.",
        "default_steps": 6,
        "default_cfg": 2.0
    },
    "realvis_xl": {
        "id": "SG161222/RealVisXL_V4.0",
        "name": "RealVisXL V4.0 (Photorealistic)",
        "description": "Stunning photorealistic results. Recommended 20-30 steps, CFG scale 5.0 - 7.0.",
        "default_steps": 25,
        "default_cfg": 6.0
    },
    "juggernaut_xl": {
        "id": "RunDiffusion/Juggernaut-XL-v9",
        "name": "Juggernaut XL V9 (Ultra-Realism & Cinematic)",
        "description": "The gold standard for photorealistic humans, landscapes, and cinematic lighting. Recommended 25 steps, CFG scale 5.0 - 6.0.",
        "default_steps": 25,
        "default_cfg": 5.5
    },
    "pony_xl": {
        "id": "Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime",
        "name": "Pony Diffusion V6 XL (Uncensored Illustrative)",
        "description": "Legendary uncensored character/anime fine-tune. Tip: Start prompt with 'score_9, score_8_up, score_7_up, [your prompt]'. Recommended 25 steps, CFG scale 5.0 - 7.0.",
        "default_steps": 25,
        "default_cfg": 6.0
    },
    "animagine_xl": {
        "id": "cagliostrolab/animagine-xl-3.1",
        "name": "Animagine XL V3.1 (Anime & Illustrative)",
        "description": "High-quality anime illustration model. Recommended 25-30 steps, CFG scale 5.0 - 7.0.",
        "default_steps": 28,
        "default_cfg": 5.5
    }
}

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 6
    cfg_scale: float = 2.0
    width: int = 512
    height: int = 512
    seed: int = -1
    model_key: str = "dreamshaper_turbo"

def ensure_model_downloaded(model_id: str) -> bool:
    from huggingface_hub import snapshot_download, list_repo_files
    print(f"Ensuring model is downloaded locally: {model_id}...")
    has_fp16 = False
    try:
        # Step 1: Configs
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
            max_workers=1
        )
        
        # Step 2: Check repo files for FP16 weights
        files = list_repo_files(model_id)
        has_fp16 = any(".fp16.safetensors" in f for f in files)
        
        if has_fp16:
            print("Downloading optimized FP16 model weights...")
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "*.fp16.safetensors",
                    "model.fp16.safetensors",
                    "diffusion_pytorch_model.fp16.safetensors"
                ],
                max_workers=1
            )
        else:
            print("No FP16 weights found. Downloading standard safetensors weights...")
            snapshot_download(
                repo_id=model_id,
                allow_patterns=["*.safetensors"],
                max_workers=1
            )
    except Exception as e:
        print(f"Error checking/downloading model {model_id}: {e}")
    return has_fp16

def get_pipeline(model_key: str):
    global current_pipe, current_model_id
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model key")
    
    model_info = MODELS[model_key]
    model_id = model_info["id"]
    
    if current_model_id != model_id or current_pipe is None:
        print(f"Loading pipeline for: {model_id}...")
        
        # Dynamically ensure all files are downloaded before loading and detect FP16 support
        has_fp16 = ensure_model_downloaded(model_id)
        
        # Clean up old pipeline memory
        if current_pipe is not None:
            del current_pipe
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine optimal torch dtype (float16 for RTX 3070 Ti)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Optimize PyTorch compiler & math settings for Ampere GPUs (RTX 30-series)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        try:
            # We use AutoPipelineForText2Image to load
            current_pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype,
                variant="fp16" if has_fp16 else None,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                # Enable model CPU offloading: offloads inactive components (U-Net, VAE, Text Encoders)
                # to CPU RAM, keeping VRAM usage under 5GB and completely preventing Windows shared memory paging lag.
                current_pipe.enable_model_cpu_offload()
                
                # Enable VAE slicing (saves VRAM on VAE decoding using the modern VAE-specific API)
                current_pipe.vae.enable_slicing()
            
            current_model_id = model_id
            print("Pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")
            
    return current_pipe

@app.get("/api/models")
def get_models():
    return MODELS

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    pipe = get_pipeline(req.model_key)
    
    # Handle seed
    if req.seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = req.seed
        
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    print(f"Generating image. Prompt: '{req.prompt}', Steps: {req.steps}, Seed: {seed}")
    
    import time
    start_time = time.perf_counter()
    
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg_scale,
                width=req.width,
                height=req.height,
                generator=generator
            )
        
        generation_time = time.perf_counter() - start_time
        it_per_sec = req.steps / generation_time if generation_time > 0 else 0
        
        image = result.images[0]
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        
        # Save metadata to a text file next to the image
        meta_filepath = filepath.replace(".png", ".txt")
        with open(meta_filepath, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {req.prompt}\n")
            f.write(f"Negative Prompt: {req.negative_prompt}\n")
            f.write(f"Model: {MODELS[req.model_key]['name']}\n")
            f.write(f"Steps: {req.steps}\n")
            f.write(f"CFG Scale: {req.cfg_scale}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Dimensions: {req.width}x{req.height}\n")
            f.write(f"Time Taken: {generation_time:.2f} seconds\n")
            f.write(f"Speed: {it_per_sec:.2f} it/s\n")

        return {
            "image_url": f"/static/outputs/{filename}",
            "seed": seed,
            "filename": filename,
            "time_taken": f"{generation_time:.2f}s",
            "speed": f"{it_per_sec:.2f} it/s"
        }
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history():
    images = []
    if not os.path.exists(OUTPUT_DIR):
        return images
        
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".png"):
            # Try to read associated metadata if it exists
            meta_path = os.path.join(OUTPUT_DIR, file.replace(".png", ".txt"))
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            if ":" in line:
                                k, v = line.split(":", 1)
                                meta[k.strip().lower().replace(" ", "_")] = v.strip()
                except Exception:
                    pass
            
            images.append({
                "image_url": f"/static/outputs/{file}",
                "filename": file,
                "meta": meta
            })
            
    # Sort by creation time (newest first)
    images.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x["filename"])), reverse=True)
    return images

# Pre-load default model on startup to avoid request timeouts during load
@app.on_event("startup")
def preload_model():
    print("===================================================")
    print(" Pre-loading DreamShaper XL Turbo to GPU...")
    print(" This ensures instant generation and prevents timeouts.")
    print("===================================================")
    try:
        get_pipeline("dreamshaper_turbo")
        print(" Model loaded successfully and ready!")
        
        # Auto-open browser when model is fully loaded and ready
        import webbrowser
        webbrowser.open("http://127.0.0.1:7860/")
    except Exception as e:
        print(f" Warning: Could not pre-load model on startup: {e}")
        import webbrowser
        webbrowser.open("http://127.0.0.1:7860/")
    print("===================================================")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)

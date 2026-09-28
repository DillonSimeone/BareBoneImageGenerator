import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv(os.path.join(_BASE_DIR, ".env"))

# Configure Hugging Face cache directory locally to avoid using C: drive
os.environ["HF_HOME"] = os.path.join(_BASE_DIR, "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"
# Persist compiled Inductor/Triton kernels in the project so restarts skip most of the compile warmup
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_BASE_DIR, ".inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_BASE_DIR, ".inductor_cache", "triton"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

import uuid
import torch
import gc
import json
import random
import time
import psutil
import shutil
import warnings
import webbrowser
import subprocess
import io
import base64
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DiffusionPipeline,
    StableVideoDiffusionPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DEISMultistepScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    TCDScheduler,
    AutoencoderKL,
    AutoencoderTiny
)
import numpy as np
import imageio
from PIL import Image

# Hardware Acceleration & Precision Settings for Modern NVIDIA GPUs (RTX 30/40/50 Series on Windows 11)
if torch.cuda.is_available():
    # Enable TensorFloat-32 (TF32) for fast float32 matmuls on Ampere, Ada Lovelace, and Blackwell GPUs
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Pick the fastest cuDNN conv algorithms for the UNet/VAE (re-tuned once per new resolution)
    torch.backends.cudnn.benchmark = True

# Suppress known deprecation & benign warnings
warnings.filterwarnings("ignore", message=".*upcast_vae.*", category=FutureWarning)

class InductorSMFilter(logging.Filter):
    def filter(self, record):
        return "Not enough SMs to use max_autotune_gemm mode" not in record.getMessage()

logging.getLogger("torch._inductor.utils").addFilter(InductorSMFilter())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load default model on startup to avoid request timeouts during load
    run_on_gpu_thread(preload_model)
    yield

app = FastAPI(title="Antigravity Image Gen WebUI", lifespan=lifespan)

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

# Directory for local LoRAs
LORAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
os.makedirs(LORAS_DIR, exist_ok=True)

# Directory for manual model checkpoints
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def link_cached_models():
    hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache", "hub")
    if not os.path.exists(hub_dir):
        return
    for item in os.listdir(hub_dir):
        if item.startswith("models--"):
            parts = item.split("--")
            model_name = parts[-1] if len(parts) >= 3 else item
            model_dir = os.path.join(hub_dir, item)
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    dest_path = os.path.join(CHECKPOINTS_DIR, model_name)
                    if not os.path.exists(dest_path):
                        print(f"Creating directory junction: {dest_path} -> {snapshot_path}")
                        try:
                            subprocess.run(
                                ["cmd", "/c", "mklink", "/J", dest_path, snapshot_path],
                                check=True,
                                capture_output=True
                            )
                        except Exception as e:
                            print(f"Failed to create junction for {model_name}: {e}")

link_cached_models()


_cached_models = None

def get_all_models(force_refresh=False):
    global _cached_models
    if _cached_models is not None and not force_refresh:
        return _cached_models

    all_models = dict(MODELS)
    if os.path.exists(CHECKPOINTS_DIR):
        builtin_repo_names = {
            os.path.basename(model["id"]).lower()
            for model in MODELS.values()
            if "/" in model.get("id", "")
        }
        top_level_checkpoint_stems = {
            os.path.splitext(file)[0].lower()
            for file in os.listdir(CHECKPOINTS_DIR)
            if os.path.isfile(os.path.join(CHECKPOINTS_DIR, file))
            and file.lower().endswith(".safetensors")
        }
        
        for root, dirs, files in os.walk(CHECKPOINTS_DIR):
            # Check if this folder contains model_index.json (Diffusers model)
            if "model_index.json" in files:
                rel_path = os.path.relpath(root, CHECKPOINTS_DIR)
                rel_path = rel_path.replace("\\", "/")
                folder_name = os.path.basename(root).lower()
                real_root = os.path.realpath(root)
                is_cache_junction = os.path.commonpath([os.path.abspath(real_root), os.environ["HF_HOME"]]) == os.environ["HF_HOME"]
                
                # Hide cache junctions for built-in Hugging Face models and generated
                # Diffusers folders that mirror a top-level checkpoint file.
                if is_cache_junction or folder_name in builtin_repo_names or folder_name in top_level_checkpoint_stems:
                    dirs.clear()
                    continue
                
                model_key = f"local_dir_{rel_path.replace('/', '_').replace('.', '_')}"
                is_z_image_cp = "z_image_turbo" in rel_path.lower()
                is_qwen_cp = "qwen" in rel_path.lower()
                display_name = f"Qwen Edit: {rel_path}" if is_qwen_cp else f"Local Folder: {rel_path}"
                all_models[model_key] = {
                    "id": root,
                    "name": display_name,
                    "description": f"Diffusers model directory loaded from checkpoints/{rel_path}.",
                    "default_steps": 6 if is_qwen_cp else (8 if is_z_image_cp else 25),
                    "default_cfg": 1.0 if (is_qwen_cp or is_z_image_cp) else 6.0,
                    "is_local": True,
                    "is_single_file": False
                }
                # Clear dirs to prevent walking subfolders of this diffusers model
                dirs.clear()
                continue
                
            for file in files:
                if file.lower().endswith(".safetensors") and "model.fp16.safetensors" not in file and "diffusion_pytorch_model.fp16.safetensors" not in file:
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, CHECKPOINTS_DIR)
                    rel_path = rel_path.replace("\\", "/")
                    model_key = f"local_file_{rel_path.replace('/', '_').replace('.', '_')}"
                    is_z_image_cp = "z_image_turbo" in file.lower()
                    is_qwen_cp = "qwen" in file.lower()
                    display_name = f"Qwen Edit Checkpoint: {rel_path}" if is_qwen_cp else f"Local Checkpoint: {rel_path}"
                    all_models[model_key] = {
                        "id": filepath,
                        "name": display_name,
                        "description": f"Manually added checkpoint checkpoints/{rel_path}.",
                        "default_steps": 6 if is_qwen_cp else (8 if is_z_image_cp else 25),
                        "default_cfg": 1.0 if (is_qwen_cp or is_z_image_cp) else 6.0,
                        "is_local": True,
                        "is_single_file": True
                    }
    _cached_models = all_models
    return all_models

SAMPLERS = {
    "euler": {
        "name": "Euler",
        "class": EulerDiscreteScheduler,
        "kwargs": {}
    },
    "euler_karras": {
        "name": "Euler Karras",
        "class": EulerDiscreteScheduler,
        "kwargs": {"use_karras_sigmas": True}
    },
    "euler_a": {
        "name": "Euler Ancestral (Euler a)",
        "class": EulerAncestralDiscreteScheduler,
        "kwargs": {}
    },
    "heun": {
        "name": "Heun",
        "class": HeunDiscreteScheduler,
        "kwargs": {}
    },
    "dpm_1s": {
        "name": "DPM++ 1S",
        "class": DPMSolverSinglestepScheduler,
        "kwargs": {}
    },
    "dpm_2m": {
        "name": "DPM++ 2M",
        "class": DPMSolverMultistepScheduler,
        "kwargs": {}
    },
    "dpm_2m_karras": {
        "name": "DPM++ 2M Karras",
        "class": DPMSolverMultistepScheduler,
        "kwargs": {"use_karras_sigmas": True}
    },
    "dpm_2m_sde": {
        "name": "DPM++ 2M SDE",
        "class": DPMSolverMultistepScheduler,
        "kwargs": {"algorithm_type": "sde-dpmsolver++"}
    },
    "dpm_2m_sde_karras": {
        "name": "DPM++ 2M SDE Karras",
        "class": DPMSolverMultistepScheduler,
        "kwargs": {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}
    },
    "dpm_3m_sde_karras": {
        "name": "DPM++ 3M SDE Karras",
        "class": DPMSolverMultistepScheduler,
        "kwargs": {"solver_order": 3, "use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}
    },
    "kdpm2": {
        "name": "KDPM2",
        "class": KDPM2DiscreteScheduler,
        "kwargs": {}
    },
    "kdpm2_a": {
        "name": "KDPM2 Ancestral",
        "class": KDPM2AncestralDiscreteScheduler,
        "kwargs": {}
    },
    "deis": {
        "name": "DEIS",
        "class": DEISMultistepScheduler,
        "kwargs": {}
    },
    "pndm": {
        "name": "PNDM",
        "class": PNDMScheduler,
        "kwargs": {}
    },
    "lms": {
        "name": "LMS",
        "class": LMSDiscreteScheduler,
        "kwargs": {}
    },
    "lms_karras": {
        "name": "LMS Karras",
        "class": LMSDiscreteScheduler,
        "kwargs": {"use_karras_sigmas": True}
    },
    "ddim": {
        "name": "DDIM",
        "class": DDIMScheduler,
        "kwargs": {}
    },
    "unipc": {
        "name": "UniPC",
        "class": UniPCMultistepScheduler,
        "kwargs": {}
    },
    "lcm": {
        "name": "LCM (Turbo / DMD2)",
        "class": LCMScheduler,
        "kwargs": {}
    },
    "tcd": {
        "name": "TCD (Turbo)",
        "class": TCDScheduler,
        "kwargs": {}
    }
}

# Global variables for active pipeline, model ID, and performance mode
current_pipe = None
current_model_id = None
current_performance_mode = None
current_torch_compile = None
current_turbo_mode = None
_compile_warmup_done = False
_last_compiled_shape = None  # Track (width, height) to detect resolution changes under torch.compile

# Per-pipeline caches (reset whenever the pipeline is reloaded)
_loaded_adapters = {}      # resolved LoRA path -> adapter name
_embed_cache = {}          # (prompt, negative, cfg, lora signature) -> SDXL prompt embeddings
_EMBED_CACHE_MAX = 32
_preview_vae = None        # AutoencoderTiny for live step previews
_vae_timing = {"decode": 0.0}

# Live progress shared with /api/progress and /api/cancel
_progress = {"active": False, "step": 0, "total": 0, "preview": None, "preview_id": 0}
_cancel_requested = False

TURBO_LORA_REPO = "tianweiy/DMD2"
TURBO_LORA_FILE = "dmd2_sdxl_4step_lora_fp16.safetensors"
SDXL_FP16_VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"

CONFIG_PATH = os.path.join(_BASE_DIR, "config.json")
CONFIG_DEFAULTS = {
    "preload_model": "dreamshaper_turbo",
    "performance_mode": "max_speed",
    "torch_compile": False,
    "turbo_mode": False
}

def read_config():
    cfg = dict(CONFIG_DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"Warning reading config.json: {e}")
    return cfg

class GenerationCancelled(Exception):
    pass

# torch.compile's CUDA-graph trees keep per-thread state, so recording graphs from FastAPI's
# rotating worker threads fails with an AssertionError. All GPU work runs on this one thread.
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")

def run_on_gpu_thread(fn, *args, **kwargs):
    if threading.current_thread().name.startswith("gpu-worker"):
        return fn(*args, **kwargs)
    return _gpu_executor.submit(fn, *args, **kwargs).result()

MODELS = {
    "flux_lite": {
        "id": "Freepik/FLUX.1-Lite-8B-Alpha",
        "name": "FLUX.1 Lite 8B (Un-gated Open FLUX)",
        "description": "Freepik's open-access 8B FLUX model. 100% un-gated: works instantly out of the box with zero Hugging Face login or token required. Recommended 4-8 steps, CFG scale 3.5.",
        "default_steps": 4,
        "default_cfg": 3.5
    },
    "flux_uncensored": {
        "id": "lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
        "name": "FLUX.1 Uncensored (Un-gated & Zero Restrictions)",
        "description": "Uncensored FLUX model fine-tune. Completely un-gated: no Hugging Face login or token required. Recommended 8 steps, CFG scale 2.5.",
        "default_steps": 8,
        "default_cfg": 2.5
    },
    "flux_schnell": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "name": "FLUX.1 Schnell (Official BFL 12B - 4 Steps)",
        "description": "Official Black Forest Labs 12B DiT model. Note: Requires Hugging Face login (huggingface-cli login) to access gated repo. Recommended 4 steps, CFG scale 1.0.",
        "default_steps": 4,
        "default_cfg": 1.0
    },
    "flux_dev": {
        "id": "black-forest-labs/FLUX.1-dev",
        "name": "FLUX.1 Dev (Pro 12B DiT - 28 Steps)",
        "description": "Black Forest Labs flagship 12B DiT model. Highest quality photorealism, fine details, and text rendering. Recommended 28 steps, CFG scale 3.5.",
        "default_steps": 28,
        "default_cfg": 3.5
    },
    "dreamshaper_turbo": {
        "id": "Lykon/dreamshaper-xl-v2-turbo",
        "name": "DreamShaper XL Turbo (Fast & Stylized)",
        "description": "High-quality, fast SDXL turbo model. Recommended 4-8 steps, CFG scale 1.5 - 2.5.",
        "default_steps": 6,
        "default_cfg": 2.0
    },
    "z_image_turbo": {
        "id": "Tongyi-MAI/Z-Image-Turbo",
        "name": "Z-Image-Turbo (Sub-Second 6B DiT)",
        "description": "Alibaba Tongyi Lab's efficient 6B single-stream DiT model. Very fast, excellent text rendering. Recommended 8 steps, CFG scale 1.0.",
        "default_steps": 8,
        "default_cfg": 1.0
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
    },
    "qwen_image_edit_rapid_aio": {
        "id": "Phr00t/Qwen-Image-Edit-Rapid-AIO",
        "name": "Qwen Image Edit Rapid AIO (Fast Edit & T2I)",
        "description": "Phr00t's All-In-One accelerated Qwen Image Edit model. 4-8 steps, CFG 1.0. Supports text-to-image and instruct image editing / img2img.",
        "default_steps": 6,
        "default_cfg": 1.0
    }
}

class ConfigRequest(BaseModel):
    preload_model: str = ""
    performance_mode: str = "max_speed"
    torch_compile: bool = False
    turbo_mode: bool = False

class LoraRequestConfig(BaseModel):
    name: str
    scale: float = 0.8
    enabled: bool = True

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 6
    cfg_scale: float = 2.0
    width: int = 768
    height: int = 768
    seed: int = -1
    model_key: str = "dreamshaper_turbo"
    sampler_key: str = "euler_a"
    loras: list[LoraRequestConfig] = []
    init_image: Optional[str] = None
    strength: float = 0.75
    edit_mode: str = "edit"
    cfg_cutoff: float = 1.0  # fraction of steps that use CFG; 1.0 = always (off)
    live_preview: bool = True

class VideoRequest(BaseModel):
    image_filename: str
    num_frames: int = 25
    fps: int = 12
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    seed: int = -1

# SVD Pipeline cache
svd_pipe = None

def _reset_pipeline_caches():
    global _preview_vae
    _loaded_adapters.clear()
    _embed_cache.clear()
    _preview_vae = None

def get_svd_pipeline():
    global svd_pipe, current_pipe, current_model_id, current_performance_mode, current_torch_compile, current_turbo_mode, _compile_warmup_done
    if current_pipe is not None:
        _reset_pipeline_caches()
        current_turbo_mode = None
        print("Evicting text-to-image pipeline to ensure dedicated VRAM for SVD...")
        try:
            if hasattr(current_pipe, "remove_all_hooks"):
                current_pipe.remove_all_hooks()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_pipe.to("cpu")
        except Exception as e:
            print(f"Warning during image pipeline offload: {e}")
        del current_pipe
        current_pipe = None
        current_model_id = None
        current_performance_mode = None
        current_torch_compile = None
        _compile_warmup_done = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Image pipeline evicted. VRAM freed.")
    
    if svd_pipe is None:
        print("Loading Stable Video Diffusion (SVD) pipeline...")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            # Use CPU offloading to keep peak VRAM manageable on 16GB cards
            svd_pipe.enable_model_cpu_offload()
            if hasattr(svd_pipe, "vae") and svd_pipe.vae is not None:
                try:
                    svd_pipe.vae.enable_slicing()
                except (NotImplementedError, AttributeError):
                    print("SVD VAE does not support slicing (temporal decoder), skipping.")
                try:
                    svd_pipe.vae.enable_tiling()
                except (NotImplementedError, AttributeError):
                    print("SVD VAE does not support tiling (temporal decoder), skipping.")
        print("SVD pipeline loaded successfully with CPU offloading!")
    return svd_pipe

def release_svd_pipeline():
    """Free SVD pipeline from memory so the image pipeline can reload."""
    global svd_pipe
    if svd_pipe is not None:
        print("Releasing SVD pipeline from memory...")
        try:
            if hasattr(svd_pipe, "remove_all_hooks"):
                svd_pipe.remove_all_hooks()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                svd_pipe.to("cpu")
        except Exception:
            pass
        del svd_pipe
        svd_pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("SVD pipeline released.")

def ensure_model_downloaded(model_id: str) -> bool:
    from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
    print(f"Ensuring model is downloaded locally: {model_id}...")
    has_fp16 = False
    try:
        if "qwen-image-edit-rapid-aio" in model_id.lower():
            print("Ensuring base Qwen-Image-Edit-2511 configs, tokenizer, and processor...")
            snapshot_download(
                repo_id="Qwen/Qwen-Image-Edit-2511",
                ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.onnx", "*.pt"],
                max_workers=2
            )
            snapshot_download(
                repo_id="Qwen/Qwen-Image-Edit-2511",
                allow_patterns=["vae/*", "text_encoder/config.json", "transformer/config.json"],
                ignore_patterns=["text_encoder/*.safetensors", "transformer/*.safetensors"],
                max_workers=2
            )
            # Check if checkpoint already exists in checkpoints dir
            has_local_ckpt = any("qwen" in f.lower() and f.endswith(".safetensors") for f in os.listdir(CHECKPOINTS_DIR))
            if not has_local_ckpt:
                print("Downloading default Qwen-Rapid-AIO checkpoint (v23 SFW)...")
                ckpt_path = hf_hub_download(
                    repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                    filename="v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
                )
                dest_path = os.path.join(CHECKPOINTS_DIR, "Qwen-Rapid-AIO-SFW-v23.safetensors")
                if not os.path.exists(dest_path):
                    try:
                        os.link(ckpt_path, dest_path)
                    except Exception:
                        shutil.copyfile(ckpt_path, dest_path)
            return False

        # Step 1: Configs
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.onnx", "*.pt"],
            max_workers=4
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
                max_workers=4
            )
        else:
            print("No FP16 weights found. Downloading standard safetensors weights...")
            has_subfolders = any(f.endswith(".safetensors") and "/" in f for f in files)
            ignore_list = ["*.bin", "*.pth", "*.ckpt", "*.onnx"]
            if has_subfolders:
                # Exclude root single-file checkpoints when structured diffusers folders exist
                ignore_list.extend(["flux.1-lite-8B-alpha.safetensors", "*.safetensors"])
                allow_list = ["*/*.safetensors", "*/*/*.safetensors"]
            else:
                allow_list = ["*.safetensors"]

            snapshot_download(
                repo_id=model_id,
                allow_patterns=allow_list,
                ignore_patterns=ignore_list,
                max_workers=4
            )
    except Exception as e:
        print(f"Error checking/downloading model {model_id}: {e}")
    return has_fp16

def shard_safetensors_manually(source_path, output_dir, max_shard_size_gb=4.0):
    import torch
    import safetensors.torch
    import gc
    import json
    import os
    import psutil
    
    proc = psutil.Process(os.getpid())
    print(f"[Memory Debug] System VM: {psutil.virtual_memory()}")
    print(f"[Memory Debug] Process RSS: {proc.memory_info().rss / (1024**2):.2f} MB, VMS: {proc.memory_info().vms / (1024**2):.2f} MB")
    
    print(f"Loading checkpoint {source_path} to convert and shard...")
    state_dict = safetensors.torch.load_file(source_path, device="cpu")
    
    print("Converting state dict keys and splitting tensors...")
    mapped_state_dict = {}
    for key, tensor in state_dict.items():
        # 1. Map x_embedder
        if key == "x_embedder.weight":
            mapped_key = "all_x_embedder.2-1.weight"
        elif key == "x_embedder.bias":
            mapped_key = "all_x_embedder.2-1.bias"
        # 2. Map final_layer
        elif key.startswith("final_layer."):
            mapped_key = key.replace("final_layer.", "all_final_layer.2-1.")
        else:
            mapped_key = key

        # 3. Map attention layers (k_norm -> norm_k, q_norm -> norm_q, out -> to_out.0)
        if ".attention." in mapped_key or ".attention_norm" in mapped_key:
            mapped_key = mapped_key.replace(".attention.k_norm.", ".attention.norm_k.")
            mapped_key = mapped_key.replace(".attention.q_norm.", ".attention.norm_q.")
            mapped_key = mapped_key.replace(".attention.out.weight", ".attention.to_out.0.weight")
            mapped_key = mapped_key.replace(".attention.out.bias", ".attention.to_out.0.bias")

        # 4. Handle QKV splitting
        if ".attention.qkv.weight" in mapped_key:
            # Split qkv.weight [11520, 3840] into q, k, v of size [3840, 3840]
            q, k, v = tensor.chunk(3, dim=0)
            mapped_state_dict[mapped_key.replace(".attention.qkv.weight", ".attention.to_q.weight")] = q.clone()
            mapped_state_dict[mapped_key.replace(".attention.qkv.weight", ".attention.to_k.weight")] = k.clone()
            mapped_state_dict[mapped_key.replace(".attention.qkv.weight", ".attention.to_v.weight")] = v.clone()
        elif ".attention.qkv.bias" in mapped_key:
            q, k, v = tensor.chunk(3, dim=0)
            mapped_state_dict[mapped_key.replace(".attention.qkv.bias", ".attention.to_q.bias")] = q.clone()
            mapped_state_dict[mapped_key.replace(".attention.qkv.bias", ".attention.to_k.bias")] = k.clone()
            mapped_state_dict[mapped_key.replace(".attention.qkv.bias", ".attention.to_v.bias")] = v.clone()
        else:
            mapped_state_dict[mapped_key] = tensor

    del state_dict
    gc.collect()
    
    os.makedirs(output_dir, exist_ok=True)
    max_shard_bytes = int(max_shard_size_gb * 1024 * 1024 * 1024)
    
    shards = []
    current_shard = {}
    current_size = 0
    
    for key, tensor in mapped_state_dict.items():
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_bytes
        
    if current_shard:
        shards.append(current_shard)
        
    total_shards = len(shards)
    weight_map = {}
    
    for idx, shard_dict in enumerate(shards, start=1):
        shard_name = f"diffusion_pytorch_model-{idx:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        
        print(f"Saving shard {idx}/{total_shards}: {shard_name}...")
        safetensors.torch.save_file(shard_dict, shard_path)
        
        for key in shard_dict.keys():
            weight_map[key] = shard_name
            
        del shard_dict
        gc.collect()
        
    index_data = {
        "metadata": {"total_size": sum(os.path.getsize(os.path.join(output_dir, f"diffusion_pytorch_model-{i:05d}-of-{total_shards:05d}.safetensors")) for i in range(1, total_shards + 1))},
        "weight_map": weight_map
    }
    with open(os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as idx_f:
        json.dump(index_data, idx_f, indent=2)
    print("Sharding and conversion completed successfully!")

def _swap_in_fp16_vae(pipe):
    """Replace the SDXL VAE with the fp16-safe finetune so decoding doesn't need an fp32 upcast."""
    try:
        print(f"Loading fp16-safe SDXL VAE ({SDXL_FP16_VAE_REPO})...")
        pipe.vae = AutoencoderKL.from_pretrained(SDXL_FP16_VAE_REPO, torch_dtype=torch.float16, use_safetensors=True)
    except Exception as e:
        print(f"Could not load fp16-safe VAE, keeping checkpoint VAE (fp32 upcast): {e}")

def _apply_turbo_lora(pipe):
    """Fuse the DMD2 4-step distillation LoRA into the UNet (4-8 steps, CFG 1, LCM sampler)."""
    try:
        local_file = next(
            (f for f in os.listdir(LORAS_DIR) if f.lower().startswith("dmd2") and f.lower().endswith(".safetensors")),
            None
        )
        if local_file:
            print(f"Turbo mode: fusing local DMD2 LoRA loras/{local_file}...")
            pipe.load_lora_weights(LORAS_DIR, weight_name=local_file, adapter_name="turbo")
        else:
            print(f"Turbo mode: fusing DMD2 LoRA from {TURBO_LORA_REPO}...")
            pipe.load_lora_weights(TURBO_LORA_REPO, weight_name=TURBO_LORA_FILE, adapter_name="turbo")
        pipe.fuse_lora(lora_scale=1.0, adapter_names=["turbo"])
        # Drop the PEFT wrapper layers; the fused weights stay baked into the UNet
        pipe.unload_lora_weights()
        print("Turbo LoRA fused. Use the LCM sampler, 4-8 steps, CFG 1.0.")
    except Exception as e:
        print(f"Warning: failed to apply Turbo LoRA: {e}")

def _load_preview_vae(is_sdxl):
    global _preview_vae
    repo = "madebyollin/taesdxl" if is_sdxl else "madebyollin/taesd"
    try:
        _preview_vae = AutoencoderTiny.from_pretrained(repo, torch_dtype=torch.float16).to("cuda").eval()
    except Exception as e:
        print(f"Live preview decoder unavailable ({repo}): {e}")
        _preview_vae = None

def _install_vae_timer(pipe):
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    inner_decode = vae.decode

    def timed_decode(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = inner_decode(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _vae_timing["decode"] += time.perf_counter() - t0
        return out

    vae.decode = timed_decode

def _resolve_lora_path(name):
    local_path = os.path.join(LORAS_DIR, name)
    if not os.path.exists(local_path):
        local_path = os.path.join(LORAS_DIR, name.replace("/", os.sep))
    return local_path if os.path.exists(local_path) else name

def apply_loras(pipe, active_loras):
    """Load each LoRA once per pipeline and switch between them with set_adapters instead of reloading from disk."""
    names, weights = [], []
    for lora in active_loras:
        load_path = _resolve_lora_path(lora.name)
        adapter_name = _loaded_adapters.get(load_path)
        if adapter_name is None:
            adapter_name = f"lora_{len(_loaded_adapters)}"
            print(f"Loading LoRA: {load_path} as '{adapter_name}' (cached for later requests)")
            try:
                if os.path.isfile(load_path):
                    pipe.load_lora_weights(os.path.dirname(load_path), weight_name=os.path.basename(load_path), adapter_name=adapter_name)
                else:
                    pipe.load_lora_weights(load_path, adapter_name=adapter_name)
            except Exception as e:
                print(f"Error loading LoRA {lora.name}: {e}")
                continue
            _loaded_adapters[load_path] = adapter_name
        names.append(adapter_name)
        weights.append(lora.scale)

    if not _loaded_adapters:
        return ()
    try:
        if names:
            pipe.enable_lora()
            pipe.set_adapters(names, adapter_weights=weights)
        else:
            pipe.disable_lora()
    except Exception as e:
        print(f"Error setting active adapters {names}: {e}")
    return tuple(zip(names, weights))

def encode_sdxl_prompt(pipe, prompt, negative_prompt, do_cfg, lora_signature):
    """Cache SDXL text-encoder output so repeated prompts (seed sweeps, re-rolls) skip both CLIP encoders."""
    key = (prompt, negative_prompt, do_cfg, lora_signature)
    cached = _embed_cache.get(key)
    if cached is not None:
        return cached
    pe, npe, ppe, nppe = pipe.encode_prompt(
        prompt=prompt,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt
    )
    embeds = {
        "prompt_embeds": pe,
        "negative_prompt_embeds": npe,
        "pooled_prompt_embeds": ppe,
        "negative_pooled_prompt_embeds": nppe
    }
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[key] = embeds
    return embeds

def _update_preview(latents, timings):
    if _preview_vae is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            small = torch.nn.functional.interpolate(latents[:1].to(_preview_vae.dtype), scale_factor=0.5, mode="bilinear")
            decoded = _preview_vae.decode(small).sample
        img = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=70)
        _progress["preview"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        _progress["preview_id"] += 1
        timings["preview"] = timings.get("preview", 0.0) + time.perf_counter() - t0
    except Exception as e:
        print(f"Preview decode failed: {e}")

def build_step_callback(pipe, total_steps, cfg_cutoff, live_preview, timings):
    """Per-step hook: progress, cancel, throttled TAESD previews and optional CFG truncation."""
    tensor_inputs = getattr(pipe, "_callback_tensor_inputs", ["latents"])
    wanted = ["latents"]
    truncate_cfg = cfg_cutoff < 1.0 and "prompt_embeds" in tensor_inputs
    if truncate_cfg:
        wanted += [k for k in ("prompt_embeds", "add_text_embeds", "add_time_ids") if k in tensor_inputs]
    state = {"last_preview": 0.0, "cfg_dropped": False}

    def callback(p, step_index, timestep, callback_kwargs):
        if _cancel_requested:
            raise GenerationCancelled()
        # img2img runs fewer steps than requested; the pipeline knows the real count
        steps_total = getattr(p, "_num_timesteps", None) or total_steps
        _progress["step"] = step_index + 1
        _progress["total"] = steps_total

        cutoff_step = max(1, int(round(steps_total * cfg_cutoff)))
        if truncate_cfg and not state["cfg_dropped"] and step_index + 1 >= cutoff_step and p.do_classifier_free_guidance:
            for k in ("prompt_embeds", "add_text_embeds", "add_time_ids"):
                if k in callback_kwargs:
                    callback_kwargs[k] = callback_kwargs[k].chunk(2)[-1]
            p._guidance_scale = 0.0
            state["cfg_dropped"] = True

        if step_index + 1 >= steps_total:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings["denoise_end"] = time.perf_counter()
        elif live_preview and time.perf_counter() - state["last_preview"] > 0.75:
            _update_preview(callback_kwargs["latents"], timings)
            state["last_preview"] = time.perf_counter()
        return callback_kwargs

    return callback, wanted

def get_pipeline(model_key: str):
    global current_pipe, current_model_id, current_performance_mode, current_torch_compile, current_turbo_mode, _compile_warmup_done, _preview_vae
    all_models = get_all_models()
    if model_key not in all_models:
        raise HTTPException(status_code=400, detail="Invalid model key")
    
    cfg = read_config()
    performance_mode = cfg["performance_mode"]
    torch_compile = bool(cfg["torch_compile"])
    turbo_mode = bool(cfg["turbo_mode"])

    model_info = all_models[model_key]
    model_id = model_info["id"]
    is_local = model_info.get("is_local", False)
    is_single_file = model_info.get("is_single_file", False)
    
    if (current_model_id != model_id or 
        current_performance_mode != performance_mode or 
        current_torch_compile != torch_compile or 
        current_turbo_mode != turbo_mode or
        current_pipe is None):
        print(f"Loading pipeline for: {model_id} (Performance Mode: {performance_mode}, Turbo: {turbo_mode})...")
        _reset_pipeline_caches()
        
        # Dynamically ensure all files are downloaded before loading and detect FP16 support
        if not is_local:
            has_fp16 = ensure_model_downloaded(model_id)
        else:
            has_fp16 = False
        
        # Clean up old pipeline memory
        if current_pipe is not None:
            try:
                # Remove CPU offload hooks which cause circular references and memory leaks
                if hasattr(current_pipe, "remove_all_hooks"):
                    current_pipe.remove_all_hooks()
                # Move remaining tensors to CPU while silencing float16 cpu offload warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_pipe.to("cpu")
            except Exception as e:
                print(f"Warning during pipeline cleanup: {e}")
            del current_pipe
        
        gc.collect()
        gc.collect() # Double collection to clear circular refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine optimal torch dtype (float16 for RTX 3070 Ti)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Optimize PyTorch compiler & math settings for Ampere GPUs (RTX 30-series)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        try:
            if "z_image_turbo" in model_key or "z_image_turbo" in model_id:
                from diffusers import ZImagePipeline
                local_z_dir = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo")
                z_model_id = local_z_dir if os.path.exists(local_z_dir) else "Tongyi-MAI/Z-Image-Turbo"
                
                # Check if we are loading a single-file checkpoint of Z-Image-Turbo (e.g. V2).
                # If so, we dynamically create a local Diffusers-compatible folder structure
                # by junction-linking components from the base Z-Image-Turbo folder,
                # and linking or sharding our single safetensors file as the transformer.
                if os.path.isfile(model_id) and model_id.endswith(".safetensors"):
                    z_name = os.path.splitext(os.path.basename(model_id))[0]
                    v2_dir = os.path.join(CHECKPOINTS_DIR, z_name)
                    
                    # Check if the folder is fully initialized
                    is_complete = False
                    if os.path.exists(os.path.join(v2_dir, "model_index.json")):
                        transformer_dir = os.path.join(v2_dir, "transformer")
                        if os.path.exists(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")) or \
                           os.path.exists(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")):
                            is_complete = True
                    
                    if not is_complete:
                        print(f"Creating dynamic diffusers directory structure for {z_name}...")
                        shutil.rmtree(v2_dir, ignore_errors=True)
                        os.makedirs(os.path.join(v2_dir, "transformer"), exist_ok=True)
                        
                        # Copy model_index.json and transformer config.json from base directory
                        shutil.copy(os.path.join(local_z_dir, "model_index.json"), os.path.join(v2_dir, "model_index.json"))
                        shutil.copy(os.path.join(local_z_dir, "transformer", "config.json"), os.path.join(v2_dir, "transformer", "config.json"))
                        
                        # Check if the safetensors file is large and needs sharding to prevent OOM
                        file_size_gb = os.path.getsize(model_id) / (1024**3)
                        if file_size_gb > 4.0:
                            print(f"Large checkpoint file ({file_size_gb:.2f} GB) detected. Sharding on the fly...")
                            shard_safetensors_manually(model_id, os.path.join(v2_dir, "transformer"), max_shard_size_gb=4.0)
                        else:
                            # Small enough to hardlink directly
                            os.link(model_id, os.path.join(v2_dir, "transformer", "diffusion_pytorch_model.safetensors"))
                        
                        # Create junctions for the other folders (vae, text_encoder, tokenizer, scheduler)
                        for folder in ["vae", "text_encoder", "tokenizer", "scheduler"]:
                            src_folder = os.path.join(local_z_dir, folder)
                            dst_folder = os.path.join(v2_dir, folder)
                            if os.path.exists(src_folder) and not os.path.exists(dst_folder):
                                subprocess.run(["cmd", "/c", "mklink", "/J", dst_folder, src_folder], check=True)
                    
                    z_model_id = v2_dir
 
                load_local_only = os.path.exists(z_model_id) or is_local
                
                if torch.cuda.is_available():
                    from diffusers import BitsAndBytesConfig
                    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
                    from transformers import BitsAndBytesConfig as TransformersBnBConfig
                    from transformers import AutoModel, AutoTokenizer
                    
                    # Z-Image-Turbo (like Flux) requires bfloat16 to prevent NaN overflows in DiT layers
                    z_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    
                    # Diffusers quantization config for the transformer
                    diff_quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=z_dtype,
                    )
                    
                    # Transformers quantization config for the Qwen3 text encoder (~4B params)
                    tf_quant_config = TransformersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=z_dtype,
                    )
                    
                    print(f"Loading quantized Z-Image Transformer (4-bit NF4 in {z_dtype})...")
                    transformer = ZImageTransformer2DModel.from_pretrained(
                        z_model_id,
                        subfolder="transformer",
                        quantization_config=diff_quant_config,
                        torch_dtype=z_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Loading quantized Qwen3 Text Encoder (4-bit NF4 in {z_dtype})...")
                    text_encoder = AutoModel.from_pretrained(
                        z_model_id,
                        subfolder="text_encoder",
                        quantization_config=tf_quant_config,
                        torch_dtype=z_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                    
                    print("Loading Z-Image Pipeline with quantized components...")
                    current_pipe = ZImagePipeline.from_pretrained(
                        z_model_id,
                        transformer=transformer,
                        text_encoder=text_encoder,
                        torch_dtype=z_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                else:
                    current_pipe = ZImagePipeline.from_pretrained(
                        z_model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
            elif "flux" in model_key.lower() or "flux" in model_id.lower():
                from diffusers import FluxPipeline, FluxTransformer2DModel
                from transformers import T5EncoderModel
                local_flux_dir = os.path.join(CHECKPOINTS_DIR, os.path.basename(model_id))
                flux_model_id = local_flux_dir if os.path.exists(local_flux_dir) else model_id
                load_local_only = os.path.exists(flux_model_id) or is_local

                flux_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                if is_single_file:
                    print(f"Loading single-file FLUX checkpoint from {model_id}...")
                    flux_config_repo = "Freepik/FLUX.1-Lite-8B-Alpha" if ("lite" in model_id.lower() or "lite" in model_key.lower()) else "black-forest-labs/FLUX.1-schnell"
                    try:
                        current_pipe = FluxPipeline.from_single_file(
                            model_id,
                            config=flux_config_repo,
                            torch_dtype=flux_dtype,
                            use_safetensors=True,
                            local_files_only=False,
                            low_cpu_mem_usage=True
                        )
                    except Exception as s_err:
                        print(f"Single-file FLUX loading require text_encoders fallback: {s_err}")
                        from transformers import CLIPTextModel, T5EncoderModel
                        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=flux_dtype)
                        text_encoder_2 = T5EncoderModel.from_pretrained(flux_config_repo, subfolder="text_encoder_2", torch_dtype=flux_dtype)
                        current_pipe = FluxPipeline.from_single_file(
                            model_id,
                            config=flux_config_repo,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            torch_dtype=flux_dtype,
                            use_safetensors=True,
                            local_files_only=False,
                            low_cpu_mem_usage=True
                        )
                elif torch.cuda.is_available():
                    from diffusers import BitsAndBytesConfig
                    from transformers import BitsAndBytesConfig as TransformersBnBConfig
                    
                    diff_quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=flux_dtype,
                    )
                    tf_quant_config = TransformersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=flux_dtype,
                    )
                    
                    print(f"Loading quantized FLUX Transformer (4-bit NF4 in {flux_dtype})...")
                    transformer = FluxTransformer2DModel.from_pretrained(
                        flux_model_id,
                        subfolder="transformer",
                        quantization_config=diff_quant_config,
                        torch_dtype=flux_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Loading quantized T5 Text Encoder (4-bit NF4 in {flux_dtype})...")
                    text_encoder_2 = T5EncoderModel.from_pretrained(
                        flux_model_id,
                        subfolder="text_encoder_2",
                        quantization_config=tf_quant_config,
                        torch_dtype=flux_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                    
                    print("Loading FLUX Pipeline with quantized components...")
                    current_pipe = FluxPipeline.from_pretrained(
                        flux_model_id,
                        transformer=transformer,
                        text_encoder_2=text_encoder_2,
                        torch_dtype=flux_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
                else:
                    current_pipe = FluxPipeline.from_pretrained(
                        flux_model_id,
                        torch_dtype=flux_dtype,
                        use_safetensors=True,
                        local_files_only=load_local_only,
                        low_cpu_mem_usage=True
                    )
            elif "qwen" in model_key.lower() or "qwen" in model_id.lower():
                from diffusers import (
                    QwenImageEditPlusPipeline,
                    QwenImageTransformer2DModel,
                    AutoencoderKLQwenImage,
                    FlowMatchEulerDiscreteScheduler
                )
                from transformers import (
                    Qwen2VLProcessor,
                    Qwen2Tokenizer,
                    Qwen2_5_VLForConditionalGeneration
                )
                from safetensors import safe_open

                print(f"Loading Qwen-Image-Edit pipeline for: {model_id}...")
                base_repo = "Qwen/Qwen-Image-Edit-2511"
                local_base_dir = os.path.join(CHECKPOINTS_DIR, "Qwen-Image-Edit-2511")
                qwen_base_id = local_base_dir if os.path.exists(local_base_dir) else base_repo

                # Determine checkpoint file
                checkpoint_file = None
                if os.path.isfile(model_id) and model_id.endswith(".safetensors"):
                    checkpoint_file = model_id
                else:
                    for f in os.listdir(CHECKPOINTS_DIR):
                        if "qwen" in f.lower() and f.endswith(".safetensors"):
                            checkpoint_file = os.path.join(CHECKPOINTS_DIR, f)
                            break
                    if not checkpoint_file:
                        from huggingface_hub import hf_hub_download
                        print("Fetching default Qwen-Rapid-AIO checkpoint (v23 SFW)...")
                        checkpoint_file = hf_hub_download(
                            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                            filename="v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
                        )

                qwen_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                print("Loading Qwen base scheduler, processor, tokenizer, VAE, and text encoder...")
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    qwen_base_id, subfolder="scheduler"
                )
                processor = Qwen2VLProcessor.from_pretrained(
                    qwen_base_id, subfolder="processor"
                )
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    qwen_base_id, subfolder="tokenizer"
                )
                vae = AutoencoderKLQwenImage.from_pretrained(
                    qwen_base_id, subfolder="vae", torch_dtype=qwen_dtype
                )
                text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    qwen_base_id,
                    subfolder="text_encoder",
                    torch_dtype=qwen_dtype,
                    low_cpu_mem_usage=True
                )

                transformer_config = QwenImageTransformer2DModel.load_config(
                    qwen_base_id, subfolder="transformer"
                )

                if checkpoint_file and os.path.exists(checkpoint_file):
                    print(f"Loading Rapid-AIO DiT transformer weights from {checkpoint_file}...")
                    with torch.device("meta"):
                        transformer = QwenImageTransformer2DModel.from_config(transformer_config)
                    
                    transformer = transformer.to_empty(device="cpu").to(dtype=qwen_dtype)
                    
                    prefix = "model.diffusion_model."
                    with safe_open(checkpoint_file, framework="pt", device="cpu") as sf:
                        sf_keys = sf.keys()
                        state_dict = {}
                        has_prefix = any(k.startswith(prefix) for k in sf_keys)
                        for k in sf_keys:
                            if has_prefix:
                                if k.startswith(prefix):
                                    mapped_k = k[len(prefix):]
                                    if mapped_k != "__index_timestep_zero__":
                                        state_dict[mapped_k] = sf.get_tensor(k).to(dtype=qwen_dtype)
                            else:
                                if k != "__index_timestep_zero__":
                                    state_dict[k] = sf.get_tensor(k).to(dtype=qwen_dtype)
                        
                        transformer.load_state_dict(state_dict, strict=True)
                        del state_dict
                else:
                    print(f"Loading transformer from {qwen_base_id}...")
                    transformer = QwenImageTransformer2DModel.from_pretrained(
                        qwen_base_id,
                        subfolder="transformer",
                        torch_dtype=qwen_dtype,
                        low_cpu_mem_usage=True
                    )

                current_pipe = QwenImageEditPlusPipeline(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    processor=processor,
                    transformer=transformer
                )
            elif is_single_file:
                from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
                print(f"Attempting to load single-file checkpoint as SDXL from {model_id}...")
                try:
                    current_pipe = StableDiffusionXLPipeline.from_single_file(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True
                    )
                    if getattr(current_pipe, "tokenizer_2", None) is None or getattr(current_pipe, "text_encoder_2", None) is None:
                        raise ValueError("Model loaded as SDXL but missing tokenizer_2 or text_encoder_2. Falling back to SD 1.5.")
                except Exception as xl_err:
                    print(f"Failed to load as SDXL: {xl_err}. Retrying as SD 1.5...")
                    current_pipe = StableDiffusionPipeline.from_single_file(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True
                    )
            else:
                # We use AutoPipelineForText2Image to load
                current_pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype,
                    variant="fp16" if has_fp16 else None,
                    use_safetensors=True,
                    local_files_only=is_local,
                    low_cpu_mem_usage=True
                )
            
            is_sdxl = "StableDiffusionXL" in type(current_pipe).__name__
            is_unet_pipe = getattr(current_pipe, "unet", None) is not None

            if torch.cuda.is_available() and is_sdxl:
                _swap_in_fp16_vae(current_pipe)

            if torch.cuda.is_available():
                if "qwen" in type(current_pipe).__name__.lower():
                    print("Performance Mode: Qwen DiT + 7B VL. Enabling model CPU offload & VAE tiling/slicing.")
                    current_pipe.enable_model_cpu_offload()
                    if hasattr(current_pipe, "vae") and current_pipe.vae is not None:
                        current_pipe.vae.enable_slicing()
                        current_pipe.vae.enable_tiling()
                elif performance_mode == "max_speed":
                    print("Performance Mode: MAX SPEED. Loading entire pipeline directly to GPU.")
                    current_pipe.to("cuda")

                    if hasattr(current_pipe, "vae") and current_pipe.vae is not None:
                        # Upcast VAE to float32 if force_upcast is True (standard for SDXL).
                        # Eliminates dynamic weight conversion overhead, prevents black/NaN outputs,
                        # and safely upcasts latents at decode time to prevent dtype mismatch.
                        if getattr(current_pipe.vae.config, "force_upcast", False):
                            current_pipe.vae.to(dtype=torch.float32)
                            _orig_decode = current_pipe.vae.decode
                            current_pipe.vae.decode = lambda z, *args, **kwargs: _orig_decode(z.to(current_pipe.vae.dtype), *args, **kwargs)
                        try:
                            current_pipe.vae.enable_slicing()
                        except Exception:
                            pass

                    for module_name in ("unet", "vae"):
                        module = getattr(current_pipe, module_name, None)
                        if module is not None:
                            try:
                                module.to(memory_format=torch.channels_last)
                            except Exception:
                                pass
                else:
                    print("Performance Mode: BALANCED. Enabling CPU offloading & slicing/tiling.")
                    # Enable model CPU offloading for all pipelines.
                    current_pipe.enable_model_cpu_offload()
                    
                    if hasattr(current_pipe, "unet") and current_pipe.unet is not None:
                        try:
                            current_pipe.unet.to(memory_format=torch.channels_last)
                        except Exception:
                            pass

                    # Enable VAE slicing and tiling (saves VRAM on VAE decoding)
                    if hasattr(current_pipe, "vae") and current_pipe.vae is not None:
                        if getattr(current_pipe.vae.config, "force_upcast", False):
                            current_pipe.vae.to(dtype=torch.float32)
                            _orig_decode = current_pipe.vae.decode
                            current_pipe.vae.decode = lambda z, *args, **kwargs: _orig_decode(z.to(current_pipe.vae.dtype), *args, **kwargs)
                        current_pipe.vae.enable_slicing()
                        current_pipe.vae.enable_tiling()
                    
                    # Enable attention slicing to limit VRAM spikes during processing
                    if hasattr(current_pipe, "enable_attention_slicing"):
                        current_pipe.enable_attention_slicing()

                if turbo_mode:
                    if is_sdxl:
                        _apply_turbo_lora(current_pipe)
                    else:
                        print("Turbo mode only applies to SDXL pipelines. Skipping for this model.")
                if is_unet_pipe:
                    _load_preview_vae(is_sdxl)
                _install_vae_timer(current_pipe)
            
            # Apply torch.compile() if enabled
            if torch.cuda.is_available() and torch_compile:
                try:
                    import triton
                    has_triton = True
                except Exception:
                    has_triton = False
                
                if has_triton:
                    print("Performance Mode: COMPILING UNet/Transformer with torch.compile...")
                    try:
                        import torch._inductor.config as inductor_cfg
                        # Skip GEMM autotuning & coordinate descent tuning to prevent 3+ min stalls
                        inductor_cfg.max_autotune_gemm = False
                        inductor_cfg.coordinate_descent_tuning = False
                        inductor_cfg.epilogue_fusion = True
                        inductor_cfg.triton.cudagraphs = True
                        # Disable split_cat_fx_passes to prevent CantSplit symbolic scheduler errors
                        inductor_cfg.split_cat_fx_passes = False
                    except Exception:
                        pass
                    if hasattr(current_pipe, "unet") and current_pipe.unet is not None:
                        print("Compiling UNet with torch.compile (reduce-overhead, dynamic=False)...")
                        current_pipe.unet = torch.compile(current_pipe.unet, mode="reduce-overhead", fullgraph=False, dynamic=False)
                    elif hasattr(current_pipe, "transformer") and current_pipe.transformer is not None:
                        print("Compiling Transformer with torch.compile (reduce-overhead, dynamic=False)...")
                        current_pipe.transformer = torch.compile(current_pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False)
                else:
                    print("Warning: torch_compile is enabled but Triton is not installed. Skipping compilation.")
            
            current_model_id = model_id
            current_performance_mode = performance_mode
            current_torch_compile = torch_compile
            current_turbo_mode = turbo_mode
            _compile_warmup_done = False
            print(f"Pipeline loaded successfully. (torch_compile={torch_compile}, turbo_mode={turbo_mode})")
        except Exception as e:
            import traceback
            print("Error loading pipeline:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")
            
    return current_pipe

@app.get("/api/models")
def get_models():
    return get_all_models()

@app.get("/api/samplers")
def get_samplers():
    return {k: v["name"] for k, v in SAMPLERS.items()}

@app.get("/api/loras")
def get_loras():
    loras = []
    if os.path.exists(LORAS_DIR):
        for root, _, files in os.walk(LORAS_DIR):
            for file in files:
                if file.lower().endswith((".safetensors", ".bin", ".pt")):
                    rel_path = os.path.relpath(os.path.join(root, file), LORAS_DIR)
                    rel_path = rel_path.replace("\\", "/")
                    loras.append({
                        "name": rel_path,
                        "filename": file
                    })
    return loras

@app.get("/api/config")
def get_config():
    return read_config()

@app.post("/api/config")
def update_config(req: ConfigRequest):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(req.model_dump(), f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")
    return {"status": "ok"}

@app.get("/api/progress")
def get_progress():
    return _progress

@app.post("/api/cancel")
def cancel_generation():
    global _cancel_requested
    if _progress["active"]:
        _cancel_requested = True
    return {"status": "ok"}

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    return run_on_gpu_thread(_generate_entry, req)

def _generate_entry(req: GenerateRequest):
    global _cancel_requested
    _cancel_requested = False
    _progress.update({"active": True, "step": 0, "total": req.steps, "preview": None})
    try:
        return _generate_image_locked(req)
    finally:
        _progress["active"] = False
        _cancel_requested = False

def _generate_image_locked(req: GenerateRequest):
    timings = {}
    t_request = time.perf_counter()
    pipe = get_pipeline(req.model_key)
    timings["load"] = time.perf_counter() - t_request
    
    # Handle Sampler (Scheduler)
    is_z_image = "ZImage" in type(pipe).__name__
    is_flux = "Flux" in type(pipe).__name__
    is_qwen = "QwenImage" in type(pipe).__name__
    if not is_z_image and not is_flux and not is_qwen and req.sampler_key in SAMPLERS and hasattr(pipe, "scheduler"):
        sampler_info = SAMPLERS[req.sampler_key]
        try:
            pipe.scheduler = sampler_info["class"].from_config(
                pipe.scheduler.config,
                **sampler_info["kwargs"]
            )
        except Exception as e:
            print(f"Warning: Could not set scheduler {req.sampler_key}: {e}")
            
    # Handle LoRAs
    t0 = time.perf_counter()
    lora_signature = ()
    active_loras = [l for l in req.loras if l.enabled and l.name]
    if hasattr(pipe, "load_lora_weights"):
        if is_z_image or is_flux or is_qwen:
            if active_loras:
                print("Warning: Quantized/Merged DiT models (Qwen/Z-Image/FLUX) do not support loading standard SDXL LoRAs. Skipping LoRA application.")
        elif active_loras or _loaded_adapters:
            lora_signature = apply_loras(pipe, active_loras)
    timings["lora"] = time.perf_counter() - t0
                
    # Handle init image for img2img or image edit
    init_img = None
    if req.init_image:
        try:
            raw_img_data = req.init_image.strip()
            if raw_img_data.startswith("data:image"):
                raw_img_data = raw_img_data.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(raw_img_data)
                init_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                possible_path = os.path.join(OUTPUT_DIR, req.init_image)
                if os.path.exists(possible_path):
                    init_img = Image.open(possible_path).convert("RGB")
                elif os.path.exists(req.init_image):
                    init_img = Image.open(req.init_image).convert("RGB")
            
            if init_img is not None:
                # Resize to target dimensions ensuring multiples of 16
                target_w = max(64, (req.width // 16) * 16)
                target_h = max(64, (req.height // 16) * 16)
                init_img = init_img.resize((target_w, target_h), Image.LANCZOS)
                print(f"Loaded init_image successfully. Target size: {target_w}x{target_h}, strength: {req.strength}")
        except Exception as img_err:
            print(f"Warning: Failed to load init_image: {img_err}")
            init_img = None

    # Handle seed
    if req.seed == -1:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = req.seed
        
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    # Dynamically check total GPU VRAM to determine high-res tiling threshold (2048 for >=12GB cards like RTX 5060 Ti 16GB, 1024 for <=8GB cards)
    total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0
    tiling_threshold = 2048 if total_vram_gb >= 12.0 else 1024
    
    is_high_res = req.width >= tiling_threshold or req.height >= tiling_threshold
    if hasattr(pipe, "vae") and pipe.vae is not None:
        try:
            pipe.vae.enable_slicing()
            if is_high_res:
                print(f"High resolution ({req.width}x{req.height} >= {tiling_threshold}px) detected. Enabling VAE tiling...")
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
        except Exception:
            pass

    # Clear console message on first compiled generation
    global _compile_warmup_done, _last_compiled_shape
    if current_performance_mode == "max_speed" and current_torch_compile and not _compile_warmup_done:
        print("Note: Torch compile is enabled. The initial generation run compiles C++/CUDA kernels (takes 1-3 min warmup). Subsequent runs will be significantly faster.")
        _compile_warmup_done = True
    
    # Warn about resolution-change recompilation under torch.compile
    current_shape = (req.width, req.height)
    if current_torch_compile and _last_compiled_shape is not None and current_shape != _last_compiled_shape:
        print(f"⚠ Resolution changed from {_last_compiled_shape[0]}x{_last_compiled_shape[1]} to {req.width}x{req.height}. "
              f"torch.compile(reduce-overhead) must re-capture CUDA graphs for the new shape. This will take 1-3 minutes (one-time per resolution).")
    _last_compiled_shape = current_shape
        
    action_type = "Image Edit / Img2Img" if init_img is not None else "Text-to-Image"
    print(f"Generating image ({action_type}). Prompt: '{req.prompt}', Steps: {req.steps}, Seed: {seed}")
    
    is_unet_pipe = getattr(pipe, "unet", None) is not None
    is_sdxl = "StableDiffusionXL" in type(pipe).__name__
    step_callback, callback_inputs = build_step_callback(
        pipe,
        req.steps,
        req.cfg_cutoff if is_unet_pipe else 1.0,
        req.live_preview and is_unet_pipe and _preview_vae is not None,
        timings
    )
    cb_kwargs = {"callback_on_step_end": step_callback, "callback_on_step_end_tensor_inputs": callback_inputs}
    _vae_timing["decode"] = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    try:
        # SD/SDXL prompt conditioning (SDXL embeddings are cached across requests)
        prompt_kwargs = {"prompt": req.prompt, "negative_prompt": req.negative_prompt}
        if is_sdxl:
            do_cfg = req.cfg_scale > 1.0 and pipe.unet.config.time_cond_proj_dim is None
            with torch.inference_mode():
                prompt_kwargs = encode_sdxl_prompt(pipe, req.prompt, req.negative_prompt, do_cfg, lora_signature)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["encode"] = time.perf_counter() - start_time
        t_pipe_start = time.perf_counter()

        def _run_inference():
            if init_img is not None:
                if is_qwen:
                    if req.edit_mode == "img2img":
                        from diffusers import QwenImageImg2ImgPipeline
                        i2i_pipe = QwenImageImg2ImgPipeline.from_pipe(pipe)
                        return i2i_pipe(
                            prompt=req.prompt,
                            negative_prompt=req.negative_prompt if req.negative_prompt else None,
                            image=init_img,
                            strength=max(0.05, min(1.0, req.strength)),
                            num_inference_steps=req.steps,
                            true_cfg_scale=req.cfg_scale,
                            generator=generator,
                            **cb_kwargs
                        )
                    else:
                        # Default Instruct Edit mode
                        return pipe(
                            image=init_img,
                            prompt=req.prompt,
                            negative_prompt=req.negative_prompt if req.negative_prompt else None,
                            num_inference_steps=req.steps,
                            true_cfg_scale=req.cfg_scale,
                            generator=generator,
                            **cb_kwargs
                        )
                elif is_flux:
                    from diffusers import FluxImg2ImgPipeline
                    i2i_pipe = FluxImg2ImgPipeline.from_pipe(pipe)
                    return i2i_pipe(
                        prompt=req.prompt,
                        image=init_img,
                        strength=max(0.05, min(1.0, req.strength)),
                        num_inference_steps=req.steps,
                        guidance_scale=req.cfg_scale,
                        generator=generator,
                        max_sequence_length=512,
                        **cb_kwargs
                    )
                else:
                    from diffusers import AutoPipelineForImage2Image
                    i2i_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
                    if not is_sdxl:
                        prompt_kwargs["negative_prompt"] = req.negative_prompt or None
                    return i2i_pipe(
                        **prompt_kwargs,
                        image=init_img,
                        strength=max(0.05, min(1.0, req.strength)),
                        num_inference_steps=req.steps,
                        guidance_scale=req.cfg_scale,
                        generator=generator,
                        **cb_kwargs
                    )
            else:
                if is_qwen:
                    return pipe(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt if req.negative_prompt else None,
                        num_inference_steps=req.steps,
                        true_cfg_scale=req.cfg_scale,
                        width=req.width,
                        height=req.height,
                        generator=generator,
                        **cb_kwargs
                    )
                elif is_flux:
                    return pipe(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt if req.negative_prompt else None,
                        num_inference_steps=req.steps,
                        guidance_scale=req.cfg_scale,
                        width=req.width,
                        height=req.height,
                        generator=generator,
                        max_sequence_length=512,
                        **cb_kwargs
                    )
                else:
                    return pipe(
                        **prompt_kwargs,
                        num_inference_steps=req.steps,
                        guidance_scale=req.cfg_scale,
                        width=req.width,
                        height=req.height,
                        generator=generator,
                        **cb_kwargs
                    )

        try:
            with torch.inference_mode():
                result = _run_inference()
        except GenerationCancelled:
            raise
        except Exception as compile_err:
            err_str = repr(compile_err)
            if isinstance(compile_err, AssertionError) or any(k in err_str for k in ["CantSplit", "InductorError", "dynamo", "CUDA error", "cudaGraphs", "cuda_graph"]):
                print(f"⚠ torch.compile / Inductor issue detected: {compile_err}")
                print("🔄 Automatically executing with uncompiled eager mode fallback...")
                try:
                    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
                        torch._dynamo.reset()
                except Exception:
                    pass
                with torch._dynamo.disable():
                    with torch.inference_mode():
                        result = _run_inference()
                print("✓ Eager mode fallback generation succeeded!")
            else:
                raise compile_err
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pipe_end = time.perf_counter()
        generation_time = t_pipe_end - start_time
        steps_done = _progress["total"] or req.steps
        denoise_time = timings.get("denoise_end", t_pipe_end) - t_pipe_start
        timings["denoise"] = denoise_time
        timings["decode"] = _vae_timing["decode"]
        timings["other"] = max(0.0, (t_pipe_end - t_pipe_start) - denoise_time - timings["decode"])
        it_per_sec = steps_done / denoise_time if denoise_time > 0 else 0
        print(f"Generation complete in {generation_time:.2f}s (denoise {denoise_time:.2f}s @ {it_per_sec:.2f} it/s, VAE {timings['decode']*1000:.0f}ms)")
        
        image = result.images[0]
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        all_models = get_all_models()
        model_name = all_models[req.model_key]['name'] if req.model_key in all_models else req.model_key
        model_id_str = all_models[req.model_key]['id'] if req.model_key in all_models else req.model_key
        
        # Prepare metadata string (A1111 / Comfy / Standard PIL format)
        img2img_meta = f", Img2Img: True, Strength: {req.strength}, Mode: {req.edit_mode}" if init_img is not None else ""
        png_meta_text = (
            f"{req.prompt}\n"
            f"Negative prompt: {req.negative_prompt}\n"
            f"Steps: {req.steps}, Sampler: {req.sampler_key}, CFG scale: {req.cfg_scale}, Seed: {seed}, "
            f"Size: {req.width}x{req.height}, Model: {model_name}, Checkpoint: {model_id_str}{img2img_meta}, "
            f"Time: {generation_time:.2f}s ({it_per_sec:.2f} it/s)"
        )
        
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        png_info.add_text("parameters", png_meta_text)
        png_info.add_text("prompt", req.prompt)
        png_info.add_text("negative_prompt", req.negative_prompt)
        png_info.add_text("steps", str(req.steps))
        png_info.add_text("cfg_scale", str(req.cfg_scale))
        png_info.add_text("seed", str(seed))
        png_info.add_text("model", model_name)
        png_info.add_text("checkpoint", model_id_str)
        png_info.add_text("sampler", req.sampler_key)
        png_info.add_text("dimensions", f"{req.width}x{req.height}")
        if init_img is not None:
            png_info.add_text("img2img_strength", str(req.strength))
            png_info.add_text("img2img_mode", req.edit_mode)
        
        t0 = time.perf_counter()
        image.save(filepath, pnginfo=png_info, compress_level=1)
        timings["save"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_request
        breakdown = _format_breakdown(timings)
        print(f"Breakdown: {breakdown}")
        
        # Save complete metadata to a text file next to the image
        meta_filepath = filepath.replace(".png", ".txt")
        with open(meta_filepath, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {req.prompt}\n")
            f.write(f"Negative Prompt: {req.negative_prompt}\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Checkpoint File/ID: {model_id_str}\n")
            f.write(f"Sampler: {req.sampler_key}\n")
            f.write(f"Steps: {req.steps}\n")
            f.write(f"CFG Scale: {req.cfg_scale}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Dimensions: {req.width}x{req.height}\n")
            f.write(f"Performance Mode: {current_performance_mode}\n")
            f.write(f"Torch Compiled: {current_torch_compile}\n")
            if init_img is not None:
                f.write(f"Img2Img Active: True\n")
                f.write(f"Denoising Strength: {req.strength}\n")
                f.write(f"Edit Mode: {req.edit_mode}\n")
            if req.loras:
                f.write(f"Active LoRAs: {json.dumps([l.model_dump() for l in req.loras])}\n")
            if req.cfg_cutoff < 1.0:
                f.write(f"CFG Cutoff: {req.cfg_cutoff}\n")
            f.write(f"Turbo Mode: {current_turbo_mode}\n")
            f.write(f"Time Taken: {generation_time:.2f}s\n")
            f.write(f"Speed: {it_per_sec:.2f} it/s\n")
            f.write(f"Breakdown: {breakdown}\n")

        return {
            "image_url": f"/static/outputs/{filename}",
            "seed": seed,
            "filename": filename,
            "time_taken": f"{generation_time:.2f}s",
            "speed": f"{it_per_sec:.2f} it/s",
            "breakdown": breakdown,
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items() if k != "denoise_end"}
        }
    except GenerationCancelled:
        print("Generation cancelled by user.")
        raise HTTPException(status_code=409, detail="Generation cancelled")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Generation error: {e!r}")
        raise HTTPException(status_code=500, detail=str(e) or repr(e))
    finally:
        if 'result' in locals():
            del result
        if 'image' in locals():
            del image
        if 'init_img' in locals():
            del init_img
        # Emptying the CUDA cache every request forces fresh allocations next run; only do it when offloading
        if current_performance_mode != "max_speed":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def _format_breakdown(timings):
    def ms(key):
        v = timings.get(key, 0.0)
        return f"{v:.2f}s" if v >= 1.0 else f"{v * 1000:.0f}ms"
    parts = [f"denoise {ms('denoise')}", f"vae {ms('decode')}", f"encode {ms('encode')}", f"save {ms('save')}"]
    if timings.get("preview", 0) > 0:
        parts.append(f"preview {ms('preview')}")
    if timings.get("lora", 0) > 0.005:
        parts.append(f"lora {ms('lora')}")
    if timings.get("load", 0) > 0.05:
        parts.append(f"load {ms('load')}")
    return " | ".join(parts)


@app.post("/api/animate-video")
def animate_image_to_video(req: VideoRequest):
    return run_on_gpu_thread(_animate_image_to_video, req)

def _animate_image_to_video(req: VideoRequest):
    input_image_path = os.path.join(OUTPUT_DIR, req.image_filename)
    if not os.path.exists(input_image_path):
        raise HTTPException(status_code=404, detail="Input image file not found.")

    try:
        raw_img = Image.open(input_image_path).convert("RGB")
        # Resize image to SVD compatible resolution (e.g. 1024x576 or scaled to 1024 width/height max)
        w, h = raw_img.size
        target_w = (w // 64) * 64
        target_h = (h // 64) * 64
        if target_w > 1024 or target_h > 1024:
            target_w, target_h = 1024, 576
        resized_img = raw_img.resize((target_w, target_h), Image.LANCZOS)

        pipeline = get_svd_pipeline()
        
        if req.seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = req.seed

        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        print(f"Executing Image-to-Video Animation for {req.image_filename} ({req.num_frames} frames @ {req.fps} fps)...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.inference_mode():
            frames = pipeline(
                resized_img,
                decode_chunk_size=8,
                generator=generator,
                num_frames=req.num_frames,
                motion_bucket_id=req.motion_bucket_id,
                noise_aug_strength=req.noise_aug_strength
            ).frames[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_taken = time.perf_counter() - start_time
        print(f"Video animation rendered successfully in {time_taken:.2f}s!")

        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)

        imageio.mimsave(video_path, frames, fps=req.fps)

        return {
            "video_url": f"/static/outputs/{video_filename}",
            "filename": video_filename,
            "frames": len(frames),
            "fps": req.fps,
            "seed": seed,
            "time_taken": f"{time_taken:.2f}s"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video rendering error: {str(e)}")
    finally:
        if 'frames' in locals():
            del frames
        release_svd_pipeline()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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


class SamplerTestRequest(BaseModel):
    model_key: str
    prompt: str = "epic cinematic cybernetic dragon, hyperdetailed masterpiece, 8k resolution, vibrant lighting"
    negative_prompt: str = "blurry, low quality, distorted, ugly, pixelated"
    steps: int = 25
    cfg_scale: float = 6.0
    width: int = 1024
    height: int = 1024
    seed: int = 4224917681

@app.post("/api/test-samplers")
def run_sampler_test(req: SamplerTestRequest):
    return run_on_gpu_thread(_run_sampler_test, req)

def _run_sampler_test(req: SamplerTestRequest):
    """Run batch sampler test for a model across all registered samplers, saving output to test_output/<CheckpointName>/"""
    all_models = get_all_models()
    if req.model_key not in all_models:
        raise HTTPException(status_code=400, detail="Model key not found")
        
    model_info = all_models[req.model_key]
    model_name_clean = model_info['name'].replace(" ", "_").replace("/", "_").replace(":", "")
    checkpoint_test_dir = os.path.join("static", "test_output", model_name_clean)
    os.makedirs(checkpoint_test_dir, exist_ok=True)
    
    # Ensure model is loaded
    pipe = get_pipeline(req.model_key)
    
    test_results = []
    test_seed = req.seed if req.seed != -1 else random.randint(0, 2**32 - 1)
    
    for sampler_key, sampler_info in SAMPLERS.items():
        try:
            print(f"[TEST MODE] Testing sampler '{sampler_info['name']}' ({sampler_key})...")
            
            # Switch scheduler
            scheduler_cls = sampler_info["class"]
            kwargs = sampler_info["kwargs"].copy()
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
            
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(test_seed)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
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
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dur = time.perf_counter() - t0
            speed = req.steps / dur if dur > 0 else 0
            
            filename = f"{sampler_key}.png"
            filepath = os.path.join(checkpoint_test_dir, filename)
            
            # Save PNG & Metadata
            from PIL.PngImagePlugin import PngInfo
            png_info = PngInfo()
            png_info.add_text("prompt", req.prompt)
            png_info.add_text("sampler", sampler_info["name"])
            png_info.add_text("steps", str(req.steps))
            png_info.add_text("seed", str(test_seed))
            png_info.add_text("time_taken", f"{dur:.2f}s")
            
            result.images[0].save(filepath, pnginfo=png_info)
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            meta_path = filepath.replace(".png", ".txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"Model: {model_info['name']}\n")
                f.write(f"Sampler: {sampler_info['name']} ({sampler_key})\n")
                f.write(f"Prompt: {req.prompt}\n")
                f.write(f"Steps: {req.steps}\n")
                f.write(f"CFG Scale: {req.cfg_scale}\n")
                f.write(f"Seed: {test_seed}\n")
                f.write(f"Dimensions: {req.width}x{req.height}\n")
                f.write(f"Time Taken: {dur:.2f}s\n")
                f.write(f"Speed: {speed:.2f} it/s\n")
                
            test_results.append({
                "sampler_key": sampler_key,
                "sampler_name": sampler_info["name"],
                "image_url": f"/static/test_output/{model_name_clean}/{filename}",
                "time_taken": f"{dur:.2f}s",
                "speed": f"{speed:.2f} it/s",
                "seed": test_seed
            })
        except Exception as err:
            print(f"Error testing sampler {sampler_key}: {err}")
            test_results.append({
                "sampler_key": sampler_key,
                "sampler_name": sampler_info["name"],
                "error": str(err)
            })
            
    return {
        "model_name": model_info["name"],
        "folder_name": model_name_clean,
        "results": test_results
    }

@app.get("/api/test-results")
def get_all_test_results():
    """Retrieve all benchmark test matrix output folders across all checkpoints"""
    base_dir = os.path.join("static", "test_output")
    test_runs = {}
    if not os.path.exists(base_dir):
        return test_runs
        
    for cp_folder in os.listdir(base_dir):
        cp_path = os.path.join(base_dir, cp_folder)
        if os.path.isdir(cp_path):
            images = []
            for file in os.listdir(cp_path):
                if file.endswith(".png"):
                    sampler_key = file.replace(".png", "")
                    meta_path = os.path.join(cp_path, file.replace(".png", ".txt"))
                    meta = {}
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    if ":" in line:
                                        k, v = line.split(":", 1)
                                        meta[k.strip().lower().replace(" ", "_")] = v.strip()
                        except Exception:
                            pass
                    images.append({
                        "sampler_key": sampler_key,
                        "sampler_name": meta.get("sampler", sampler_key),
                        "image_url": f"/static/test_output/{cp_folder}/{file}",
                        "meta": meta
                    })
            test_runs[cp_folder] = images
    return test_runs

def _warmup_compiled_pipeline(model_key, pipe):
    """Run a short 1024x1024 generation so torch.compile / CUDA graph capture happens before the first real request."""
    global _compile_warmup_done, _last_compiled_shape
    if not (current_torch_compile and current_performance_mode == "max_speed" and getattr(pipe, "unet", None) is not None):
        return
    is_sdxl = "StableDiffusionXL" in type(pipe).__name__
    model_info = get_all_models().get(model_key, {})
    cfg_scale = 1.0 if (current_turbo_mode and is_sdxl) else model_info.get("default_cfg", 6.0)
    print(f" Warming up compiled UNet at 1024x1024 (CFG {cfg_scale})... first boot after a code/torch change takes 1-3 min.")
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            # Full decode too, so cuDNN autotunes the VAE convs now instead of on the first real image
            pipe(prompt="warmup", negative_prompt="", num_inference_steps=3, guidance_scale=cfg_scale,
                 width=1024, height=1024)
            if _preview_vae is not None:
                _update_preview(torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.float16), {})
        _compile_warmup_done = True
        _last_compiled_shape = (1024, 1024)
        print(f" Warmup finished in {time.perf_counter() - t0:.1f}s.")
    except Exception as e:
        print(f" Warmup skipped: {e}")

# Pre-load default model on startup to avoid request timeouts during load
def preload_model():
    preload_key = read_config()["preload_model"]

    if not preload_key:
        print("===================================================")
        print(" Pre-loading disabled by config.")
        print("===================================================")
        webbrowser.open("http://127.0.0.1:7860/")
        return

    print("===================================================")
    print(f" Pre-loading model '{preload_key}' to GPU...")
    print(" This ensures instant generation and prevents timeouts.")
    print("===================================================")
    try:
        pipe = get_pipeline(preload_key)
        print(" Model loaded successfully and ready!")
        _warmup_compiled_pipeline(preload_key, pipe)
        
        # Auto-open browser when model is fully loaded and ready
        webbrowser.open("http://127.0.0.1:7860/")
    except Exception as e:
        print(f" Warning: Could not pre-load model '{preload_key}' on startup: {e}")
        webbrowser.open("http://127.0.0.1:7860/")
    print("===================================================")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    response = FileResponse(os.path.join("static", "index.html"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=False)

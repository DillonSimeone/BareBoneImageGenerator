# BareBoneImageGenerator

> **All of the speed, none of the bloat.**

A lightweight, high-performance, and fully local text-to-image generation studio. Built with **FastAPI** on the backend and a responsive **Cyberpunk-Neon** dashboard on the frontend. Optimized specifically for consumer GPUs (like the RTX 3070 Ti 8GB VRAM) to prevent memory allocation crashes and slow disk-swapping lag.

---

## ⚡ Key Performance Optimizations

1. **Zero-Lag VRAM Control**: Automatically uses `enable_model_cpu_offload()` and `pipe.vae.enable_slicing()` to keep peak VRAM under **5 GB**. This prevents the slow Windows Shared Memory Paging lag and gets you maximum iteration throughput on 8GB cards.
2. **Ampere TF32 Acceleration**: Automatically enables TensorFloat-32 math for RTX 30-series/40-series cards, boosting matrix multiplication speeds on the Tensor cores by 15-20% with zero loss in image quality.
3. **No-Attention Slicing**: Processes attention layers in parallel rather than sequentially, unlocking your GPU's full speed.
4. **Dynamic Sequential Downloader**: Background model pre-loading downloads only the optimized `fp16` model variants sequentially, keeping system memory usage extremely low during setup and preventing Out of Memory (OOM) crashes.

---

## 🎨 Features

- **Community Fine-Tunes Support**:
  - `DreamShaper XL Turbo`: Hyper-fast 6-step general generation.
  - `Juggernaut XL V9`: The gold standard for photorealism, cinematic lighting, and human models.
  - `Pony Diffusion V6 XL`: Legendary uncensored anime, illustrative, and character-focused styles.
  - `Animagine XL V3.1`: High-quality anime illustration engine.
- **Auto-Lifespan Preloading**: Pre-loads the default model during server boot. This prevents browser request timeouts during generation.
- **Sleek Cyberpunk HUD**: Clean, neon-styled parameters, generation timer, and speed indicator (`it/s`).
- **Interactive Lightbox View**: Click any generated image to view it in an absolute-centered overlay with a blurred backdrop.
- **Backtick Key Trigger**: Press the backtick (`` ` ``) key anywhere on the page (even while typing in the prompt area) to execute the image generation instantly.
- **Parameter Logs**: Saves a `.txt` file next to every image containing the seed, prompt, CFG, time taken, and generation speed.

---

## 🚀 How to Run

1. **Double-click `run.bat`** in the root folder.
2. The console will boot up and pre-load the default model (DreamShaper Turbo).
3. Once the model is ready, your default browser will automatically open to `http://127.0.0.1:7860/`.
4. Enter your prompt and press the **backtick (`` ` ``)** key or click **GENERATE // EXECUTE**!

---

## 🛠️ Setup (First-time installation)

Ensure you have Python 3.10+ installed.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Install PyTorch with CUDA 12.1:
   ```bash
   .\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install dependencies:
   ```bash
   .\venv\Scripts\pip install -r requirements.txt
   ```

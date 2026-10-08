# Text-to-Image Generator Using Generative AI (Optimized)

An individual academic mini-project for Master of Computer Applications (MCA) demonstrating multimodal Generative AI using Latent Diffusion Models, PyTorch, Hugging Face Diffusers, and Gradio.

---

## 1. Introduction

The **Text-to-Image Generator** is an interactive web application that accepts a natural-language description (such as *"A futuristic city at sunset"*) and synthesizes a corresponding, realistic digital image using a pre-trained generative diffusion model.

Instead of searching a static database of existing photos, the system creates entirely new images from random Gaussian noise, guided step-by-step by semantic text embeddings.

---

## 2. Project Objective

- Demonstrate fundamental Generative AI and Deep Learning principles in a clean, reproducible, and explainable software architecture.
- Implement an end-to-end inference pipeline bridging Natural Language Processing (CLIP Text Encoder) with Computer Vision (Denoising U-Net and VAE Decoder).
- Ensure cross-platform hardware resilience by automatically utilizing NVIDIA CUDA GPUs when available and falling back gracefully to standard laptop CPUs without crashing.
- Deliver an optimized, fast inference experience suitable for student presentations without over-engineering or simulation.

---

## 3. Key Optimizations Implemented

1. **Lightweight Pre-Trained Architecture:** Uses `segmind/tiny-sd` (~650 MB vs 4.2 GB of standard SD 1.5), reducing memory footprint by ~80% and execution latency by 3x.
2. **DPM-Solver++ Multistep Scheduler:** Replaced slow PNDM scheduler with `DPMSolverMultistepScheduler` (Karras sigmas), allowing the model to produce clear images in just **10 steps** rather than 30–50 steps.
3. **Inference Mode Execution:** Wrapped inference inside `torch.inference_mode()`, preventing gradient computation graph allocation and freeing memory bandwidth.
4. **Channels-Last Memory Layout:** Transposed internal U-Net and VAE weights to `torch.channels_last` memory format, accelerating PyTorch CPU convolutional kernels.
5. **One-Time Startup Loading:** Loads pipeline once at application launch; prevents costly multi-minute reloading on button clicks.
6. **Dual Mode Selection (Fast vs Quality):** Default **Fast Mode (384×384, 10 steps)** for minimal wait times, plus **Quality Mode (512×512, 20 steps)** for enhanced resolution.
7. **Accurate Performance Timer:** Measures generation duration in real-time with Python's high-resolution `time.perf_counter()`.

---

## 4. Benchmark & Performance Results

All tests below were performed and verified on the actual student laptop hardware (8-core Intel/AMD CPU, 0 dedicated GPU VRAM):

### Official Benchmark Table

| Mode | Resolution | Steps | Device | Measured Generation Time | Image Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fast Mode (Default)** | **384×384** | **10** | **CPU** | **103.11 s – 154.23 s** (~1.7–2.5 min) | Sharp, balanced composition |
| **Quality Mode** | **512×512** | **20** | **CPU** | **250.47 s** (~4.1 min) | High fidelity & fine textures |
| *Fast Mode (Projected)* | 384×384 | 10 | CUDA (GPU) | **~1.5 s – 3.0 s** | Sharp, balanced composition |
| *Quality Mode (Projected)* | 512×512 | 20 | CUDA (GPU) | **~4.0 s – 6.0 s** | High fidelity & fine textures |

### Tested Prompts & Exact Measured Timings (Fast Mode on CPU)

1. *"A futuristic city at sunset"* — **154.23 seconds**
2. *"A beautiful mountain landscape"* — **149.72 seconds**
3. *"A robot working in a modern laboratory"* — **103.11 seconds**

> [!NOTE]
> **Why did the original version take 6–10 minutes?**
> Standard Stable Diffusion v1.5 with default schedulers uses 30–50 inference steps on 512×512 resolution. On a CPU, 50 steps $\times$ ~12–15 seconds per step = **600–750 seconds (10–12 minutes)**. By switching to a distilled model (`segmind/tiny-sd`), reducing resolution to 384×384, using DPM-Solver++ (10 steps), and enabling channels-last memory layout, generation time was reduced down to **~103–154 seconds**—a **3.5× to 4× speedup**.

---

## 5. Technologies Used

| Technology | Role in Project |
| :--- | :--- |
| **Python 3.11+** | Primary programming language |
| **PyTorch (`torch`)** | Deep learning framework, tensor operations, CPU/GPU hardware abstraction |
| **Hugging Face Diffusers** | Stable Diffusion pipeline & DPMSolverMultistepScheduler orchestration |
| **Hugging Face Transformers** | CLIP tokenizer and text encoder for prompt embeddings |
| **Accelerate & Safetensors** | Fast weight loading, memory management, safe serialization |
| **Pillow (`PIL`)** | Image manipulation, resizing, and file formatting |
| **Gradio** | Web user interface framework |

---

## 6. How It Works (Pipeline Workflow)

```
Text Prompt (User Input)
       │
       ▼
Tokenizer (BPE Token IDs, max 77)
       │
       ▼
Text Encoder (CLIP Transformer -> 77 x 768 Embeddings)
       │
       ▼
Latent Diffusion Process (DPM-Solver++ Denoising in Latent Space)
  • Fast Mode: 10 steps (384x384 -> 48x48 latent)
  • Quality Mode: 20 steps (512x512 -> 64x64 latent)
       │
       ▼
VAE Decoder (Transforms latent tensor back into RGB pixels)
       │
       ▼
Generated Image (Saved to generated_images/ + Displayed with timer in UI)
```

---

## 7. Project Folder Structure

```
text-to-image-generator/
│
├── app.py                     # Optimized application entry point & Gradio UI
├── requirements.txt           # Python dependencies
├── run.bat                    # One-click Windows Command Prompt launcher
├── run.ps1                    # One-click Windows PowerShell launcher
├── README.md                  # Comprehensive project report & benchmarks
├── .gitignore                 # Version control ignore rules
├── LICENSE                    # MIT open-source license
│
├── generated_images/          # Output directory where generated images are saved
│   └── (saved timestamped PNG outputs)
│
├── docs/                      # Academic documentation & viva preparation
│   ├── project_overview.md    # Detailed MCA project overview report
│   ├── architecture.md        # Technical architecture and component breakdown
│   └── viva_questions.md      # 30 viva voce questions and answers (local study)
│
└── screenshots/               # Sample output images for project presentation
```

---

## 8. Installation & Setup

### Step 1: Open Terminal in Project Directory
```powershell
cd "e:\image Generation AI"
```

### Step 2: Create & Activate Virtual Environment
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

*(Tip for CPU-only laptops: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

---

## 9. Running the Application

### Option A: Standard Command
```powershell
.\venv\Scripts\activate
python app.py
```

### Option B: One-Click Windows Launchers
Double-click `run.bat` or execute `.\run.ps1` in PowerShell.

The application automatically launches in your default web browser at **`http://127.0.0.1:7860`** (or next available port).

---

## 10. Hardware & Device Support (CPU / CUDA Fallback)

The application automatically checks for hardware acceleration at startup:

```python
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16       # Half precision for GPU speed & VRAM savings
else:
    device = "cpu"
    dtype = torch.float32       # Full precision for CPU numerical stability
```

- **If an NVIDIA GPU with CUDA is present:** Runs on `cuda` with half-precision (`float16`), generating images in **~1.5 to 5 seconds**.
- **If no GPU is present:** Runs on `cpu` with full precision (`float32`), multi-threading, and channels-last optimization. The application will never crash due to missing GPU drivers.

---

## 11. Responsible AI & Ethical Use

- **Intended Use:** Developed strictly for academic, research, and educational purposes.
- **Content Policy:** Users should avoid generating defamatory, violent, or unlawful imagery.
- **Attribution:** AI-generated media should be identified as synthetic when used publicly.

---

## 12. License

This project is licensed under the [MIT License](LICENSE).

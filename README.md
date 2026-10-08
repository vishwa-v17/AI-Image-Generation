# Text-to-Image Generator Using Generative AI

An individual academic mini-project for Master of Computer Applications (MCA) demonstrating multimodal Generative AI using Latent Diffusion Models, PyTorch, Hugging Face Diffusers, and Gradio.

---

## 1. Introduction

The **Text-to-Image Generator** is an interactive web application that accepts a natural-language description (such as *"A cute puppy sitting in a beautiful garden"*) and synthesizes a corresponding, realistic digital image using a pre-trained generative diffusion model.

Instead of searching a static database of existing photos, the system creates entirely new images from random Gaussian noise, guided step-by-step by semantic text embeddings.

---

## 2. Project Objective

- Demonstrate fundamental Generative AI and Deep Learning principles in a clean, reproducible, and explainable software architecture.
- Implement an end-to-end inference pipeline bridging Natural Language Processing (CLIP Text Encoder) with Computer Vision (Denoising U-Net and VAE Decoder).
- Ensure cross-platform hardware resilience by automatically utilizing NVIDIA CUDA GPUs when available and falling back gracefully to standard laptop CPUs without crashing.
- Provide a simple, user-friendly web interface with one-click image generation, status reporting, and download functionality.

---

## 3. Key Features

- **Natural-Language Prompt Understanding:** Converts plain English prompts into visual scenes.
- **Hardware-Aware Execution:**
  - Automatically detects **NVIDIA CUDA GPU** (executing in FP16 precision with attention slicing).
  - Automatically falls back to **CPU execution** (in FP32 precision) on standard student laptops.
- **One-Time Startup Loading:** Loads model weights into memory once at application launch to avoid reloading delays on subsequent requests.
- **Input Sanitization & Validation:** Prevents empty prompts and caps prompt lengths to under 350 characters to avoid memory anomalies.
- **Automated Local Storage & Downloads:** Automatically saves every generated image to `generated_images/` with a timestamped filename and provides a direct browser download link.
- **Interactive UI:** Built using Gradio Blocks featuring curated sample prompts, progress updates, and optional parameter sliders (inference steps, guidance scale, resolution).

---

## 4. Technologies Used

| Technology | Role in Project |
| :--- | :--- |
| **Python 3.11+** | Primary programming language |
| **PyTorch (`torch`)** | Deep learning framework, tensor mathematical operations, hardware device management |
| **Hugging Face Diffusers** | Diffusion pipelines, scheduling algorithms, and latent denoising loops |
| **Hugging Face Transformers** | CLIP tokenizer and text encoder for extracting prompt semantic vectors |
| **Accelerate & Safetensors** | Fast model loading, safe tensor serialization, and memory optimization |
| **Pillow (`PIL`)** | Image processing, formatting, and file export |
| **Gradio** | Web user interface framework running directly from Python |

---

## 5. How It Works (Pipeline Workflow)

```
Text Prompt (User Input)
       │
       ▼
Tokenizer (BPE Token IDs)
       │
       ▼
Text Encoder (CLIP Transformer)
       │
       ▼
Text Embedding (Semantic Context Vectors)
       │
       ▼
Latent Diffusion Process (Iterative U-Net Denoising in Latent Space)
       │
       ▼
VAE Decoder (Latent Space -> Pixel Space)
       │
       ▼
Generated Image (Displayed in Gradio UI + Saved to Disk)
```

1. **Prompt Tokenization & Embedding:** The raw prompt is tokenized into numeric tokens and converted into 768-dimensional semantic embedding vectors using the CLIP text encoder.
2. **Latent Noise Initialization:** A low-dimensional latent grid (e.g. $64 \times 64 \times 4$) filled with random Gaussian noise is initialized.
3. **Iterative Denoising (U-Net + Scheduler):** Over a series of inference steps (e.g. 15–20), the U-Net predicts the noise component at each timestep, conditioned on the text embeddings through cross-attention layers. The scheduler subtracts the noise.
4. **VAE Decoding:** The cleaned latent tensor is transformed by the Variational Autoencoder (VAE) decoder into a full-resolution RGB image.
5. **Display & Persistence:** The image is saved locally with a timestamp and displayed in the Gradio web UI.

---

## 6. Project Folder Structure

```
text-to-image-generator/
│
├── app.py                     # Main application entry point & Gradio web UI
├── requirements.txt           # Project dependencies
├── run.bat                    # One-click Windows batch launcher
├── run.ps1                    # One-click PowerShell launcher
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Version control ignore rules
├── LICENSE                    # MIT open-source license
│
├── generated_images/          # Output directory where generated images are saved
│   └── .gitkeep
│
├── docs/                      # Academic documentation & viva preparation
│   ├── project_overview.md    # Detailed MCA project overview report
│   ├── architecture.md        # Technical architecture and component breakdown
│   └── viva_questions.md      # 30 viva voce questions and answers
│
└── screenshots/               # Sample generated output images
    ├── 20261008_094959_a_cute_puppy_sitting_in_a_gard.png
    ├── 20261008_095035_a_futuristic_city_at_night_wit.png
    ├── 20261008_095112_a_peaceful_waterfall_surrounde.png
    ├── 20261008_095148_a_robot_studying_artificial_in.png
    ├── 20261008_095224_a_spaceship_traveling_through.png
    └── 20261008_095507_a_cute_puppy_sitting_in_a_beau.png
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.11 or higher installed on your machine.
- Internet connection (for initial one-time model weight download).

### Step 1: Open Terminal in Project Directory
Navigate into the project root directory:
```powershell
cd "e:\image Generation AI"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```
*(On Windows systems where `py` launcher is preferred: `py -3.11 -m venv venv`)*

### Step 3: Activate the Virtual Environment
- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Tip for CPU-Only Laptops:** If you do not have an NVIDIA GPU, you can install the lightweight CPU-only PyTorch build to save download time and disk space:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install diffusers transformers accelerate safetensors pillow gradio
> ```

---

## 8. Running the Application

### Option A: Standard Command
With the virtual environment activated:
```bash
python app.py
```

### Option B: One-Click Windows Launchers
Simply double-click or run:
- `run.bat` (Command Prompt / Explorer)
- `.\run.ps1` (PowerShell)

### Accessing the Web Interface
Once the model loads, the terminal will display:
```
* Running on local URL:  http://127.0.0.1:7860
```
Open your web browser and navigate to:
**`http://127.0.0.1:7860`**

---

## 9. Hardware & Device Support (CPU / CUDA Fallback)

The application automatically checks for hardware acceleration at startup:

```python
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16       # Half precision for GPU speed & VRAM savings
else:
    device = "cpu"
    dtype = torch.float32       # Full precision for CPU numerical stability
```

- **If an NVIDIA GPU with CUDA is present:** The pipeline runs on `cuda` with half-precision (`float16`), generating images in approximately 2 to 6 seconds.
- **If no GPU is present:** The application logs `CPU (No CUDA GPU detected) - Float32 Precision` and runs on the CPU. The application will not crash.
- CPU generation takes approximately 30 to 120 seconds depending on CPU core count and resolution.

---

## 10. Model Information & Dataset Clarification

### Pre-Trained Model Used
By default, this application utilizes **`segmind/tiny-sd`** (~650 MB), a distilled, highly optimized Latent Diffusion Model based on the Stable Diffusion architecture:
- Downloads quickly and fits easily within typical student laptop storage.
- Operates reliably within standard system RAM (4 GB–8 GB) without crashing.
- Can be changed to standard **`runwayml/stable-diffusion-v1-5`** simply by editing the `MODEL_ID` variable at the top of `app.py`.

### No Custom Dataset Required
**Important Academic Note:** This project does **not** train a model from scratch and does not require a custom image dataset. It uses pre-trained weights distributed under the open Hugging Face repository, demonstrating **zero-shot text-to-image inference**.

---

## 11. Verified Test Prompts

The implementation was tested and verified with the following prompts:

| # | Test Prompt | Resolution | Steps | Status |
| :- | :--- | :--- | :--- | :--- |
| 1 | *"A cute puppy sitting in a garden"* | 256x256 | 8 | Verified (Saved to disk) |
| 2 | *"A futuristic city at night with flying cars"* | 256x256 | 8 | Verified (Saved to disk) |
| 3 | *"A peaceful waterfall surrounded by mountains"* | 256x256 | 8 | Verified (Saved to disk) |
| 4 | *"A robot studying artificial intelligence in a classroom"* | 256x256 | 8 | Verified (Saved to disk) |
| 5 | *"A spaceship traveling through a colorful galaxy"* | 256x256 | 8 | Verified (Saved to disk) |
| 6 | *"A cute puppy sitting in a beautiful garden"* | 512x512 | 10 | Verified (Saved to disk) |

Sample outputs are stored in `screenshots/` and `generated_images/`.

---

## 12. Limitations

1. **CPU Latency:** Generating images on a CPU takes longer than on a GPU.
2. **Resolution Bounds:** Optimized for $256 \times 256$ to $512 \times 512$ resolutions; ultra-high resolutions require high-end GPU VRAM.
3. **Complex Text Rendering:** Diffusion models may struggle to render coherent written text or typography inside generated images.
4. **Spatial Logic:** Subtle relationships (e.g., "a green book under a yellow pen") may occasionally exhibit semantic bleeding.

---

## 13. Future Enhancements

- **Negative Prompts:** Allow users to specify concepts to exclude (e.g., *"blurry, low quality, deformed"*).
- **Image-to-Image (Img2Img):** Allow users to supply an initial reference sketch to guide generation.
- **LoRA (Low-Rank Adaptation):** Add modular style adapters for custom artistic styles.
- **Cloud Acceleration:** Toggle between local inference and cloud GPU endpoints.
- **Gallery & History:** Add a searchable gallery of previously generated images.

---

## 14. Responsible AI & Ethical Use

- **Intended Use:** This application is developed strictly for educational, academic, and research demonstration purposes.
- **Content Integrity:** Users should not use this tool to generate defamatory, violent, sexually explicit, or unlawful imagery.
- **Verification:** AI-synthesized imagery may contain artifacts or factual inaccuracies. Any media intended for public dissemination should be disclosed as AI-generated.

---

## 15. License

This project is licensed under the [MIT License](LICENSE).

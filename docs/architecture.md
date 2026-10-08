# System Architecture: Text-to-Image Generator Using Generative AI

This document provides a detailed breakdown of the system architecture, mathematical principles, component interactions, and execution pipeline for the **Text-to-Image Generator Using Generative AI** academic MCA project.

---

## 1. High-Level Architecture Flowchart

```
┌────────────────────────────────────────────────────────┐
│                        USER                            │
│           (Enters Natural Language Prompt)             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                      GRADIO UI                         │
│            • Prompt Textbox                            │
│            • Hyperparameter Sliders (Steps, Guidance)  │
│            • Status Display & Download Link            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  PROMPT VALIDATION                     │
│            • Whitespace Stripping                      │
│            • Non-Empty Check                           │
│            • Length Boundary (< 350 chars)             │
└───────────────────────────┬────────────────────────────┘
                            │ (Valid Prompt)
                            ▼
┌────────────────────────────────────────────────────────┐
│               TEXT PROCESSING SUBSYSTEM                │
│            • CLIP Tokenizer (Tokens: 77 max)           │
│            • CLIP Text Encoder (Transformer)           │
│            • Output: Text Embedding Vector (77 x 768)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             LATENT DIFFUSION MODEL (LDM)               │
│                                                        │
│   1. Initial Latent: Pure Gaussian Noise z_T ~ N(0,I) │
│                                                        │
│   2. Iterative Denoising Loop (U-Net + Scheduler)      │
│      ┌──────────────────────────────────────────────┐  │
│      │ U-Net Neural Network                         │  │
│      │ • Residual Blocks (Feature extraction)       │  │
│      │ • Cross-Attention (Conditioned on Text Embed)│  │
│      │ • Noise Prediction: ε_θ(z_t, t, prompt)      │  │
│      └──────────────────────┬───────────────────────┘  │
│                             │                          │
│      ▼ Scheduler Update Step: z_{t-1} = Step(z_t, ε)   │
│      (Repeated for N Inference Steps, e.g. 15-20)       │
│                                                        │
│   3. Final Clean Latent Representation z_0             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 VAE DECODER SUBSYSTEM                  │
│       • Maps Latent Space (64 x 64 x 4) to             │
│         Pixel Space (512 x 512 x 3 RGB)                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               PYTORCH INFERENCE ENGINE                 │
│       • Device Manager: CUDA (FP16) or CPU (FP32)      │
│       • Attention Slicing (Memory Optimization)        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   GENERATED IMAGE                      │
│       • Converted to PIL (Pillow) RGB Object           │
│       • Auto-Saved to: generated_images/               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               GRADIO UI DISPLAY & DOWNLOAD             │
│       • Displayed in Browser                           │
│       • Downloadable .png File Link                    │
│       • Execution Status & Timestamp Logged            │
└────────────────────────────────────────────────────────┘
```

---

## 2. Component-by-Component Explanation

### A. User Interface (Gradio Layer)
- Built with Python using `gradio.Blocks`.
- Provides an intuitive, responsive interface with no external frontend dependencies (no Node.js, React, or CSS configurations needed).
- Captures the natural-language prompt, optional inference steps (5–40), guidance scale (1.0–15.0), and resolution selection.

### B. Prompt Validation Layer
- Sanitizes the input string by stripping leading and trailing whitespace.
- Verifies that the string is non-empty.
- Ensures the prompt does not exceed 350 characters to prevent memory overhead and token buffer truncation.
- Returns clean, user-friendly feedback without causing backend exceptions.

### C. Text Processing Subsystem (CLIP Text Encoder)
Text diffusion models do not process raw text strings directly. They require continuous numerical representations:
1. **Tokenizer:** Converts the raw sentence into discrete token IDs using Byte-Pair Encoding (BPE), padded or truncated to a fixed sequence length of 77 tokens.
2. **Text Encoder:** A 12-layer Transformer neural network (CLIP ViT-L/14) maps the sequence of tokens into high-dimensional semantic feature vectors $\mathbf{c} \in \mathbb{R}^{77 \times 768}$.
3. These semantic vectors act as the conditioning context for every layer of the diffusion U-Net.

### D. Latent Space Representation (Why Latent Diffusion?)
Standard diffusion models (e.g., DDPM) operate directly in pixel space. For a $512 \times 512 \times 3$ image, that corresponds to $786,432$ values per step, making it computationally prohibitive for student laptops.

**Latent Diffusion Models (LDMs)** solve this problem by operating in a compressed **latent space** using a pre-trained **Variational Autoencoder (VAE)**:
- Compression factor: $8\times$ spatial compression.
- A $512 \times 512 \times 3$ image is compressed into a $64 \times 64 \times 4$ latent tensor.
- Number of values per step: $64 \times 64 \times 4 = 16,384$ (a **$48\times$ reduction** in computational complexity).
- This spatial compression is what allows the model to run on standard consumer CPUs and laptops.

### E. Denoising U-Net Neural Network
The core generative engine is an encoder-decoder convolutional network with skip connections:
- **Input:** A noisy latent tensor $\mathbf{z}_t$ and the current timestep $t$.
- **Cross-Attention Mechanism:** At each intermediate layer, spatial feature maps compute query-key-value attention with the text embedding vectors $\mathbf{c}$:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  where $Q$ originates from image latents and $K, V$ originate from the text embeddings.
- **Output:** Predicted noise residual $\mathbf{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})$.

### F. Noise Scheduler
- Manages the progressive transition from pure random Gaussian noise $\mathbf{z}_T \sim \mathcal{N}(0, \mathbf{I})$ to a coherent latent representation $\mathbf{z}_0$.
- Uses discrete numerical sampling steps (such as PNDM, DPMSolver, or Euler) to compute $\mathbf{z}_{t-1}$ from $\mathbf{z}_t$ and the predicted noise.
- **Guidance Scale (Classifier-Free Guidance, CFG):**
  Combines an unconditioned prediction with a text-conditioned prediction:
  $$\mathbf{\epsilon} = \mathbf{\epsilon}_{\text{uncond}} + s \cdot (\mathbf{\epsilon}_{\text{cond}} - \mathbf{\epsilon}_{\text{uncond}})$$
  where $s$ is the guidance scale (default: 7.5). Higher values force the image to adhere more strictly to the prompt.

### G. VAE Decoder
Once the $N$ denoising steps conclude, the clean latent representation $\mathbf{z}_0 \in \mathbb{R}^{64 \times 64 \times 4}$ is passed through the pre-trained VAE Decoder:
- Expands the $64 \times 64$ latent representations back into a full $512 \times 512 \times 3$ RGB pixel image.
- Normalizes pixel values into the standard $[0, 255]$ integer range.

### H. Hardware Abstraction & PyTorch Engine
- **Device Placement:** Automatically tests `torch.cuda.is_available()`.
  - If `True`: Binds pipeline to `"cuda"`, casts weights to `torch.float16`, and activates attention slicing.
  - If `False`: Binds pipeline to `"cpu"`, maintains `torch.float32` stability, and uses multi-threaded CPU tensor kernels.
- **Error Protection:** Wraps generation calls in `try...except` blocks with explicit handlers for `torch.cuda.OutOfMemoryError` and generic runtime faults.

### I. Storage and Delivery
- Generates a timestamped filename (e.g., `20261008_095507_a_cute_puppy_sitting_in_a_beau.png`).
- Saves image to `generated_images/` using Pillow.
- Supplies file reference to Gradio's `gr.File` component to enable direct browser downloads.

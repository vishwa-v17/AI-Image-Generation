"""
Text-to-Image Generator Using Generative AI (Optimized)
======================================================
An MCA Academic Project demonstrating high-performance diffusion-based Generative AI.

Key Optimizations:
- Pre-trained Lightweight Model: segmind/tiny-sd (~650 MB vs 4.2 GB of standard SD 1.5).
- Fast DPM-Solver++ Scheduler: Replaces default PNDM scheduler, enabling high quality in just 10 steps.
- Inference Mode: Runs under torch.inference_mode() with disabled gradient tracking.
- Memory Format: Channels-last memory layout on CPU for optimized convolutional performance.
- One-Time Startup Loading: Pipeline loaded once at startup; never reloaded on generation clicks.
- Hardware-Aware Execution: Auto-detects CUDA GPU (FP16) or falls back to multi-core CPU (FP32).
- Fast Mode vs Quality Mode: Simple toggle defaulting to Fast Mode (384x384, 10 steps).
- Accurate Real-Time Benchmarking: Uses time.perf_counter() to report exact generation duration.
- Robust Error Handling & Input Validation: Prevents empty prompts, length overflow, and crashes.
"""

import os
import sys
import re
import time
import uuid
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr

# Suppress Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==============================================================================
# 1. APPLICATION CONFIGURATION & MODE DEFINITIONS
# ==============================================================================

# Pre-trained model identifier from Hugging Face Hub
# "segmind/tiny-sd" is a 4x smaller, 2.5x faster distilled Stable Diffusion model.
# Total download size is ~650 MB (compared to 4.2 GB for full SD 1.5).
MODEL_ID = "segmind/tiny-sd"

# Output directory for saving generated images
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum prompt character limit to prevent token overflow
MAX_PROMPT_LENGTH = 350

# Generation Modes: Fast Mode (default) vs Quality Mode
MODE_CONFIGS = {
    "Fast Mode (384×384, 10 steps)": {
        "width": 384,
        "height": 384,
        "steps": 10,
        "guidance_scale": 7.0,
        "description": "Optimized for speed. Generates 384×384 in 10 steps using DPM-Solver++."
    },
    "Quality Mode (512×512, 20 steps)": {
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance_scale": 7.5,
        "description": "Higher fidelity. Generates 512×512 in 20 steps for enhanced detail."
    }
}
DEFAULT_MODE_KEY = "Fast Mode (384×384, 10 steps)"


# ==============================================================================
# 2. HARDWARE & DEVICE DETECTION
# ==============================================================================

def detect_device():
    """
    Detects whether an NVIDIA CUDA GPU is available or falls back to CPU.
    
    Returns:
        tuple: (device_str, torch_dtype, device_display_name, hardware_note)
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # FP16 half-precision for GPU speed & memory savings
        gpu_name = torch.cuda.get_device_name(0)
        display_name = f"CUDA (GPU: {gpu_name})"
        note = "Hardware acceleration active (FP16 enabled)."
    else:
        device = "cpu"
        dtype = torch.float32  # FP32 standard precision for CPU numerical stability
        display_name = "CPU"
        note = "CPU mode: generation may take longer depending on hardware."

    return device, dtype, display_name, note


DEVICE, TORCH_DTYPE, DEVICE_DISPLAY, HARDWARE_NOTE = detect_device()
print(f"[*] Detected Hardware: {DEVICE_DISPLAY}")
print(f"[*] Hardware Note: {HARDWARE_NOTE}")


# ==============================================================================
# 3. OPTIMIZED MODEL INITIALIZATION (LOADED ONCE AT STARTUP)
# ==============================================================================

def load_optimized_pipeline():
    """
    Loads the pre-trained diffusion model once at startup and applies inference optimizations:
    1. Loads pipeline with appropriate hardware dtype (FP16 for GPU, FP32 for CPU).
    2. Equips DPMSolverMultistepScheduler (DPM-Solver++ Karras) for 2x faster convergence.
    3. Formats U-Net and VAE with channels-last memory layout on CPU for faster convolution.
    4. Enables attention slicing on CUDA to prevent out-of-memory errors.
    
    Returns:
        StableDiffusionPipeline: High-performance diffusion pipeline.
    """
    print(f"\n[*] Loading model '{MODEL_ID}' onto {DEVICE}...")
    start_time = time.perf_counter()
    
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            dtype=TORCH_DTYPE,
            safety_checker=None
        )
        pipeline = pipeline.to(DEVICE)
        
        # Optimization A: Fast DPM-Solver++ Multistep Scheduler (converges in 10-15 steps)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True
        )
        
        # Optimization B: Memory layout optimization for CPU / GPU
        if DEVICE == "cpu":
            # Channels-last memory format optimizes PyTorch CPU convolution kernels
            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)
        elif DEVICE == "cuda":
            pipeline.enable_attention_slicing()
            
        load_time = time.perf_counter() - start_time
        print(f"[+] Model loaded and optimized in {load_time:.2f} seconds!")
        return pipeline

    except Exception as e:
        print(f"[!] Error loading model: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to load diffusion pipeline '{MODEL_ID}'. Details: {e}")


# Global pipeline instance: loaded ONCE at module launch
PIPELINE = load_optimized_pipeline()


# ==============================================================================
# 4. PROMPT VALIDATION & SANITIZATION
# ==============================================================================

def validate_prompt(prompt: str) -> tuple[bool, str, str]:
    """
    Validates user text prompt to prevent empty submissions or excessive length.
    
    Returns:
        tuple: (is_valid, cleaned_prompt, status_message)
    """
    if prompt is None:
        return False, "", "Please enter a prompt before generating an image."

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return False, "", "Please enter a prompt before generating an image."

    if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
        return False, cleaned_prompt, (
            f"Prompt is too long ({len(cleaned_prompt)} characters). "
            f"Please shorten it to under {MAX_PROMPT_LENGTH} characters."
        )

    return True, cleaned_prompt, "Prompt is valid."


# ==============================================================================
# 5. IMAGE SAVING UTILITY
# ==============================================================================

def save_generated_image(image: Image.Image, prompt: str) -> str:
    """
    Saves the generated image to generated_images/ with a unique timestamp and UUID.
    Never overwrites existing images.
    
    Returns:
        str: Absolute file path to the saved PNG image.
    """
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', prompt.lower())[:25].strip('_')
    if not slug:
        slug = "image"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    filename = f"{timestamp}_{slug}_{unique_suffix}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    image.save(filepath, format="PNG")
    print(f"[+] Image saved locally: {filepath}")
    return filepath


# ==============================================================================
# 6. HIGH-PERFORMANCE INFERENCE FUNCTION
# ==============================================================================

def generate_image(prompt: str, mode_choice: str):
    """
    Executes single-request image generation with precise timing measurement.
    
    Returns:
        tuple: (PIL.Image or None, status_text, generation_time_text, saved_file_path)
    """
    # Step 1: Validate prompt
    is_valid, clean_prompt, validation_msg = validate_prompt(prompt)
    if not is_valid:
        return None, validation_msg, "N/A", None

    # Step 2: Extract mode configuration (Fast Mode vs Quality Mode)
    config = MODE_CONFIGS.get(mode_choice, MODE_CONFIGS[DEFAULT_MODE_KEY])
    width = config["width"]
    height = config["height"]
    steps = config["steps"]
    guidance = config["guidance_scale"]

    print(f"\n[*] Generating Image | Prompt: '{clean_prompt}'")
    print(f"[*] Configuration: {mode_choice} -> {width}×{height}, {steps} steps, guidance={guidance}, device={DEVICE}")

    start_timer = time.perf_counter()

    try:
        # Step 3: Run inference under torch.inference_mode() (no autograd overhead)
        with torch.inference_mode():
            result = PIPELINE(
                prompt=clean_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height
            )

        elapsed_seconds = time.perf_counter() - start_timer
        generated_image = result.images[0]

        # Step 4: Save image to local disk
        saved_file_path = save_generated_image(generated_image, clean_prompt)

        status_text = (
            f"Image generated successfully! "
            f"Mode: {mode_choice} | Resolution: {width}×{height} | Steps: {steps}"
        )
        time_text = f"{elapsed_seconds:.2f} seconds"
        print(f"[+] Completed in {time_text} on {DEVICE.upper()}!")

        return generated_image, status_text, time_text, saved_file_path

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_msg = "Error: GPU ran out of memory. Try Fast Mode or lower resolution."
        print(f"[!] {error_msg}", file=sys.stderr)
        return None, error_msg, "Failed", None

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(f"[!] {error_msg}", file=sys.stderr)
        return None, error_msg, "Failed", None


# ==============================================================================
# 7. GRADIO USER INTERFACE
# ==============================================================================

def create_interface():
    """
    Builds the clean, responsive Gradio web interface.
    """
    with gr.Blocks(title="Text-to-Image Generator Using Generative AI") as demo:
        # Header Section
        gr.Markdown(
            """
            # Text-to-Image Generator Using Generative AI
            ### Generate images from natural-language prompts using a lightweight pre-trained Diffusion Model
            """
        )

        # Hardware & Performance Banner
        status_color = "🟢" if DEVICE == "cuda" else "🟡"
        gr.Markdown(
            f"""
            > **Hardware:** {status_color} `Device: {DEVICE_DISPLAY}` | **Model:** `{MODEL_ID}`  
            > *Notice:* {HARDWARE_NOTE}
            """
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Enter your image prompt",
                    placeholder="Example: A futuristic city at sunset with flying cars",
                    lines=3,
                    max_lines=5
                )

                mode_selector = gr.Radio(
                    choices=list(MODE_CONFIGS.keys()),
                    value=DEFAULT_MODE_KEY,
                    label="Generation Mode",
                    info="Fast Mode uses 384×384 with 10 steps for quickest generation. Quality Mode uses 512×512 with 20 steps."
                )

                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg"
                )

                # Quick Example Prompts
                gr.Examples(
                    examples=[
                        ["A futuristic city at sunset"],
                        ["A beautiful mountain landscape"],
                        ["A robot working in a modern laboratory"],
                        ["A cute puppy sitting in a garden"],
                        ["A spaceship traveling through a colorful galaxy"]
                    ],
                    inputs=[prompt_input],
                    label="Sample Prompts (Click to test)"
                )

            # Right Column: Outputs, Timers & Download
            with gr.Column(scale=5):
                image_output = gr.Image(
                    label="Generated Image",
                    type="pil",
                    interactive=False
                )

                with gr.Row():
                    device_box = gr.Textbox(
                        label="Device",
                        value=DEVICE_DISPLAY,
                        interactive=False,
                        scale=1
                    )
                    time_box = gr.Textbox(
                        label="Generation Time",
                        value="Ready",
                        interactive=False,
                        scale=1
                    )

                status_box = gr.Textbox(
                    label="Status",
                    value=f"Model ready on {DEVICE_DISPLAY}. Select mode and click 'Generate Image'.",
                    interactive=False
                )

                download_file = gr.File(
                    label="Download Generated Image (.png)",
                    interactive=False
                )

        # Connect button click with single-generation execution & UI queue lock
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, mode_selector],
            outputs=[image_output, status_box, time_box, download_file]
        )

        # Academic Project Footer
        gr.Markdown(
            """
            ---
            **MCA Academic Mini-Project:** Text-to-Image Generator Using Generative AI  
            *Engineered with PyTorch, Hugging Face Diffusers, DPM-Solver++, and Gradio.*  
            **Responsible AI Notice:** For academic demonstration only. Do not generate harmful or unlawful imagery.
            """
        )

    return demo


# ==============================================================================
# 8. APPLICATION LAUNCHER
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Text-to-Image Generator Using Generative AI")
    print(f" Active Device: {DEVICE_DISPLAY}")
    print(f" Pre-Trained Model: {MODEL_ID}")
    print(f" Default Mode: {DEFAULT_MODE_KEY}")
    print("=" * 60 + "\n")

    app = create_interface()
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    
    # Launch Gradio app locally; auto-allocates port 7860 or next free port
    app.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        theme=theme
    )

"""
Text-to-Image Generator Using Generative AI
============================================
An MCA Academic Project demonstrating diffusion-based Generative AI.

Features:
- Natural-language text prompt to image generation.
- Automatic hardware detection (CUDA GPU with float16 or CPU with float32).
- One-time model loading at application startup.
- Safe prompt validation and error handling.
- Local image saving and download functionality.
- Clean and interactive Gradio web interface.

Technology Stack:
- Python 3.11+
- PyTorch (Inference & Tensor Operations)
- Hugging Face Diffusers (Stable Diffusion Pipeline)
- Hugging Face Transformers (CLIP Text Encoder & Tokenizer)
- Accelerate & Safetensors (Optimized Model Weight Loading)
- Pillow (Image Processing & Saving)
- Gradio (Interactive Web UI)
"""

import os
import sys
import re
import time
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Suppress Hugging Face symlinks warning on Windows systems without Developer Mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==============================================================================
# 1. APPLICATION CONFIGURATION
# ==============================================================================

# Pre-trained model identifier from Hugging Face Hub
# "segmind/tiny-sd" is a distilled, lightweight (~650 MB) version of Stable Diffusion.
# It is specifically chosen for academic demonstration because it downloads quickly,
# runs reliably on standard student laptops with CPU, and performs excellently on GPU.
# (To use full SD 1.5 instead, simply change to: "runwayml/stable-diffusion-v1-5")
MODEL_ID = "segmind/tiny-sd"

# Default generation parameters suitable for student laptops
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS_CPU = 15
DEFAULT_STEPS_GPU = 20
DEFAULT_GUIDANCE_SCALE = 7.5

# Directory to save all generated images locally
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum allowed prompt length (characters) to prevent prompt injection or memory overload
MAX_PROMPT_LENGTH = 350


# ==============================================================================
# 2. HARDWARE DETECTION (CUDA GPU vs CPU FALLBACK)
# ==============================================================================

def detect_device():
    """
    Detects whether an NVIDIA CUDA GPU is available or falls back to CPU.
    
    Returns:
        tuple: (device_str, torch_dtype, hardware_description)
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Half precision for GPU to save VRAM and speed up compute
        gpu_name = torch.cuda.get_device_name(0)
        desc = f"CUDA GPU ({gpu_name}) - Float16 Precision"
        default_steps = DEFAULT_STEPS_GPU
    else:
        device = "cpu"
        dtype = torch.float32  # Standard float32 precision for CPU stability
        desc = "CPU (No CUDA GPU detected) - Float32 Precision"
        default_steps = DEFAULT_STEPS_CPU

    return device, dtype, desc, default_steps


# Detect hardware on script import
DEVICE, TORCH_DTYPE, HARDWARE_INFO, DEFAULT_STEPS = detect_device()
print(f"[*] Detected Hardware: {HARDWARE_INFO}")


# ==============================================================================
# 3. MODEL INITIALIZATION (LOAD ONCE AT STARTUP)
# ==============================================================================

def load_model():
    """
    Loads the pre-trained diffusion pipeline from Hugging Face Diffusers.
    
    This function is executed once when the application starts, preventing
    redundant model reloading on every user request.
    
    Returns:
        StableDiffusionPipeline: Ready-to-use PyTorch diffusion pipeline.
    """
    print(f"[*] Loading model '{MODEL_ID}' onto {DEVICE}...")
    start_time = time.time()
    
    try:
        # Load the pre-trained diffusion pipeline
        # - torch_dtype controls precision (float16 on GPU, float32 on CPU)
        # - safety_checker=None is used for faster startup in educational environments
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            dtype=TORCH_DTYPE,
            safety_checker=None
        )
        
        # Move pipeline to the detected hardware target (GPU or CPU)
        pipeline = pipeline.to(DEVICE)
        
        # Optimize memory usage for low VRAM situations when using CUDA
        if DEVICE == "cuda":
            pipeline.enable_attention_slicing()
            
        load_time = time.time() - start_time
        print(f"[+] Model loaded successfully in {load_time:.2f} seconds!")
        return pipeline
        
    except Exception as e:
        print(f"[!] Error loading model: {e}", file=sys.stderr)
        raise RuntimeError(
            f"Failed to load the diffusion model '{MODEL_ID}'. "
            f"Please verify internet connectivity and package installations. Details: {e}"
        )


# Global pipeline instance loaded at application startup
PIPELINE = load_model()


# ==============================================================================
# 4. PROMPT VALIDATION
# ==============================================================================

def validate_prompt(prompt: str) -> tuple[bool, str, str]:
    """
    Validates user text prompt to prevent empty submissions or excessive length.
    
    Args:
        prompt: Raw user input text.
        
    Returns:
        tuple: (is_valid, cleaned_prompt, message)
    """
    if prompt is None:
        return False, "", "Please enter a prompt before generating an image."
        
    cleaned_prompt = prompt.strip()
    
    # Check for empty string
    if not cleaned_prompt:
        return False, "", "Please enter a prompt before generating an image."
        
    # Check for excessive length
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
    Saves a generated PIL image to the local output folder with a descriptive name.
    
    Args:
        image: Generated PIL Image.
        prompt: Cleaned prompt text used for naming.
        
    Returns:
        str: Absolute file path of the saved PNG image.
    """
    # Create safe slug from prompt (first 30 characters, alphanumeric only)
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', prompt.lower())[:30].strip('_')
    if not slug:
        slug = "image"
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{slug}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    image.save(filepath, format="PNG")
    print(f"[+] Image saved to: {filepath}")
    return filepath


# ==============================================================================
# 6. INFERENCE & IMAGE GENERATION FUNCTION
# ==============================================================================

def generate_image(prompt: str, steps: int = DEFAULT_STEPS, guidance_scale: float = DEFAULT_GUIDANCE_SCALE, size_option: str = "512x512"):
    """
    Generates an image from a text prompt using the pre-trained diffusion model.
    
    Workflow:
    1. Validate input prompt.
    2. Parse target resolution (width, height).
    3. Run PyTorch diffusion inference loop.
    4. Save output image to disk.
    5. Return generated image, user-friendly status, and download file path.
    
    Returns:
        tuple: (PIL.Image or None, status_message_str, saved_filepath_or_None)
    """
    # Step 1: Validate prompt
    is_valid, clean_prompt, validation_msg = validate_prompt(prompt)
    if not is_valid:
        return None, validation_msg, None

    # Step 2: Parse resolution
    try:
        width_str, height_str = size_option.split("x")
        width, height = int(width_str), int(height_str)
    except Exception:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

    # Step 3: Run inference
    start_time = time.time()
    print(f"\n[*] Generating image for prompt: '{clean_prompt}'")
    print(f"[*] Parameters: steps={steps}, guidance_scale={guidance_scale}, resolution={width}x{height}, device={DEVICE}")

    try:
        # Run inference using the pre-loaded diffusion pipeline
        # - prompt: input text description
        # - num_inference_steps: denoising iterations
        # - guidance_scale: adherence to text prompt vs creative freedom
        result = PIPELINE(
            prompt=clean_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=width,
            height=height
        )
        
        generated_image = result.images[0]
        elapsed_time = time.time() - start_time
        
        # Step 4: Save image to local directory
        saved_path = save_generated_image(generated_image, clean_prompt)
        
        status_msg = (
            f"Image generated successfully in {elapsed_time:.1f}s! "
            f"(Resolution: {width}x{height}, Steps: {steps}, Device: {DEVICE.upper()})"
        )
        print(f"[+] {status_msg}")
        return generated_image, status_msg, saved_path

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_msg = "Error: GPU ran out of memory (OOM). Please try a smaller image size or fewer steps."
        print(f"[!] {error_msg}", file=sys.stderr)
        return None, error_msg, None

    except Exception as e:
        error_msg = f"Error while generating image: {str(e)}"
        print(f"[!] {error_msg}", file=sys.stderr)
        return None, error_msg, None


# ==============================================================================
# 7. GRADIO WEB INTERFACE CONSTRUCTION
# ==============================================================================

def create_interface():
    """
    Builds the student-friendly Gradio web application UI.
    """
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    )

    with gr.Blocks(title="Text-to-Image Generator") as demo:
        # Header Section
        gr.Markdown(
            """
            # Text-to-Image Generator
            ### Generate images from natural-language prompts using Generative AI
            """
        )

        # Hardware and Model Status Banner
        hardware_badge = "🟢 GPU Accelerated" if DEVICE == "cuda" else "🟡 CPU Execution (Fallback Mode)"
        gr.Markdown(
            f"""
            > **System Status:** {hardware_badge} | **Hardware:** `{HARDWARE_INFO}` | **Model:** `{MODEL_ID}`
            """
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Enter your image prompt",
                    placeholder="Example: A futuristic city at night with flying cars",
                    lines=3,
                    max_lines=5
                )

                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg"
                )

                # Curated Examples for Academic Demonstration
                gr.Examples(
                    examples=[
                        ["A cute puppy sitting in a beautiful garden"],
                        ["A futuristic city at night with flying cars"],
                        ["A peaceful waterfall surrounded by mountains"],
                        ["A robot studying artificial intelligence in a classroom"],
                        ["A spaceship traveling through a colorful galaxy"]
                    ],
                    inputs=[prompt_input],
                    label="Click an example prompt to test"
                )

                # Optional Advanced Settings Accordion
                with gr.Accordion("⚙️ Generation Settings (Optional)", open=False):
                    gr.Markdown("*Fine-tune hyperparameters for diffusion inference.*")
                    steps_slider = gr.Slider(
                        minimum=5,
                        maximum=40,
                        step=1,
                        value=DEFAULT_STEPS,
                        label=f"Inference Steps (Default: {DEFAULT_STEPS})"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        step=0.5,
                        value=DEFAULT_GUIDANCE_SCALE,
                        label=f"Guidance Scale (Default: {DEFAULT_GUIDANCE_SCALE})"
                    )
                    size_dropdown = gr.Dropdown(
                        choices=["256x256", "384x384", "512x512"],
                        value="512x512",
                        label="Image Resolution (Default: 512x512)"
                    )

            # Right Column: Outputs & Download
            with gr.Column(scale=5):
                image_output = gr.Image(
                    label="Generated Image",
                    type="pil",
                    interactive=False
                )

                status_output = gr.Textbox(
                    label="Status",
                    value=f"Model ready on {DEVICE.upper()}. Enter a prompt and click 'Generate Image'.",
                    interactive=False
                )

                download_output = gr.File(
                    label="Download Generated Image (.png)",
                    interactive=False
                )

        # Wire the Generate button to the generate_image function
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, steps_slider, guidance_slider, size_dropdown],
            outputs=[image_output, status_output, download_output]
        )

        # Footer & Academic Reference Notes
        gr.Markdown(
            """
            ---
            **Academic Mini-Project (MCA):** Text-to-Image Generator Using Generative AI  
            *Powered by PyTorch, Hugging Face Diffusers, and Latent Diffusion Models.*  
            **Responsible AI Notice:** Use responsibly for educational purposes. Do not generate unlawful or harmful media.
            """
        )

    return demo


# ==============================================================================
# 8. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Text-to-Image Generator - Generative AI Academic Project")
    print(f" Running on device: {DEVICE.upper()}")
    print(f" Loaded model: {MODEL_ID}")
    print("=" * 60 + "\n")
    
    app = create_interface()
    # Launch Gradio interface locally (auto-allocates 7860 or next free port, opens browser)
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    app.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        theme=theme
    )

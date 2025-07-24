#!/usr/bin/env python3
"""
Shared Space Generator - A Gradio Space for generating cozy interior images
inspired by creative figures with color palette extraction.
Fixed version with better error handling.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Critical: Set cache directory to writable location BEFORE any imports
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache'

# Create cache directories
os.makedirs('/tmp/hf_cache', exist_ok=True)

# Stub out the deprecated cached_download to prevent import errors
try:
    import huggingface_hub
    if hasattr(huggingface_hub, 'cached_download'):
        def stub_cached_download(*args, **kwargs):
            from huggingface_hub import hf_hub_download
            if 'filename' in kwargs:
                return hf_hub_download(kwargs.get('repo_id', args[0]), kwargs['filename'])
            return hf_hub_download(*args, **kwargs)
        huggingface_hub.cached_download = stub_cached_download
except ImportError:
    print("Warning: huggingface_hub not available")

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from sklearn.cluster import KMeans
import io
import base64
from typing import List, Tuple, Optional
import traceback

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global pipeline variable
pipeline = None

def setup_pipeline():
    """Setup the Stable Diffusion pipeline with optimizations"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    try:
        print("Loading Stable Diffusion pipeline...")
        
        # Try to import diffusers
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            print("Error: diffusers not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers"])
            from diffusers import StableDiffusionPipeline
        
        # Use a smaller, faster model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load pipeline with optimizations
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir='/tmp/hf_cache',
            low_cpu_mem_usage=True,
            variant="fp16" if device == "cuda" else None,
            safety_checker=None,  # Disable safety checker for speed
            requires_safety_checker=False
        )
        
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            try:
                pipeline.enable_attention_slicing()
                pipeline.enable_model_cpu_offload()
            except Exception as e:
                print(f"Warning: Could not enable CUDA optimizations: {e}")
        else:
            # For CPU, use different optimizations
            pipeline.enable_attention_slicing(1)
        
        print("Pipeline loaded successfully!")
        return pipeline
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        traceback.print_exc()
        return None

def get_reference_images():
    """Get reference images for each creator"""
    creators = {
        "Satyajit Ray": {
            "colors": ["#8B4513", "#DEB887", "#F4A460", "#D2691E", "#CD853F"],
            "description": "Warm earth tones reflecting Bengali cinema aesthetics"
        },
        "Hayao Miyazaki": {
            "colors": ["#98FB98", "#87CEEB", "#F0E68C", "#FFB6C1", "#DDA0DD"],
            "description": "Natural, whimsical colors from Studio Ghibli"
        },
        "Orhan Pamuk": {
            "colors": ["#4682B4", "#B0C4DE", "#778899", "#2F4F4F", "#708090"],
            "description": "Istanbul blues and literary melancholy"
        },
        "Humayun Ahmed": {
            "colors": ["#228B22", "#32CD32", "#90EE90", "#8FBC8F", "#6B8E23"],
            "description": "Verdant greens of rural Bangladesh"
        }
    }
    
    return creators

def create_color_swatch(colors: List[str], title: str) -> Image.Image:
    """Create a visual color swatch"""
    try:
        width, height = 400, 100
        swatch = Image.new('RGB', (width, height), 'white')
        
        color_width = width // len(colors)
        
        draw = ImageDraw.Draw(swatch)
        
        for i, color in enumerate(colors):
            x1 = i * color_width
            x2 = (i + 1) * color_width
            draw.rectangle([x1, 0, x2, height-20], fill=color)
        
        # Add title
        try:
            # Try to use a default font
            font = ImageFont.load_default()
            draw.text((10, height-15), title, fill='black', font=font)
        except:
            draw.text((10, height-15), title, fill='black')
        
        return swatch
    
    except Exception as e:
        print(f"Error creating color swatch: {e}")
        # Return a simple fallback image
        fallback = Image.new('RGB', (400, 100), 'lightgray')
        draw = ImageDraw.Draw(fallback)
        draw.text((10, 40), f"Color palette for {title}", fill='black')
        return fallback

def create_fallback_image(creator: str, error_msg: str = ""):
    """Create a fallback image when Stable Diffusion fails"""
    try:
        colors = {
            "Satyajit Ray": "#8B4513",
            "Hayao Miyazaki": "#98FB98",
            "Orhan Pamuk": "#4682B4", 
            "Humayun Ahmed": "#228B22"
        }
        
        color = colors.get(creator, "#888888")
        
        # Create 512x512 image
        img = Image.new('RGB', (512, 512), color)
        draw = ImageDraw.Draw(img)
        
        # Add decorative elements
        draw.rectangle([50, 50, 462, 462], outline='white', width=3)
        draw.rectangle([80, 80, 432, 432], outline='white', width=1)
        
        # Add text
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((100, 200), f"Cozy Interior", fill='white', font=font)
        draw.text((100, 230), f"Inspired by {creator}", fill='white', font=font)
        draw.text((100, 280), "Style Preview", fill='white', font=font)
        
        if error_msg:
            draw.text((100, 350), "Note: Using style preview", fill='white', font=font)
            draw.text((100, 370), "(AI generation unavailable)", fill='white', font=font)
        
        return img
        
    except Exception as e:
        print(f"Error creating fallback image: {e}")
        # Ultimate fallback
        return Image.new('RGB', (512, 512), 'lightgray')

def generate_room_image(creator: str, custom_prompt: str = "") -> Tuple[Optional[Image.Image], str]:
    """Generate a cozy interior image inspired by the selected creator"""
    
    try:
        # Try to setup pipeline if not already done
        if pipeline is None:
            setup_result = setup_pipeline()
            if setup_result is None:
                # Use fallback image instead of failing
                fallback_img = create_fallback_image(creator, "AI model not available")
                return fallback_img, "Using style preview (AI generation not available)"
        
        # Base prompts for each creator
        base_prompts = {
            "Satyajit Ray": "cozy interior in the style of Bengali cinema, warm lighting, traditional furniture, vintage aesthetic, cinematic composition",
            "Hayao Miyazaki": "cozy interior inspired by Studio Ghibli, whimsical furniture, natural elements, soft lighting, magical atmosphere",
            "Orhan Pamuk": "cozy interior with Istanbul influences, literary atmosphere, blue tones, contemplative space, book-filled room",
            "Humayun Ahmed": "cozy interior with rural Bangladesh influences, natural materials, green accents, peaceful domestic space"
        }
        
        # Construct the prompt
        base_prompt = base_prompts.get(creator, "cozy interior room")
        if custom_prompt:
            prompt = f"{base_prompt}, {custom_prompt}"
        else:
            prompt = base_prompt
        
        # Add quality modifiers
        prompt += ", high quality, detailed, warm and inviting, photorealistic"
        
        print(f"Generating image with prompt: {prompt}")
        
        # Generate image with optimized settings
        try:
            if device == "cuda":
                with torch.autocast(device):
                    image = pipeline(
                        prompt,
                        num_inference_steps=15,  # Reduced for speed
                        guidance_scale=7.5,
                        width=512,
                        height=512,
                        negative_prompt="blurry, low quality, distorted"
                    ).images[0]
            else:
                # CPU generation
                image = pipeline(
                    prompt,
                    num_inference_steps=10,  # Even more reduced for CPU
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
            
            return image, "Image generated successfully!"
            
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory, falling back to style preview")
            fallback_img = create_fallback_image(creator, "Memory limit exceeded")
            return fallback_img, "Using style preview (memory limit exceeded)"
            
        except Exception as e:
            print(f"Pipeline generation error: {e}")
            fallback_img = create_fallback_image(creator, str(e))
            return fallback_img, f"Using style preview (generation error: {str(e)[:50]}...)"
            
    except Exception as e:
        error_msg = f"Error in generate_room_image: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        fallback_img = create_fallback_image(creator, "Error occurred")
        return fallback_img, f"Using style preview (error: {str(e)[:50]}...)"

def process_generation(creator: str, custom_prompt: str = "") -> Tuple[Optional[Image.Image], Image.Image, str]:
    """Main processing function"""
    
    try:
        # Generate the room image
        room_image, status = generate_room_image(creator, custom_prompt)
        
        # Get creator's color palette
        creators_data = get_reference_images()
        creator_data = creators_data.get(creator, creators_data["Satyajit Ray"])
        
        # Create color swatch
        palette_colors = creator_data["colors"]
        color_swatch = create_color_swatch(
            palette_colors, 
            f"{creator} - {creator_data['description']}"
        )
        
        return room_image, color_swatch, status
        
    except Exception as e:
        error_msg = f"Error in process_generation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # Return fallback content
        fallback_img = create_fallback_image(creator, "Processing error")
        fallback_palette = create_color_swatch(["#888888"] * 5, "Error loading palette")
        return fallback_img, fallback_palette, f"Error: {str(e)[:100]}..."

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Shared Space Generator",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .creator-info { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏠 Shared Space Generator
        
        Generate cozy interior spaces inspired by renowned creators and see their characteristic color palettes.
        Choose from Satyajit Ray, Hayao Miyazaki, Orhan Pamuk, or Humayun Ahmed.
        
        *Note: If AI generation is unavailable, you'll see style previews instead.*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                creator_dropdown = gr.Dropdown(
                    choices=["Satyajit Ray", "Hayao Miyazaki", "Orhan Pamuk", "Humayun Ahmed"],
                    value="Satyajit Ray",
                    label="Choose Creator",
                    info="Select a creator to inspire your interior design"
                )
                
                custom_prompt = gr.Textbox(
                    label="Additional Prompt (Optional)",
                    placeholder="Add specific elements like 'with plants' or 'modern furniture'...",
                    lines=2
                )
                
                generate_btn = gr.Button("🎨 Generate Space", variant="primary", size="lg")
                
                with gr.Accordion("Creator Info", open=False):
                    creators_info = get_reference_images()
                    info_text = "\n\n".join([
                        f"**{name}**: {data['description']}"
                        for name, data in creators_info.items()
                    ])
                    gr.Markdown(info_text, elem_classes="creator-info")
                
                gr.Markdown("""
                ### Troubleshooting
                - First generation may take 1-2 minutes
                - If you see "style preview", the AI model is loading
                - Try simpler prompts if generation fails
                """)
            
            with gr.Column(scale=2):
                with gr.Row():
                    room_output = gr.Image(
                        label="Generated Interior",
                        height=400,
                        show_download_button=True
                    )
                    
                    palette_output = gr.Image(
                        label="Creator's Color Palette",
                        height=400
                    )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
        
        # Event handlers
        generate_btn.click(
            fn=process_generation,
            inputs=[creator_dropdown, custom_prompt],
            outputs=[room_output, palette_output, status_output],
            show_progress=True
        )
        
        # Auto-generate on creator change
        creator_dropdown.change(
            fn=lambda creator: process_generation(creator, ""),
            inputs=[creator_dropdown],
            outputs=[room_output, palette_output, status_output]
        )
    
    return demo

if __name__ == "__main__":
    # Initialize with a test to make sure basic functionality works
    try:
        print("Testing basic functionality...")
        test_image, test_palette, test_status = process_generation("Satyajit Ray", "")
        print(f"Test result: {test_status}")
    except Exception as e:
        print(f"Test failed: {e}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
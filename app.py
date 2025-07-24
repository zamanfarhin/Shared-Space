#!/usr/bin/env python3
"""
Shared Space Generator - A Gradio Space for generating cozy interior images
inspired by creative figures with color palette extraction.
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
import huggingface_hub
if hasattr(huggingface_hub, 'cached_download'):
    def stub_cached_download(*args, **kwargs):
        from huggingface_hub import hf_hub_download
        if 'filename' in kwargs:
            return hf_hub_download(kwargs.get('repo_id', args[0]), kwargs['filename'])
        return hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = stub_cached_download

import torch
import numpy as np
from PIL import Image
import gradio as gr
from sklearn.cluster import KMeans
import io
import base64
from typing import List, Tuple, Optional

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
        from diffusers import StableDiffusionPipeline
        
        print("Loading Stable Diffusion pipeline...")
        
        # Use a smaller, faster model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load pipeline with optimizations
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir='/tmp/hf_cache',
            low_cpu_mem_usage=True,
            variant="fp16" if device == "cuda" else None
        )
        
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_model_cpu_offload()
        else:
            # For CPU, use different optimizations
            pipeline.enable_attention_slicing(1)
        
        print("Pipeline loaded successfully!")
        return pipeline
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

def get_reference_images():
    """Get reference images for each creator"""
    # In a real implementation, you would have actual reference images
    # For now, we'll create placeholder color palettes
    
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

def extract_color_palette(colors: List[str], num_colors: int = 5) -> List[str]:
    """Extract dominant colors (simplified for this demo)"""
    return colors[:num_colors]

def create_color_swatch(colors: List[str], title: str) -> Image.Image:
    """Create a visual color swatch"""
    width, height = 400, 100
    swatch = Image.new('RGB', (width, height), 'white')
    
    color_width = width // len(colors)
    
    from PIL import ImageDraw, ImageFont
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

def generate_room_image(creator: str, custom_prompt: str = "") -> Tuple[Optional[Image.Image], str]:
    """Generate a cozy interior image inspired by the selected creator"""
    
    if pipeline is None:
        setup_result = setup_pipeline()
        if setup_result is None:
            return None, "Failed to load the image generation model. Please try again."
    
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
    
    try:
        print(f"Generating image with prompt: {prompt}")
        
        # Generate image with optimized settings
        with torch.autocast(device):
            image = pipeline(
                prompt,
                num_inference_steps=20,  # Reduced for speed
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
        
        return image, "Image generated successfully!"
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        return None, error_msg

def process_generation(creator: str, custom_prompt: str = "") -> Tuple[Optional[Image.Image], Image.Image, str]:
    """Main processing function"""
    
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

# Initialize pipeline on startup (but don't block if it fails)
try:
    print("Initializing pipeline...")
    setup_pipeline()
    print("Pipeline initialized successfully!")
except Exception as e:
    print(f"Pipeline initialization failed: {e}")
    print("Will try to initialize on first use.")

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
                    lines=2
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
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
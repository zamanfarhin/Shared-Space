#!/usr/bin/env python3
"""
Minimal test version of Shared Space Generator
"""

import os
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: Set cache FIRST
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache'

# Create cache dir
os.makedirs('/tmp/hf_cache', exist_ok=True)

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

def create_test_image(creator: str, prompt: str = ""):
    """Create a simple test image instead of using Stable Diffusion"""
    
    # Create a simple colored rectangle as placeholder
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
    
    # Add some text
    draw.rectangle([50, 50, 462, 462], outline='white', width=5)
    draw.text((100, 200), f"Test Room", fill='white')
    draw.text((100, 250), f"Inspired by {creator}", fill='white')
    if prompt:
        draw.text((100, 300), f"Custom: {prompt[:20]}", fill='white')
    
    return img

def create_color_palette(creator: str):
    """Create color palette for the creator"""
    palettes = {
        "Satyajit Ray": ["#8B4513", "#DEB887", "#F4A460", "#D2691E", "#CD853F"],
        "Hayao Miyazaki": ["#98FB98", "#87CEEB", "#F0E68C", "#FFB6C1", "#DDA0DD"],
        "Orhan Pamuk": ["#4682B4", "#B0C4DE", "#778899", "#2F4F4F", "#708090"],
        "Humayun Ahmed": ["#228B22", "#32CD32", "#90EE90", "#8FBC8F", "#6B8E23"]
    }
    
    colors = palettes.get(creator, ["#888888"] * 5)
    
    # Create palette image
    width, height = 400, 100
    palette_img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(palette_img)
    
    color_width = width // len(colors)
    for i, color in enumerate(colors):
        x1 = i * color_width
        x2 = (i + 1) * color_width
        draw.rectangle([x1, 0, x2, height], fill=color)
    
    return palette_img

def generate_space(creator: str, prompt: str = ""):
    """Main function - generates test content"""
    try:
        # Create test image and palette
        room_image = create_test_image(creator, prompt)
        palette_image = create_color_palette(creator)
        
        status = f"✅ Test generation successful for {creator}!"
        if prompt:
            status += f" Custom prompt: {prompt}"
            
        return room_image, palette_image, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Shared Space Generator - Test") as demo:
    
    gr.Markdown("""
    # 🏠 Shared Space Generator (Test Mode)
    
    **Testing basic functionality** - This version creates simple test images instead of using AI models.
    If this works, the real version with Stable Diffusion should work too.
    """)
    
    with gr.Row():
        with gr.Column():
            creator_dropdown = gr.Dropdown(
                choices=["Satyajit Ray", "Hayao Miyazaki", "Orhan Pamuk", "Humayun Ahmed"],
                value="Satyajit Ray", 
                label="Choose Creator"
            )
            
            custom_prompt = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Add elements like 'cozy lighting' or 'vintage furniture'",
                lines=2
            )
            
            generate_btn = gr.Button("🎨 Generate Test Space", variant="primary")
            
        with gr.Column():
            room_output = gr.Image(label="Generated Room (Test)", height=300)
            palette_output = gr.Image(label="Color Palette", height=150)
            status_output = gr.Textbox(label="Status", lines=2)
    
    # Events
    generate_btn.click(
        fn=generate_space,
        inputs=[creator_dropdown, custom_prompt],
        outputs=[room_output, palette_output, status_output]
    )
    
    creator_dropdown.change(
        fn=lambda creator: generate_space(creator, ""),
        inputs=[creator_dropdown],
        outputs=[room_output, palette_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
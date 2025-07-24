#!/usr/bin/env python3
"""
Startup script for Shared Space Generator
Pre-initializes the environment and downloads models to reduce first-run latency
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Setup the environment variables and cache directories"""
    print("🔧 Setting up environment...")
    
    # Set cache directories
    cache_dir = '/tmp/hf_cache'
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    print(f"✅ Cache directory created at {cache_dir}")

def pre_download_models():
    """Pre-download models to speed up first generation"""
    print("📥 Pre-downloading Stable Diffusion model...")
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Download but don't load into memory yet
        StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir='/tmp/hf_cache',
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        print("✅ Model downloaded successfully")
        
    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        print("Model will be downloaded on first use.")

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'diffusers', 'transformers', 
        'gradio', 'pillow', 'numpy', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Installing missing packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            *missing_packages
        ])

def main():
    """Main startup function"""
    print("🚀 Starting Shared Space Generator initialization...")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    check_dependencies()
    
    # Pre-download models (optional, may timeout in Spaces)
    if os.getenv('SPACE_ID'):  # Running in Hugging Face Space
        print("🔍 Detected Hugging Face Space environment")
        print("⏭️ Skipping model pre-download to avoid timeout")
    else:
        pre_download_models()
    
    print("=" * 50)
    print("✅ Initialization complete!")
    print("🏠 Ready to generate shared spaces!")

if __name__ == "__main__":
    main()
---
title: Shared Space Generator
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏠 Shared Space Generator

A Gradio Space that generates cozy interior images inspired by renowned creators and extracts their characteristic color palettes.

## ✨ Features

- **Creator-Inspired Generation**: Choose from Satyajit Ray, Hayao Miyazaki, Orhan Pamuk, or Humayun Ahmed
- **Cozy Interior Generation**: AI-powered room generation using Stable Diffusion
- **Color Palette Extraction**: Display characteristic colors associated with each creator
- **Custom Prompts**: Add your own elements to the generated spaces
- **Optimized Performance**: Built to work within Hugging Face Space limitations

## 🎨 Supported Creators

### Satyajit Ray
Warm earth tones reflecting Bengali cinema aesthetics with vintage and traditional elements.

### Hayao Miyazaki  
Natural, whimsical colors from Studio Ghibli with magical and organic atmospheres.

### Orhan Pamuk
Istanbul blues and literary melancholy with contemplative, book-filled spaces.

### Humayun Ahmed
Verdant greens of rural Bangladesh with natural materials and peaceful domestic elements.

## 🚀 How It Works

1. **Select a Creator**: Choose from the dropdown menu
2. **Add Custom Elements** (Optional): Specify additional elements like "with plants" or "modern furniture"
3. **Generate**: Click the generate button to create your space
4. **View Results**: See both the generated interior and the creator's color palette

## 🛠️ Technical Implementation

### Performance Optimizations
- **Cache Redirection**: All Hugging Face cache redirected to `/tmp` (writable in Spaces)
- **Memory Management**: Attention slicing and CPU offload for efficient GPU usage
- **Model Loading**: Optimized pipeline initialization with error handling
- **Reduced Inference Steps**: Balanced quality vs. speed (20 steps)

### Key Features
- Stub implementation for deprecated `cached_download` API
- Automatic device detection (CUDA/CPU)
- Memory-efficient model loading
- Error handling and graceful degradation

## 📁 File Structure

```
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## 🔧 Local Development

```bash
git clone <repository>
cd shared-space-generator
pip install -r requirements.txt
python app.py
```

## 📝 Notes

- First run may take longer due to model downloading
- GPU acceleration used when available
- Fallback to CPU processing for broader compatibility
- Color palettes are curated based on each creator's aesthetic style

## 🎯 Future Enhancements

- Real image analysis for color extraction
- More creator options
- Advanced prompt engineering
- Custom color palette uploads
- Style transfer capabilities

## 📄 License

MIT License - feel free to use and modify!

---

Built with ❤️ using Gradio, Diffusers, and Hugging Face Transformers.

import os
import torch
import clip
from PIL import Image
import colorgram
import json
from tqdm import tqdm

# Set up paths
image_root = "images"
output_root = "backend/aesthetic_profiles"
os.makedirs(output_root, exist_ok=True)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Helper: extract dominant color palette
def extract_palette(image_path, num_colors=5):
    colors = colorgram.extract(image_path, num_colors)
    return ['#%02x%02x%02x' % (c.rgb.r, c.rgb.g, c.rgb.b) for c in colors]

# Loop through each person folder
for figure_name in os.listdir(image_root):
    folder_path = os.path.join(image_root, figure_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n🔍 Processing: {figure_name}")
    clip_vectors = []
    palettes = []

    for fname in tqdm(os.listdir(folder_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(folder_path, fname)

        # Color palette
        palette = extract_palette(image_path)
        palettes.extend(palette)

        # CLIP embedding
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            clip_vectors.append(image_features[0].cpu().tolist())

    # Average embedding & top colors
    if len(clip_vectors) == 0:
        continue

    avg_clip_vector = torch.tensor(clip_vectors).mean(dim=0).tolist()
    top_colors = sorted(set(palettes), key=palettes.count, reverse=True)[:5]

    # Final output
    result = {
        "name": figure_name.replace('_', ' '),
        "color_palette": top_colors,
        "clip_vector": avg_clip_vector
    }

    # Save JSON
    out_path = os.path.join(output_root, f"{figure_name.lower()}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved: {out_path}")

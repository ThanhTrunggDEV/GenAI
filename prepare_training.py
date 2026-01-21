#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Training Dataset for Stable Diffusion Fine-tuning
Creates train/val/test splits and generates text captions
"""

import json
import shutil
import random
from pathlib import Path
from collections import Counter

def generate_caption(metadata):
    """Generate descriptive caption from metadata"""
    
    # Pattern info
    motifs = metadata["pattern_info"]["specific_motifs"]
    dominant = metadata["pattern_info"]["dominant_motif"]
    
    # Color info
    colors = metadata["color_info"]["colors"]
    color_scheme = metadata["color_info"]["color_scheme"]
    
    # Technique
    technique = metadata["technique"]["primary_technique"]
    
    # Cultural
    symbolism = metadata.get("cultural_meaning", {}).get("symbolism", "")
    
    # Build caption
    caption_parts = []
    
    # Start with dominant motif
    if dominant != "unknown":
        caption_parts.append(f"Hmong {dominant} pattern")
    else:
        caption_parts.append("Hmong traditional pattern")
    
    # Add technique
    if technique not in ["illustration", "unknown"]:
        caption_parts.append(f"{technique} technique")
    
    # Add colors
    if colors and colors[0] != "unknown":
        color_str = " and ".join(colors[:3])
        caption_parts.append(f"in {color_str}")
    
    # Add style
    caption_parts.append(f"{color_scheme} style")
    
    # Add symbolism if meaningful
    if symbolism and "|" not in symbolism:
        caption_parts.append(f"symbolizing {symbolism}")
    
    caption = ", ".join(caption_parts)
    return caption

def prepare_training_data():
    """Main training preparation pipeline"""
    
    # Paths
    img_dir = Path("dataset/augmented/images")
    meta_dir = Path("dataset/augmented/metadata")
    output_dir = Path("dataset/training")
    
    # Create output structure
    (output_dir / "train/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "test/images").mkdir(parents=True, exist_ok=True)
    
    print("\n📦 TRAINING DATA PREPARATION")
    print("="*70)
    
    # Get all images and metadata
    image_files = sorted(list(img_dir.glob("*.jpg")))
    
    if not image_files:
        print("❌ No augmented images found!")
        return
    
    print(f"Total images: {len(image_files)}")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(image_files)
    
    # Split: 70% train, 15% val, 15% test
    n_total = len(image_files)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train+n_val]
    test_images = image_files[n_train+n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
    
    # Process each split
    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }
    
    all_captions = []
    
    for split_name, images in splits.items():
        print(f"\nProcessing {split_name}...")
        split_dir = output_dir / split_name
        
        captions = []
        
        for img_path in images:
            # Copy image
            dest_img = split_dir / "images" / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Load metadata
            meta_path = meta_dir / f"{img_path.stem}.json"
            if not meta_path.exists():
                continue
            
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            #Generate caption
            caption = generate_caption(metadata)
            captions.append(f"{img_path.name}\t{caption}\n")
            all_captions.append(caption)
        
        # Save captions file
        with open(split_dir / "captions.txt", 'w', encoding='utf-8') as f:
            f.writelines(captions)
        
        print(f"  ✅ {len(images)} images + captions")
    
    # Generate dataset stats
    print("\n" + "="*70)
    print("✅ TRAINING DATA READY")
    print(f"\n📁 Output: {output_dir.absolute()}")
    
    # Sample captions
    print("\n📝 Sample Captions:")
    for caption in random.sample(all_captions, min(5, len(all_captions))):
        print(f"  • {caption}")
    
    # Create README
    readme_content = f"""# Hmong Pattern Training Dataset

## Dataset Statistics

- **Total Images**: {n_total}
- **Train**: {len(train_images)}  
- **Validation**: {len(val_images)}
- **Test**: {len(test_images)}

## Structure

```
training/
├── train/
│   ├── images/      # {len(train_images)} training images
│   └── captions.txt # Image-caption pairs
├── val/
│   ├── images/      # {len(val_images)} validation images
│   └── captions.txt
└── test/
    ├── images/      # {len(test_images)} test images
    └── captions.txt
```

## Caption Format

Each line: `<filename>TAB<caption>`

Example captions:
{chr(10).join(['- ' + c for c in random.sample(all_captions, min(3, len(all_captions)))])}

## Training

Use these for Stable Diffusion fine-tuning with LoRA/DreamBooth.

**Created**: 2026-01-21
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n📄 Created README.md")

if __name__ == "__main__":
    prepare_training_data()

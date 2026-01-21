#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation for Hmong Pattern Dataset
Generates variations while preserving cultural integrity
"""

import json
from PIL import Image
from pathlib import Path
import shutil

def augment_image(image_path, output_dir):
    """Create augmented versions of an image"""
    img = Image.open(image_path)
    image_id = image_path.stem
    
    augmentations = []
    
    # Original
    original_path = output_dir / f"{image_id}_original.jpg"
    img.save(original_path, quality=95)
    augmentations.append(("original", original_path))
    
    # Rotate 90°
    img_90 = img.rotate(90, expand=True)
    path_90 = output_dir / f"{image_id}_rot90.jpg"
    img_90.save(path_90, quality=95)
    augmentations.append(("rotate_90", path_90))
    
    # Rotate 180°
    img_180 = img.rotate(180, expand=True)
    path_180 = output_dir / f"{image_id}_rot180.jpg"
    img_180.save(path_180, quality=95)
    augmentations.append(("rotate_180", path_180))
    
    # Rotate 270°
    img_270 = img.rotate(270, expand=True)
    path_270 = output_dir / f"{image_id}_rot270.jpg"
    img_270.save(path_270, quality=95)
    augmentations.append(("rotate_270", path_270))
    
    # Horizontal flip
    img_hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
    path_hflip = output_dir / f"{image_id}_hflip.jpg"
    img_hflip.save(path_hflip, quality=95)
    augmentations.append(("flip_horizontal", path_hflip))
    
    return augmentations

def augment_metadata(original_json_path, augmentation_type, new_image_id):
    """Create metadata for augmented image"""
    with open(original_json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Update metadata
    metadata["image_id"] = new_image_id
    metadata["filename"] = f"{new_image_id}.jpg"
    metadata["notes"] = f"Augmented ({augmentation_type}) from {original_json_path.stem}"
    
    return metadata

def main():
    """Main augmentation pipeline"""
    
    # Paths
    input_dir = Path("dataset/to_annotate")
    metadata_dir = Path("dataset/metadata")
    output_img_dir = Path("dataset/augmented/images")
    output_meta_dir = Path("dataset/augmented/metadata")
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔄 DATA AUGMENTATION PIPELINE")
    print("="*70)
    
    # Get all images
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    
    if not image_files:
        print("❌ No images found!")
        return
    
    total_augmented = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Get original metadata
        metadata_path = metadata_dir / f"{img_path.stem}.json"
        if not metadata_path.exists():
            print(f"  ⚠️  Metadata not found, skipping")
            continue
        
        # Create augmentations
        augmentations = augment_image(img_path, output_img_dir)
        
        # Create metadata for each augmentation
        for aug_type, aug_path in augmentations:
            new_image_id = aug_path.stem
            new_metadata = augment_metadata(metadata_path, aug_type, new_image_id)
            
            # Save new metadata
            new_meta_path = output_meta_dir / f"{new_image_id}.json"
            with open(new_meta_path, 'w', encoding='utf-8') as f:
                json.dump(new_metadata, f, indent=2, ensure_ascii=False)
            
            total_augmented += 1
        
        print(f"  ✅ Created {len(augmentations)} variations")
    
    print("\n" + "="*70)
    print(f"✅ AUGMENTATION COMPLETE")
    print(f"   Original images: {len(image_files)}")
    print(f"   Total augmented: {total_augmented}")
    print(f"   Multiplication: {total_augmented / len(image_files):.1f}x")
    print(f"\n📁 Output: {output_img_dir.absolute()}")
    print(f"📁 Metadata: {output_meta_dir.absolute()}")

if __name__ == "__main__":
    main()

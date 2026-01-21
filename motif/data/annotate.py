#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple command-line annotation tool for Hmong textile patterns
Helps you create JSON metadata for each image
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Predefined options for consistency
PROVINCES = ["Lào Cai", "Yên Bái", "Hà Giang", "Unknown"]
SUBGROUPS = ["Black Hmong", "Flower Hmong", "White Hmong", "Green Hmong", "Unknown"]
MOTIF_TYPES = ["geometric", "floral", "animal", "abstract", "mixed"]
SPECIFIC_MOTIFS = ["snail", "zigzag", "spiral", "triangle", "diamond", "dragon", 
                   "bird", "flower", "butterfly", "chicken_foot", "pig_foot", 
                   "maze", "pumpkin_flower", "hemp_tool"]
COLORS = ["indigo", "black", "red", "yellow", "white", "blue", "green", "brown", "orange"]
TECHNIQUES = ["batik", "embroidery", "applique", "patchwork", "cross_stitch", "mixed"]
SYMMETRIES = ["rotational", "bilateral", "radial", "asymmetric", "none"]
REPETITIONS = ["grid", "linear", "scattered", "concentric", "none"]

def get_choice(prompt, options, allow_multiple=False):
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    if allow_multiple:
        print("(Enter numbers separated by commas, e.g., 1,3,5)")
        choice = input("Your choice: ").strip()
        if not choice:
            return []
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            return [options[i] for i in indices if 0 <= i < len(options)]
        except:
            return []
    else:
        choice = input("Your choice (number): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except:
            pass
        return options[-1]  # Default to last option (usually "Unknown")

def get_text(prompt, default=""):
    """Get text input from user"""
    response = input(f"\n{prompt} [{default}]: ").strip()
    return response if response else default

def annotate_image(image_path, output_dir):
    """Annotate a single image"""
    
    print("\n" + "="*70)
    print(f"🖼️  ANNOTATING: {image_path.name}")
    print("="*70)
    
    # Generate image_id from filename
    image_id = image_path.stem
    
    # Initialize metadata structure
    metadata = {
        "image_id": image_id,
        "filename": image_path.name,
        "source": get_text("Source (wikimedia/freepik/craftlink/field/other)", "unknown"),
        "source_url": get_text("Source URL (if available)", ""),
    }
    
    # Location
    print("\n--- LOCATION ---")
    province = get_choice("Province:", PROVINCES)
    metadata["location"] = {
        "province": province,
        "district": get_text("District/County", "Unknown"),
        "village": get_text("Village", "Unknown"),
        "region": "Northwest Vietnam"
    }
    
    # Ethnic info
    print("\n--- ETHNIC GROUP ---")
    subgroup = get_choice("Hmong Subgroup:", SUBGROUPS)
    metadata["ethnic_info"] = {
        "subgroup": subgroup,
        "local_name": get_text("Local name (Vietnamese)", "")
    }
    
    # Pattern
    print("\n--- PATTERN INFORMATION ---")
    motif_types = get_choice("Motif types:", MOTIF_TYPES, allow_multiple=True)
    specific_motifs = get_choice("Specific motifs:", SPECIFIC_MOTIFS, allow_multiple=True)
    dominant = specific_motifs[0] if specific_motifs else "unknown"
    
    metadata["pattern_info"] = {
        "motif_type": motif_types,
        "specific_motifs": specific_motifs,
        "dominant_motif": dominant
    }
    
    # Colors
    print("\n--- COLORS ---")
    colors = get_choice("Colors present:", COLORS, allow_multiple=True)
    dominant_color = get_choice("Dominant color:", colors if colors else COLORS)
    
    metadata["color_info"] = {
        "colors": colors,
        "dominant_color": dominant_color,
        "color_scheme": get_choice("Color scheme:", ["traditional", "modern", "festive", "mourning"])
    }
    
    # Cultural meaning
    print("\n--- CULTURAL MEANING ---")
    symbolism = get_text("Symbolism (mountains/fertility/protection/etc)", "")
    ritual = get_text("Ritual use (daily_wear/festival/funeral/wedding)", "daily_wear")
    
    metadata["cultural_meaning"] = {
        "symbolism": symbolism,
        "ritual_use": ritual,
        "significance": get_choice("Significance:", ["High", "Medium", "Low"])
    }
    
    # Technique
    print("\n--- TECHNIQUE ---")
    technique = get_choice("Primary technique:", TECHNIQUES)
    metadata["technique"] = {
        "primary_technique": technique,
        "tools_used": get_text("Tools used (comma-separated)", "").split(','),
        "material": get_text("Material (linen/cotton/hemp)", "unknown")
    }
    
    # Visual structure
    print("\n--- VISUAL STRUCTURE ---")
    metadata["visual_structure"] = {
        "symmetry": get_choice("Symmetry:", SYMMETRIES),
        "repetition": get_choice("Repetition pattern:", REPETITIONS),
        "complexity": get_choice("Complexity:", ["high", "medium", "low"])
    }
    
    # Quality
    print("\n--- QUALITY ---")
    metadata["quality"] = {
        "resolution": f"{image_path.stat().st_size // 1024}KB",
        "clarity": get_choice("Clarity:", ["high", "medium", "low"]),
        "completeness": get_choice("Completeness:", ["full pattern", "partial", "fragment"]),
        "condition": get_choice("Condition:", ["new", "traditional", "worn", "restored"])
    }
    
    # Date info
    metadata["date_info"] = {
        "photo_date": get_text("Photo date (YYYY-MM-DD)", "unknown"),
        "estimated_creation": get_choice("Creation era:", ["traditional", "modern", "contemporary"]),
        "annotation_date": datetime.now().strftime("%Y-%m-%d"),
        "annotator": get_text("Your name", "researcher")
    }
    
    # Notes
    metadata["notes"] = get_text("Additional notes", "")
    
    # Save JSON
    output_path = output_dir / f"{image_id}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved: {output_path}")
    return metadata

def main():
    """Main annotation workflow"""
    print("\n🏷️  HMONG PATTERN ANNOTATION TOOL")
    print("="*70)
    
    # Setup directories
    dataset_dir = Path("dataset")
    images_dir = dataset_dir / "processed"
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for images
    if not images_dir.exists():
        print(f"❌ Image directory not found: {images_dir}")
        print(f"Please create it and add images to annotate.")
        return
    
    # Get list of images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    # Get already annotated
    annotated = {f.stem for f in metadata_dir.glob("*.json")}
    remaining = [f for f in image_files if f.stem not in annotated]
    
    print(f"\n📊 Status:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Already annotated: {len(annotated)}")
    print(f"   Remaining: {len(remaining)}")
    
    if not remaining:
        print("\n✅ All images have been annotated!")
        return
    
    # Annotate images
    for i, image_path in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}]")
        try:
            annotate_image(image_path, metadata_dir)
        except KeyboardInterrupt:
            print("\n\n⏸️  Annotation paused. Progress saved.")
            break
        except Exception as e:
            print(f"❌ Error annotating {image_path}: {e}")
            continue
        
        # Ask to continue
        if i < len(remaining):
            cont = input("\nContinue to next image? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    print("\n✅ Annotation session complete!")
    print(f"📁 Metadata saved in: {metadata_dir}")

if __name__ == "__main__":
    main()

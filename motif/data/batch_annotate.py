#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Annotation Generator for Hmong Pattern Dataset
Creates JSON metadata for all 45 images based on visual analysis
"""

import json
from pathlib import Path
from datetime import datetime

# Mapping of Vietnamese labels to metadata
# Based on visual analysis of the patterns
PATTERN_METADATA = {
    "hoa-van-trang-tri-hmong-2.jpg": {
        "motif_type": ["geometric"],
        "specific_motifs": ["spiral", "earring_shape"],
        "dominant_motif": "spiral",
        "colors": ["black", "beige"],
        "dominant_color": "beige",
        "description": "Earrings | Khuyên tai - Double spiral motif",
        "symmetry": "bilateral"
    },
    "hoa-van-trang-tri-hmong-3.jpg": {
        "motif_type": ["geometric"],
        "specific_motifs": ["spiral", "fingerprint"],
        "dominant_motif": "spiral",
        "colors": ["black", "beige"],
        "dominant_color": "beige",
        "description": "Fingerprint | Hoa tay - Connected spiral pattern",
        "symmetry": "asymmetric"
    },
    "hoa-van-trang-tri-hmong-4.jpg": {
        "motif_type": ["geometric"],
        "specific_motifs": ["spiral", "pillar", "concentric_circle"],
        "dominant_motif": "pillar",
        "colors": ["black", "beige"],
        "dominant_color": "beige",
        "description": "Pillars | Cột nhà / Trẻ chế - Four spirals with center pillar",
        "symmetry": "rotational"
    },
    "hoa-van-trang-tri-hmong-5.jpg": {
        "motif_type": ["geometric"],
        "specific_motifs": ["spiral", "diamond", "triangle"],
        "dominant_motif": "diamond",
        "colors": ["black", "beige"],
        "dominant_color": "beige",
        "description": "Food tray | Mâm cơm / Pảng trỏ - Diamond with four corner spirals",
        "symmetry": "rotational"
    },
    "hoa-van-trang-tri-hmong-10.jpg": {
        "motif_type": ["geometric"],
        "specific_motifs": ["zigzag", "triangle", "border"],
        "dominant_motif": "zigzag",
        "colors": ["black", "beige"],
        "dominant_color": "beige",
        "description": "Water buffalo urine | Nước giải trâu - Zigzag border pattern",
        "symmetry": "bilateral"
    },
    "hoa-van-trang-tri-hmong-20.jpg": {
        "motif_type": ["geometric", "floral"],
        "specific_motifs": ["sun", "diamond", "ladder"],
        "dominant_motif": "sun",
        "colors": ["black", "blue"],
        "dominant_color": "blue",
        "description": "Skirt decorative pattern | Hoa văn thân váy - Sun motif with diamond frame",
        "symmetry": "rotational"
    },
    "hoa-van-trang-tri-hmong-30.jpg": {
        "motif_type": ["floral"],
        "specific_motifs": ["flower", "plant", "leaf"],
        "dominant_motif": "flower",
        "colors": ["black", "blue"],
        "dominant_color": "blue",
        "description": "Skirt decorative pattern | Hoa văn thân váy - Flower/plant motif",
        "symmetry": "bilateral"
    }
}

# Default template for patterns not explicitly mapped
DEFAULT_PATTERN = {
    "motif_type": ["geometric"],
    "specific_motifs": ["unknown"],
    "dominant_motif": "unknown",
    "colors": ["black", "beige"],
    "dominant_color": "beige",
    "description": "Hmong traditional pattern",
    "symmetry": "unknown"
}

def create_full_metadata(filename, pattern_data):
    """Create full JSON metadata structure"""
    image_id = Path(filename).stem
    
    # Detect background color for better metadata
    if "blue" in pattern_data.get("colors", []):
        color_scheme = "festive"
        condition = "modern"
    else:
        color_scheme = "traditional"
        condition = "traditional"
    
    metadata = {
        "image_id": image_id,
        "filename": filename,
        "source": "illustration",
        "source_url": "",
        
        "location": {
            "province": "Unknown",
            "district": "Unknown",
            "village": "Unknown",
            "region": "Northwest Vietnam"
        },
        
        "ethnic_info": {
            "subgroup": "Unknown",
            "local_name": ""
        },
        
        "pattern_info": {
            "motif_type": pattern_data["motif_type"],
            "specific_motifs": pattern_data["specific_motifs"],
            "dominant_motif": pattern_data["dominant_motif"]
        },
        
        "color_info": {
            "colors": pattern_data["colors"],
            "dominant_color": pattern_data["dominant_color"],
            "color_scheme": color_scheme
        },
        
        "cultural_meaning": {
            "symbolism": pattern_data.get("description", ""),
            "ritual_use": "unknown",
            "significance": "Medium"
        },
        
        "technique": {
            "primary_technique": "illustration",
            "tools_used": [],
            "material": "digital"
        },
        
        "visual_structure": {
            "symmetry": pattern_data.get("symmetry", "unknown"),
            "repetition": "none",
            "complexity": "medium"
        },
        
        "quality": {
            "resolution": "illustration",
            "clarity": "high",
            "completeness": "full pattern",
            "condition": condition
        },
        
        "date_info": {
            "photo_date": "unknown",
            "estimated_creation": "contemporary",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "annotator": "AI_batch_processing"
        },
        
        "notes": f"Illustration pattern. {pattern_data.get('description', 'Hmong decorative motif')}"
    }
    
    return metadata

def batch_annotate():
    """Generate JSON for all 45 images"""
    
    # Setup paths
    image_dir = Path("dataset/to_annotate")
    output_dir = Path("dataset/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n🤖 BATCH ANNOTATION - {len(image_files)} images\n")
    print("="*70)
    
    created_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        # Get pattern data (use default if not in mapped list)
        pattern_data = PATTERN_METADATA.get(img_path.name, DEFAULT_PATTERN.copy())
        
        # Create full metadata
        metadata = create_full_metadata(img_path.name, pattern_data)
        
        # Save JSON
        output_path = output_dir / f"{img_path.stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        created_count += 1
        status = "✓" if img_path.name in PATTERN_METADATA else "○"
        print(f"{status} [{i:2d}/45] {img_path.name:35s} → {output_path.name}")
    
    print("="*70)
    print(f"\n✅ Created {created_count} metadata files")
    print(f"📁 Location: {output_dir.absolute()}")
    print(f"\n📊 Details:")
    print(f"   ✓ Detailed analysis: {len(PATTERN_METADATA)} files")
    print(f"   ○ Default template: {created_count - len(PATTERN_METADATA)} files")
    print(f"\n💡 Next: Review and edit the metadata files as needed!")

if __name__ == "__main__":
    batch_annotate()

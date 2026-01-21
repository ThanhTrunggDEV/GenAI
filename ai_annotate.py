#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Assisted Annotation Tool for Hmong Patterns
Uses vision AI to automatically suggest metadata
"""

import json
import base64
from pathlib import Path
from datetime import datetime
import sys

def encode_image_to_base64(image_path):
    """Convert image to base64 for AI processing"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_annotation_prompt():
    """Create detailed prompt for AI vision analysis"""
    return """Analyze this Hmong textile pattern image and provide metadata in JSON format.

Focus on these aspects:

1. **Motif Types**: Identify geometric (zigzag, spiral, triangle, diamond, maze), 
   floral (flower, butterfly), animal (dragon, bird, snail), or mixed patterns

2. **Specific Motifs**: Look for traditional Hmong motifs:
   - Geometric: snail (ốc sên), zigzag (núi non), spiral, maze
   - Organic: dragon, bird, flower, butterfly
   - Technical: chicken_foot, pig_foot, hemp_tool, pumpkin_flower

3. **Colors**: List all visible colors, identify dominant color
   Common: indigo (chàm - blue-black), red, black, yellow, white, blue, green

4. **Technique**: Identify the craft technique:
   - batik (wax-resist dyeing, fine lines on indigo)
   - embroidery (stitched patterns)
   - applique (fabric patches sewn on)
   - patchwork (multiple fabric pieces)

5. **Visual Structure**:
   - Symmetry: rotational, bilateral, radial, or asymmetric
   - Repetition: grid, linear, scattered, concentric, or none
   - Complexity: high (intricate), medium, or low (simple)

6. **Ethnic Subgroup** (if identifiable):
   - Black Hmong (Hmong Đen): dominant indigo/black, batik technique
   - Flower Hmong (Hmong Hoa): colorful, mixed techniques
   - White Hmong (Hmong Trắng): white/red/black applique, snail motif

7. **Cultural Meaning** (if recognizable):
   - Symbolism: mountains, fertility, protection, prosperity, ancestors
   - Likely ritual use: daily_wear, festival, funeral, wedding

Return ONLY valid JSON in this exact format:
{
  "motif_type": ["geometric"],
  "specific_motifs": ["snail", "spiral"],
  "dominant_motif": "snail",
  "colors": ["indigo", "white", "red"],
  "dominant_color": "indigo",
  "color_scheme": "traditional",
  "technique": "batik",
  "symmetry": "rotational",
  "repetition": "grid",
  "complexity": "high",
  "ethnic_subgroup": "Black Hmong",
  "symbolism": "protection, ancestors",
  "ritual_use": "daily_wear",
  "confidence": "high"
}

Be specific and accurate. If uncertain about any field, use "unknown" or indicate lower confidence."""

def save_annotation(image_path, metadata, output_dir):
    """Save annotation JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_id = image_path.stem
    
    # Create full metadata structure
    full_metadata = {
        "image_id": image_id,
        "filename": image_path.name,
        "source": "user_upload",
        "source_url": "",
        
        "location": {
            "province": metadata.get("province", "Unknown"),
            "district": "Unknown",
            "village": "Unknown",
            "region": "Northwest Vietnam"
        },
        
        "ethnic_info": {
            "subgroup": metadata.get("ethnic_subgroup", "Unknown"),
            "local_name": ""
        },
        
        "pattern_info": {
            "motif_type": metadata.get("motif_type", []),
            "specific_motifs": metadata.get("specific_motifs", []),
            "dominant_motif": metadata.get("dominant_motif", "unknown")
        },
        
        "color_info": {
            "colors": metadata.get("colors", []),
            "dominant_color": metadata.get("dominant_color", "unknown"),
            "color_scheme": metadata.get("color_scheme", "traditional")
        },
        
        "cultural_meaning": {
            "symbolism": metadata.get("symbolism", ""),
            "ritual_use": metadata.get("ritual_use", "unknown"),
            "significance": "Medium"
        },
        
        "technique": {
            "primary_technique": metadata.get("technique", "unknown"),
            "tools_used": [],
            "material": "unknown"
        },
        
        "visual_structure": {
            "symmetry": metadata.get("symmetry", "unknown"),
            "repetition": metadata.get("repetition", "unknown"),
            "complexity": metadata.get("complexity", "medium")
        },
        
        "quality": {
            "resolution": f"{image_path.stat().st_size // 1024}KB",
            "clarity": "unknown",
            "completeness": "full pattern",
            "condition": "unknown"
        },
        
        "date_info": {
            "photo_date": "unknown",
            "estimated_creation": "unknown",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "annotator": "AI_assisted"
        },
        
        "ai_confidence": metadata.get("confidence", "medium"),
        "notes": "Auto-annotated by AI. Please review and correct if needed."
    }
    
    # Save JSON
    output_path = output_dir / f"{image_id}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    
    return output_path

def print_usage():
    """Print usage instructions"""
    print("""
🤖 AI-ASSISTED HMONG PATTERN ANNOTATION TOOL

Usage:
    python ai_annotate.py <image_file_or_folder>

This tool uses AI vision to automatically analyze Hmong patterns and 
suggest metadata. You can then review and edit the generated JSON files.

Examples:
    python ai_annotate.py pattern.jpg
    python ai_annotate.py dataset/processed/

After running, check the 'dataset/metadata/' folder for generated JSON files.
Review and edit them as needed!
    """)

def main():
    """Main function - placeholder for actual AI integration"""
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        return
    
    # Get list of images
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {input_path}")
        return
    
    print(f"\n🔍 Found {len(image_files)} images to annotate\n")
    print("⚠️  NOTE: This script requires manual integration with AI vision API")
    print("    For now, it will create template JSON files.\n")
    print("📸 Upload images to the chat, and I'll analyze them directly!\n")
    
    # Create output directory
    output_dir = Path("dataset/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # For now, create a template annotation
        template_metadata = {
            "motif_type": ["unknown"],
            "specific_motifs": [],
            "dominant_motif": "unknown",
            "colors": [],
            "dominant_color": "unknown",
            "color_scheme": "traditional",
            "technique": "unknown",
            "symmetry": "unknown",
            "repetition": "unknown",
            "complexity": "medium",
            "ethnic_subgroup": "Unknown",
            "symbolism": "",
            "ritual_use": "unknown",
            "confidence": "low"
        }
        
        output_path = save_annotation(img_path, template_metadata, output_dir)
        print(f"   ✅ Created: {output_path}")
    
    print(f"\n✅ Created {len(image_files)} template annotations")
    print(f"📁 Location: {output_dir}")
    print("\n💡 NEXT STEP: Upload your images in the chat, and I'll analyze them!")
    print("   I can see images and provide detailed annotations automatically.\n")

if __name__ == "__main__":
    main()

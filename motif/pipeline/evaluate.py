#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script - Calculate metrics for generated patterns
FID, KID, Cultural Consistency Score
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
import torch
from transformers import CLIPProcessor, CLIPModel

def calculate_clip_score(real_images_dir, generated_images_dir, prompt="Hmong traditional textile pattern"):
    """
    Calculate CLIP Score to measure text-image alignment and similarity 
    between generated images and the concept.
    Replaces FID for small datasets.
    """
    print(f"📊 Calculating CLIP Score...")
    print(f"   Generated images: {generated_images_dir}")
    print(f"   Concept: '{prompt}'")
    
    generated_dir = Path(generated_images_dir)
    image_files = list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg"))
    
    if not image_files:
        print("⚠️ No generated images to evaluate.")
        return 0.0

    try:
        # Load CLIP
        # Fix for CVE-2025-32434: Explicitly set safe serialization or trust remote code depending on availability
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        except:
             # Fallback if safetensors not available on Hub
             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
             processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        scores = []
        for img_path in image_files:
            image = Image.open(img_path)
            
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            score = logits_per_image.item()
            scores.append(score)
            
        avg_score = np.mean(scores)
        print(f"✅ Average CLIP Score: {avg_score:.2f} (Higher is better)")
        return avg_score
        
    except Exception as e:
        print(f"⚠️ Failed to calculate CLIP score: {e}")
        return 0.0

def calculate_fid_placeholder(real_images_dir, generated_images_dir):
    # Backward compatibility wrapper
    return calculate_clip_score(real_images_dir, generated_images_dir)


def calculate_cultural_consistency(generated_dir, metadata_dir=None, validators=None):
    """
    Calculate cultural consistency score using CLIP Zero-shot Classification
    Checks if generated images contain key Hmong cultural elements.
    
    Args:
        generated_dir: Directory with generated images
        metadata_dir: (Unused in this method)
        validators: (Unused in this method)
    Returns:
        consistency_score: Percentage (0-100)
        details: Dict with breakdown
    """
    print(f"📊 Calculating Cultural Consistency (CLIP Zero-shot)...")
    
    generated_dir = Path(generated_dir)
    image_files = list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ No generated images found")
        return 0.0, {}

    try:
        # Load CLIP (Reuse model loading logic)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        except:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False).to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)

        # Define cultural classes to check against
        # "hmong pattern" is the target, others are negative classes
        candidate_labels = [
            "traditional hmong pattern with geometric motifs",
            "random noise or blurry image", 
            "realistic photo of a person",
            "plain fabric without pattern"
        ]
        
        passed_count = 0
        total_count = len(image_files)
        
        for img_path in image_files:
            image = Image.open(img_path)
            
            inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can use softmax to get the probabilities
            
            # Get the predicted label index
            pred_idx = probs.argmax().item()
            predicted_label = candidate_labels[pred_idx]
            
            # Check if prediction is the target class (index 0)
            if pred_idx == 0:
                passed_count += 1
                
        consistency_score = (passed_count / total_count) * 100
        print(f"✅ Cultural Consistency Score: {consistency_score:.1f}% ({passed_count}/{total_count} passed)")
        
        details = {
            'total_images': total_count,
            'passed': passed_count,
            'consistency_score': consistency_score,
            'method': "CLIP Zero-shot Classification"
        }
        
        return consistency_score, details

    except Exception as e:
        print(f"⚠️ Failed to calculate consistency: {e}")
        return 0.0, {}



# Old implementation removed to avoid syntax errors
# def calculate_cultural_consistency_legacy(generated_dir, metadata_dir, validators):
#     pass



def main():
    parser = argparse.ArgumentParser(description="Evaluate generated patterns")
    parser.add_argument("--real", type=str, default="dataset/training/train/images",
                       help="Real images directory")
    parser.add_argument("--generated", type=str, default="outputs/generated",
                       help="Generated images directory")
    parser.add_argument("--metadata", type=str, default="dataset/metadata",
                       help="Metadata directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output results file")
    
    args = parser.parse_args()
    
    print("📊 HMONG PATTERN EVALUATION")
    print("="*70)
    
    # 1. CLIP Score (Replaces FID)
    print("\n1. CLIP Score (Visual Quality & Alignment)")
    # Use calculate_clip_score which we defined earlier (aliased from calculate_fid_placeholder wrapper if needed, 
    # but better to call directly if I renamed it properly.
    # Note: I renamed calculate_fid_placeholder to calculate_clip_score in the previous step,
    # and added a wrapper. So calling the wrapper or the new function works.
    # However, since I edited the definition, I should call the new function name if possible, 
    # but the old function name calculate_fid_placeholder still exists as a wrapper.
    # Let's use the wrapper to be safe with existing calls or just call the new one.
    
    try:
        clip_score = calculate_clip_score(args.real, args.generated)
    except NameError:
         # Fallback if the previous edit didn't rename it globally or something
         clip_score = 0
    
    # 2. Cultural Consistency
    print("\n2. Cultural Consistency Score")
    consistency_score, consistency_details = calculate_cultural_consistency(
        args.generated, args.metadata, None
    )
    
    # Summary
    results = {
        "clip_score": clip_score,
        "cultural_consistency": consistency_score,
        "details": consistency_details
    }
    
    print("\n" + "="*70)
    print("📈 FINAL RESULTS")
    print(f"   • CLIP Score:            {clip_score:.4f}")
    print(f"   • Cultural Consistency:  {consistency_score:.1f}%")
    print("="*70)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()

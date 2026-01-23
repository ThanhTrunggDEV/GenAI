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
import cv2
import shutil
import os

# Try importing research-grade metrics
try:
    import lpips
    from torch_fidelity import calculate_metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("⚠️ Research metrics libraries (lpips, torch-fidelity) not found.")
    print("   Install for NCKH report: pip install lpips torch-fidelity")

def preprocess_for_fid(source_dir, target_dir, size=(299, 299)):
    """
    Resize images to uniform size for FID calculation to avoid stack errors 
    with mixed aspect ratios.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    
    files = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
    # print(f"   Preprocessing {len(files)} images for FID...")
    
    for file_path in files:
        try:
            with Image.open(file_path) as img:
                # Resize to standard Inception size (299x299)
                img_resized = img.resize(size, Image.Resampling.LANCZOS).convert('RGB')
                img_resized.save(target_path / file_path.name)
        except Exception as e:
            pass

def calculate_fid_lpips(real_dir, generated_dir):
    """
    Calculate FID (Fréchet Inception Distance) and LPIPS
    Standard metrics for scientific papers (NCKH)
    """
    if not HAS_METRICS:
        return 0.0, 0.0

    print(f"📊 Calculating Research Metrics (FID & LPIPS)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Calculate LPIPS
    lpips_score = 0.0
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        
        gen_files = list(Path(generated_dir).glob("*.png"))
        if len(gen_files) < 2: 
            print("⚠️ Not enough images for LPIPS")
        else:
            dists = []
            # Compare random pairs to better estimate diversity
            import random
            random.shuffle(gen_files)
            pairs_count = min(len(gen_files)-1, 100) # Limit pairs for speed
            
            for i in range(pairs_count):
                img0 = lpips.im2tensor(lpips.load_image(str(gen_files[i]))).to(device)
                img1 = lpips.im2tensor(lpips.load_image(str(gen_files[i+1]))).to(device)
                dist = loss_fn_alex(img0, img1)
                dists.append(dist.item())
                
            lpips_score = np.mean(dists) if dists else 0.0
            print(f"✅ LPIPS Score: {lpips_score:.4f} (Higher is better diversity)")
    except Exception as e:
        print(f"⚠️ LPIPS Error: {e}")
        lpips_score = 0.0

    # 2. Calculate FID using torch-fidelity
    # Note: Requires at least ~100 images for valid FID, usually 10k+
    fid_score = 0.0
    try:
        # Preprocess images to temp dirs to ensure consistent size
        temp_real = Path("temp_fid_real")
        temp_gen = Path("temp_fid_gen")
        
        preprocess_for_fid(real_dir, temp_real)
        preprocess_for_fid(generated_dir, temp_gen)
        
        # Check if we have images
        if not list(temp_real.glob("*")) or not list(temp_gen.glob("*")):
            print("⚠️ Skipping FID: No valid images found after preprocessing")
        else:
            metrics = calculate_metrics(
                input1=str(temp_real), 
                input2=str(temp_gen), 
                cuda=torch.cuda.is_available(), 
                isc=False, 
                fid=True, 
                kid=False, 
                verbose=False
            )
            fid_score = metrics['frechet_inception_distance']
            print(f"✅ FID Score: {fid_score:.4f} (Lower is better)")
            
        # Cleanup
        if temp_real.exists(): shutil.rmtree(temp_real)
        if temp_gen.exists(): shutil.rmtree(temp_gen)
            
    except Exception as e:
        print(f"⚠️ FID Error: {e}")
        # Cleanup on error
        if Path("temp_fid_real").exists(): shutil.rmtree("temp_fid_real", ignore_errors=True)
        if Path("temp_fid_gen").exists(): shutil.rmtree("temp_fid_gen", ignore_errors=True)
        fid_score = 0.0
        
    return fid_score, lpips_score

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

def calculate_diversity_score(generated_dir):
    """
    Calculate diversity among generated images using pixel-wise L2 distance.
    (Simple version, assumes images are aligned or similar structure).
    Higher score = More diverse images.
    """
    print(f"📊 Calculating Diversity Score...")
    import itertools
    
    generated_dir = Path(generated_dir)
    files = list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg"))
    
    if len(files) < 2:
        return 0.0
    
    images = []
    for f in files:
        img = Image.open(f).resize((64, 64)).convert('RGB') # Resize small for speed
        images.append(np.array(img).flatten() / 255.0)
    
    distances = []
    for img1, img2 in itertools.combinations(images, 2):
        dist = np.linalg.norm(img1 - img2)
        distances.append(dist)
        
    diversity = np.mean(distances)
    print(f"✅ Diversity Score: {diversity:.4f} (Higher is more diverse)")
    return diversity



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
    
    # 1. CLIP Score
    clip_score = calculate_clip_score(args.real, args.generated)
    
    # 2. Research Metrics (FID & LPIPS) - NEW
    fid_score, lpips_score = calculate_fid_lpips(args.real, args.generated)
    
    # 3. Cultural Consistency
    print("\n2. Cultural Consistency Score")
    consistency_score, consistency_details = calculate_cultural_consistency(
        args.generated, args.metadata, None
    )

    # 4. Diversity Score (Legacy) - LPIPS replaces this for Paper
    # diversity_score = calculate_diversity_score(args.generated)
    diversity_score = lpips_score if lpips_score > 0 else calculate_diversity_score(args.generated)
    
    # Summary
    results = {
        "clip_score": clip_score,
        "fid_score": fid_score,         # Add FID
        "lpips_score": lpips_score,     # Add LPIPS
        "cultural_consistency": consistency_score,
        "diversity_score": diversity_score,
        "details": consistency_details
    }
    
    print("\n" + "="*70)
    print("📈 FINAL RESULTS (FOR NCKH)")
    print(f"   • FID Score (Quality):   {fid_score:.4f} (Lower = Better)")
    print(f"   • LPIPS (Diversity):     {lpips_score:.4f} (Higher = Better)")
    print(f"   • CLIP Score (Align):    {clip_score:.4f} (Higher = Better)")
    print(f"   • Cultural Acc:          {consistency_score:.1f}%")
    print("="*70)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()

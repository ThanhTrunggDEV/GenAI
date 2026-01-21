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

def calculate_fid_placeholder(real_images_dir, generated_images_dir):
    """
    Calculate FID score (Placeholder)
    
    Actual implementation requires pytorch-fid:
    pip install pytorch-fid
    
    Returns:
        fid_score: Lower is better (< 50 is good)
    """
    print("📊 Calculating FID Score...")
    print(f"   Real images: {real_images_dir}")
    print(f"   Generated images: {generated_images_dir}")
    
    print("\n⚠️  Placeholder: Use pytorch-fid library")
    print("   pip install pytorch-fid")
    print("   python -m pytorch_fid path/to/real path/to/generated")
    
    # Placeholder return
    return None


def calculate_cultural_consistency(generated_dir, metadata_dir, validators):
    """
    Calculate cultural consistency score
    % of generated patterns that pass validation
    
    Args:
        generated_dir: Directory with generated images
        metadata_dir: Directory with target metadata
        validators: ValidationPipeline instance
    Returns:
        consistency_score: Percentage (0-100)
        details: Dict with breakdown
    """
    from motif.validators import ValidationPipeline
    
    generated_dir = Path(generated_dir)
    metadata_dir = Path(metadata_dir)
    
    if not generated_dir.exists():
        print(f"❌ Generated directory not found: {generated_dir}")
        return 0.0, {}
    
    image_files = list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print("❌ No generated images found")
        return 0.0, {}
    
    pipeline = ValidationPipeline()
    
    passed = 0
    failed = 0
    results_breakdown = {
        'motif_pass': 0,
        'symbolic_pass': 0,
        'structure_pass': 0
    }
    
    for img_path in image_files:
        # Load image
        image = Image.open(img_path)
        
        # Try to find corresponding metadata
        meta_path = metadata_dir / f"{img_path.stem}.json"
        if not meta_path.exists():
            # Use default metadata
            metadata = {
                "pattern_info": {"specific_motifs": []},
                "cultural_meaning": {"ritual_use": "daily_wear"},
                "visual_structure": {"symmetry": "unknown"}
            }
        else:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Validate
        is_valid, results = pipeline.validate_pattern(image, metadata)
        
        if is_valid:
            passed += 1
        else:
            failed += 1
        
        # Track breakdown
        if results['motif']['valid']:
            results_breakdown['motif_pass'] += 1
        if results['symbolic']['valid']:
            results_breakdown['symbolic_pass'] += 1
        if results['structure']['valid']:
            results_breakdown['structure_pass'] += 1
    
    total = passed + failed
    consistency_score = (passed / total * 100) if total > 0 else 0
    
    details = {
        'total_images': total,
        'passed': passed,
        'failed': failed,
        'consistency_score': consistency_score,
        'motif_pass_rate': results_breakdown['motif_pass'] / total * 100 if total > 0 else 0,
        'symbolic_pass_rate': results_breakdown['symbolic_pass'] / total * 100 if total > 0 else 0,
        'structure_pass_rate': results_breakdown['structure_pass'] / total * 100 if total > 0 else 0
    }
    
    return consistency_score, details


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated patterns")
    parser.add_argument("--real", type=str, default="dataset/training/test/images",
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
    
    # 1. FID Score
    print("\n1. FID Score (Fréchet Inception Distance)")
    fid_score = calculate_fid_placeholder(args.real, args.generated)
    
    # 2. Cultural Consistency
    print("\n2. Cultural Consistency Score")
    consistency_score, consistency_details = calculate_cultural_consistency(
        args.generated, args.metadata, None
    )
    
    # Summary
    results = {
        'fid_score': fid_score,
        'cultural_consistency': consistency_score,
        'details': consistency_details
    }
    
    print("\n" + "="*70)
    print("📋 EVALUATION SUMMARY")
    print("="*70)
    if fid_score:
        print(f"FID Score: {fid_score:.2f} (lower is better, <50 is good)")
    print(f"Cultural Consistency: {consistency_score:.1f}%")
    print(f"  - Motif Pass Rate: {consistency_details.get('motif_pass_rate', 0):.1f}%")
    print(f"  - Symbolic Pass Rate: {consistency_details.get('symbolic_pass_rate', 0):.1f}%")
    print(f"  - Structure Pass Rate: {consistency_details.get('structure_pass_rate', 0):.1f}%")
    print(f"\nTotal Images Evaluated: {consistency_details.get('total_images', 0)}")
    print(f"Passed All Checks: {consistency_details.get('passed', 0)}")
    print(f"Failed Some Checks: {consistency_details.get('failed', 0)}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()

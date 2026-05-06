#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Pipeline Automation
Runs entire workflow from encoding to generation (when model is trained)
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main():
    print("🎨 HMONG PATTERN GENERATION - FULL PIPELINE")
    print("="*70)
    
    # Check if dataset exists
    if not Path("dataset/training/train/images").exists():
        print("\n⚠️ Dataset not found! Running data augmentation and preparation automatically...")
        if not run_command([sys.executable, "-m", "motif.data.augment"], "Data Augmentation (10x)"):
            return
        if not run_command([sys.executable, "-m", "motif.data.prepare"], "Dataset Split (Train/Val/Test)"):
            return
    
    print("\n📋 Pipeline Steps:")
    print("0. Prepare and Augment Dataset")
    print("1. Extract visual embeddings")
    print("2. Extract cultural embeddings")
    print("3. Combine embeddings")
    print("4. Train model (GPU required)")
    print("5. Generate samples")
    print("6. Evaluate results")
    
    # Stage 1: Encoding
    print("\n" + "="*70)
    print("STAGE 1: CULTURAL ENCODING")
    print("="*70)
    
    if not run_command(
        [sys.executable, "-m", "motif.models.visual_encoder"],
        "Extract Visual Features"
    ):
        return
    
    if not run_command(
        [sys.executable, "-m", "motif.models.cultural_encoder"],
        "Extract Cultural Features"
    ):
        return
    
    if not run_command(
        [sys.executable, "-m", "motif.models.combine_embeddings"],
        "Combine Embeddings"
    ):
        return
    
    # Stage 2: Training (GPU required)
    print("\n" + "="*70)
    print("STAGE 2: MODEL TRAINING")
    print("="*70)
    
    checkpoint_dir = Path("outputs/hmong-pattern-lora")
    
    if checkpoint_dir.exists() and (checkpoint_dir / "adapter_model.bin").exists():
        print("✅ Trained model found. Skipping training.")
    else:
        print("⚠️ Model not found. Starting training...")
        # Automatically train if not found
        if not run_command(
            [sys.executable, "train_diffusion.py", 
             "--train_batch_size", "4", 
             "--gradient_accumulation_steps", "4",
             "--num_train_epochs", "10"],
            "Train Diffusion Model"
        ):
            print("\n❌ Training failed. Please check your GPU configuration.")
            return

    # Check again if model exists
    if not checkpoint_dir.exists():
        print("\n❌ No trained model found")
        print(f"Expected checkpoint at: {checkpoint_dir}")
        print("\nPipeline completed up to training stage.")
        print("Continue on GPU platform to complete.")
        return
    
    # Stage 3: Generation
    print("\n" + "="*70)
    print("STAGE 3: PATTERN GENERATION")
    print("="*70)
    
    if not run_command(
        [sys.executable, "-m", "motif.pipeline.generate",
         "--prompt", "Hmong spiral pattern in indigo",
         "--num_samples", "10"],
        "Generate Sample Patterns"
    ):
        return
    
    # Stage 4: Evaluation
    print("\n" + "="*70)
    print("STAGE 4: EVALUATION")
    print("="*70)
    
    if not run_command(
        [sys.executable, "-m", "motif.pipeline.evaluate"],
        "Evaluate Generated Patterns"
    ):
        return
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print("\nResults:")
    print("  - Embeddings: dataset/embeddings/")
    print("  - Generated: outputs/generated/")
    print("  - Evaluation: evaluation_results.json")


if __name__ == "__main__":
    main()

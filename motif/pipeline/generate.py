#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Generation Script (PLACEHOLDER - Requires trained model)
This script provides the interface for generating new patterns
Actual training must be done on GPU using Stable Diffusion
"""

import argparse
import logging
import os
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_patterns(
    checkpoint_path,
    prompt,
    motifs=None,
    colors=None,
    num_samples=4,
    output_dir="outputs/generated",
    base_model="runwayml/stable-diffusion-v1-5"
):
    """
    Generate Hmong patterns from text prompts using trained LoRA
    """
    
    print(f"\n🚀 Initializing Generation Pipeline...")
    print(f"   Base Model: {base_model}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"   Device: {device} ({dtype})")

    try:
        # Load Pipeline
        # Disable safety checker to avoid flagging abstract patterns
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None
        )
        
        # Load LoRA Weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"   Loading LoRA weights from {checkpoint_path}...")
            pipe.load_lora_weights(checkpoint_path)
        else:
            print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Using base model only.")

        pipe = pipe.to(device)

        # Construct Full Prompt
        full_prompt = f"{prompt}, Hmong traditional pattern style, intricate geometric textile design, vectors, high quality"
        
        if motifs:
            if isinstance(motifs, list):
                motif_str = ", ".join(motifs)
            else:
                motif_str = str(motifs)
            full_prompt += f", featuring {motif_str} motifs"
        
        if colors:
            if isinstance(colors, list):
                color_str = ", ".join(colors)
            else:
                color_str = str(colors)
            full_prompt += f", color scheme: {color_str}"

        # Negative Prompt
        negative_prompt = "blurry, low quality, distorted, watermark, text, signature, realistic photo, human face"

        print(f"\n🎨 Generating {num_samples} samples...")
        print(f"   Prompt: {full_prompt}")

        # Generate
        images = pipe(
            prompt=[full_prompt] * num_samples,
            negative_prompt=[negative_prompt] * num_samples,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images

        # Save Images
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            filename = f"generated_hmong_{i+1:03d}.png"
            save_dest = output_path / filename
            img.save(save_dest)
            saved_paths.append(str(save_dest))
            print(f"   ✅ Saved: {save_dest}")
            
        print(f"\n✨ Generation Complete! {len(saved_paths)} images saved to {output_dir}")
        return saved_paths

    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        # import traceback
        # traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate Hmong patterns")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to LoRA checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--motifs", type=str, nargs='+',
                       help="Specific motifs to include")
    parser.add_argument("--colors", type=str, nargs='+',
                       help="Colors to use")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="outputs/generated",
                       help="Output directory")
    
    args = parser.parse_args()
    
    generate_patterns(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        motifs=args.motifs,
        colors=args.colors,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    print("🎨 Hmong Pattern Generation Tool\n")
    
    # Example usage
    print("Example usage:")
    print('python generate_patterns.py \\')
    print('    --prompt "Hmong spiral pattern in traditional style" \\')
    print('    --motifs spiral zigzag \\')
    print('    --colors indigo white \\')
    print('    --num_samples 10\n')
    
    main()

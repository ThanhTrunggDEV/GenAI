#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine Visual and Cultural Embeddings
Creates unified representation for pattern conditioning
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class CombinedEmbedding(nn.Module):
    """
    Combines visual and cultural embeddings into unified representation
    """
    
    def __init__(self, visual_dim=512, cultural_dim=256, output_dim=768):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.cultural_dim = cultural_dim
        self.output_dim = output_dim
        
        # Projection layers (optional, can just concatenate)
        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.cultural_proj = nn.Linear(cultural_dim, cultural_dim)
        
        # Fusion layer
        combined_dim = visual_dim + cultural_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, visual_emb, cultural_emb):
        """
        Args:
            visual_emb: Tensor of shape (B, visual_dim)
            cultural_emb: Tensor of shape (B, cultural_dim)
        Returns:
            combined: Tensor of shape (B, output_dim)
        """
        # Project embeddings
        v_proj = self.visual_proj(visual_emb)
        c_proj = self.cultural_proj(cultural_emb)
        
        # Concatenate
        combined = torch.cat([v_proj, c_proj], dim=-1)
        
        # Fuse
        output = self.fusion(combined)
        
        return output


def create_combined_embeddings(visual_emb_file, cultural_emb_file, output_file):
    """
    Load visual and cultural embeddings and combine them
    
    Args:
        visual_emb_file: Path to visual embeddings .npz
        cultural_emb_file: Path to cultural embeddings .npz
        output_file: Path to save combined embeddings
    """
    print("📦 Loading embeddings...")
    
    # Load embeddings
    visual_data = np.load(visual_emb_file)
    cultural_data = np.load(cultural_emb_file)
    
    visual_emb = visual_data['embeddings']
    visual_files = visual_data['filenames']
    
    cultural_emb = cultural_data['embeddings']
    cultural_files = cultural_data['filenames']
    
    print(f"   Visual: {visual_emb.shape}")
    print(f"   Cultural: {cultural_emb.shape}")
    
    # Match filenames (remove extensions for matching)
    visual_map = {Path(f).stem: emb for f, emb in zip(visual_files, visual_emb)}
    cultural_map = {Path(f).stem: emb for f, emb in zip(cultural_files, cultural_emb)}
    
    # Find common files
    common_files = set(visual_map.keys()) & set(cultural_map.keys())
    print(f"   Common files: {len(common_files)}")
    
    # Combine embeddings
    combined_embeddings = []
    filenames = []
    
    combiner = CombinedEmbedding(
        visual_dim=visual_emb.shape[1],
        cultural_dim=cultural_emb.shape[1],
        output_dim=768
    )
    combiner.eval()
    
    for filename in sorted(common_files):
        v_emb = torch.FloatTensor(visual_map[filename]).unsqueeze(0)
        c_emb = torch.FloatTensor(cultural_map[filename]).unsqueeze(0)
        
        with torch.no_grad():
            combined = combiner(v_emb, c_emb)
        
        combined_embeddings.append(combined.squeeze(0).numpy())
        filenames.append(filename)
    
    combined_embeddings = np.array(combined_embeddings)
    
    # Save
    np.savez(output_file,
             embeddings=combined_embeddings,
             filenames=filenames)
    
    print(f"✅ Saved combined embeddings to {output_file}")
    print(f"   Shape: {combined_embeddings.shape}")
    
    return combined_embeddings, filenames


if __name__ == "__main__":
    print("🔧 Testing Combined Embedding...")
    
    # Test model
    combiner = CombinedEmbedding(visual_dim=512, cultural_dim=256, output_dim=768)
    
    visual_test = torch.randn(4, 512)
    cultural_test = torch.randn(4, 256)
    
    combined = combiner(visual_test, cultural_test)
    print(f"✅ Combination test: (4, 512) + (4, 256) → {combined.shape}")
    
    # Combine actual embeddings if they exist
    visual_emb_path = "dataset/embeddings/visual_embeddings.npz"
    cultural_emb_path = "dataset/embeddings/cultural_embeddings.npz"
    output_path = "dataset/embeddings/combined_embeddings.npz"
    
    if Path(visual_emb_path).exists() and Path(cultural_emb_path).exists():
        embeddings, filenames = create_combined_embeddings(
            visual_emb_path, cultural_emb_path, output_path
        )
        print(f"\n📊 Created {len(embeddings)} combined embeddings")
    else:
        print(f"\n⚠️  Run visual_encoder.py and cultural_encoder.py first")

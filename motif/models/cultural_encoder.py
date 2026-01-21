#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cultural Semantic Encoder for Hmong Patterns
Encodes cultural metadata (motifs, colors, meanings) into embeddings
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Vocabulary definitions
MOTIF_TYPES = [
    "spiral", "zigzag", "triangle", "diamond", "maze", "grid",
    "snail", "dragon", "bird", "flower", "butterfly",
    "chicken_foot", "pig_foot", "hemp_tool", "pumpkin_flower",
    "sun", "earring_shape", "fingerprint", "pillar", "concentric_circle",
    "border", "ladder", "plant", "leaf", "unknown"
]

COLORS = [
    "indigo", "black", "red", "yellow", "white",
    "blue", "green", "brown", "orange", "beige", "unknown"
]

SYMMETRY_TYPES = ["rotational", "bilateral", "radial", "asymmetric", "none", "unknown"]

REPETITION_TYPES = ["grid", "linear", "scattered", "concentric", "none", "unknown"]

COMPLEXITY_LEVELS = ["high", "medium", "low", "unknown"]

class CulturalSemanticEncoder(nn.Module):
    """
    Encodes cultural metadata into semantic embeddings
    """
    
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Create vocabular

y mappings
        self.motif_to_idx = {m: i for i, m in enumerate(MOTIF_TYPES)}
        self.color_to_idx = {c: i for i, c in enumerate(COLORS)}
        self.symmetry_to_idx = {s: i for i, s in enumerate(SYMMETRY_TYPES)}
        self.repetition_to_idx = {r: i for i, r in enumerate(REPETITION_TYPES)}
        self.complexity_to_idx = {c: i for i, c in enumerate(COMPLEXITY_LEVELS)}
        
        # Embedding layers
        motif_dim = len(MOTIF_TYPES)
        color_dim = len(COLORS)
        structure_dim = len(SYMMETRY_TYPES) + len(REPETITION_TYPES) + len(COMPLEXITY_LEVELS)
        
        # MLP to combine all features
        total_input_dim = motif_dim + color_dim + structure_dim
        
        self.embedding_net = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
    
    def encode_metadata(self, metadata):
        """
        Encode a single metadata JSON dict
        
        Args:
            metadata: Dict with pattern metadata
        Returns:
            embedding: Tensor of shape (embedding_dim,)
        """
        # Motif multi-hot encoding
        motif_vector = np.zeros(len(MOTIF_TYPES))
        specific_motifs = metadata.get("pattern_info", {}).get("specific_motifs", [])
        for motif in specific_motifs:
            if motif in self.motif_to_idx:
                motif_vector[self.motif_to_idx[motif]] = 1.0
        
        # Color multi-hot encoding
        color_vector = np.zeros(len(COLORS))
        colors = metadata.get("color_info", {}).get("colors", [])
        for color in colors:
            if color in self.color_to_idx:
                color_vector[self.color_to_idx[color]] = 1.0
        
        # Structure encoding (one-hot)
        symmetry = metadata.get("visual_structure", {}).get("symmetry", "unknown")
        repetition = metadata.get("visual_structure", {}).get("repetition", "unknown")
        complexity = metadata.get("visual_structure", {}).get("complexity", "unknown")
        
        symmetry_vector = np.zeros(len(SYMMETRY_TYPES))
        if symmetry in self.symmetry_to_idx:
            symmetry_vector[self.symmetry_to_idx[symmetry]] = 1.0
        
        repetition_vector = np.zeros(len(REPETITION_TYPES))
        if repetition in self.repetition_to_idx:
            repetition_vector[self.repetition_to_idx[repetition]] = 1.0
        
        complexity_vector = np.zeros(len(COMPLEXITY_LEVELS))
        if complexity in self.complexity_to_idx:
            complexity_vector[self.complexity_to_idx[complexity]] = 1.0
        
        # Concatenate all features
        features = np.concatenate([
            motif_vector,
            color_vector,
            symmetry_vector,
            repetition_vector,
            complexity_vector
        ])
        
        # Convert to tensor and pass through network
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            embedding = self.embedding_net(features_tensor)
        
        return embedding.squeeze(0)
    
    def forward(self, feature_batch):
        """
        Args:
            feature_batch: Tensor of shape (B, total_input_dim)
        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        return self.embedding_net(feature_batch)


def extract_cultural_features_batch(encoder, metadata_dir, output_file):
    """
    Extract cultural embeddings for all metadata files
    
    Args:
        encoder: CulturalSemanticEncoder instance
        metadata_dir: Directory containing JSON metadata files
        output_file: Path to save embeddings
    """
    from tqdm import tqdm
    
    metadata_dir = Path(metadata_dir)
    metadata_files = sorted(list(metadata_dir.glob("*.json")))
    
    embeddings = []
    filenames = []
    
    encoder.eval()
    
    print(f"Extracting cultural features from {len(metadata_files)} metadata files...")
    
    for meta_path in tqdm(metadata_files):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        embedding = encoder.encode_metadata(metadata)
        embeddings.append(embedding.numpy())
        filenames.append(meta_path.stem)
    
    embeddings = np.array(embeddings)
    
    # Save
    np.savez(output_file,
             embeddings=embeddings,
             filenames=filenames)
    
    print(f"✅ Saved cultural embeddings to {output_file}")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings, filenames


if __name__ == "__main__":
    # Test the encoder
    print("🔧 Testing Cultural Semantic Encoder...")
    
    encoder = CulturalSemanticEncoder(embedding_dim=256)
    print(f"✅ Encoder created: {sum(p.numel() for p in encoder.parameters())} params")
    
    # Test with sample metadata
    sample_metadata = {
        "pattern_info": {
            "specific_motifs": ["spiral", "zigzag"],
            "dominant_motif": "spiral"
        },
        "color_info": {
            "colors": ["indigo", "white"],
            "dominant_color": "indigo"
        },
        "visual_structure": {
            "symmetry": "rotational",
            "repetition": "grid",
            "complexity": "high"
        }
    }
    
    embedding = encoder.encode_metadata(sample_metadata)
    print(f"✅ Encoding test: metadata → {embedding.shape}")
    
    # Extract from training metadata
    metadata_dir = Path("dataset/training/train").parent.parent / "metadata"
    if metadata_dir.exists():
        output_path = "dataset/embeddings/cultural_embeddings.npz"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        embeddings, filenames = extract_cultural_features_batch(
            encoder, metadata_dir, output_path
        )
        print(f"\n📊 Extracted {len(embeddings)} cultural embeddings")
    else:
        print(f"\n⚠️  Metadata not found at {metadata_dir}")

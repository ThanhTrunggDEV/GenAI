#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Encoder для Hmong Pattern Features
Extracts visual features from pattern images using pretrained CNN
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

class HmongVisualEncoder(nn.Module):
    """
    Visual encoder for Hmong patterns
    Extracts motif structure, texture, and spatial features
    """
    
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone initially (can fine-tune later)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Add custom head for pattern-specific features
        self.feature_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)
        Returns:
            features: Tensor of shape (B, embedding_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Pass through custom head
        embeddings = self.feature_head(features)
        
        return embeddings
    
    def encode_image(self, image_path):
        """
        Encode a single image file
        
        Args:
            image_path: Path to image file
        Returns:
            embedding: Numpy array of shape (embedding_dim,)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dim
        
        # Get embedding
        with torch.no_grad():
            embedding = self.forward(image_tensor)
        
        return embedding.squeeze(0).numpy()
    
    def fine_tune(self, unfreeze_layers=2):
        """
        Unfreeze last N layers of backbone for fine-tuning
        """
        # Unfreeze feature head
        for param in self.feature_head.parameters():
            param.requires_grad = True
        
        # Unfreeze last N layers of backbone
        children = list(self.backbone.children())
        for layer in children[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


def extract_visual_features_batch(encoder, image_dir, output_file):
    """
    Extract visual features for all images in a directory
    
    Args:
        encoder: HmongVisualEncoder instance
        image_dir: Directory containing images
        output_file: Path to save embeddings (.npy or .pt)
    """
    import numpy as np
    from tqdm import tqdm
    
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    embeddings = []
    filenames = []
    
    encoder.eval()
    
    print(f"Extracting features from {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        embedding = encoder.encode_image(img_path)
        embeddings.append(embedding)
        filenames.append(img_path.name)
    
    embeddings = np.array(embeddings)
    
    # Save
    np.savez(output_file, 
             embeddings=embeddings,
             filenames=filenames)
    
    print(f"✅ Saved embeddings to {output_file}")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings, filenames


if __name__ == "__main__":
    # Test the encoder
    print("🔧 Testing Visual Encoder...")
    
    encoder = HmongVisualEncoder(embedding_dim=512)
    print(f"✅ Encoder created: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)} trainable params")
    
    # Test with sample data
    sample_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = encoder(sample_input)
    print(f"✅ Forward pass: {sample_input.shape} → {output.shape}")
    
    # Extract features from training images
    dataset_dir = Path("dataset/training/train/images")
    if dataset_dir.exists():
        output_path = "dataset/embeddings/visual_embeddings.npz"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        embeddings, filenames = extract_visual_features_batch(
            encoder, dataset_dir, output_path
        )
        print(f"\n📊 Extracted {len(embeddings)} visual embeddings")
    else:
        print(f"\n⚠️  Dataset not found at {dataset_dir}")
        print("   Run this after preparing training data")

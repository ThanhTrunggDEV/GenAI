#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Loss Functions for Culturally-Constrained Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


from .visual_encoder import HmongVisualEncoder

class CulturalConsistencyLoss(nn.Module):
    """
    Loss to enforce cultural consistency between generated and real patterns
    Compares cultural embeddings
    """
    
    def __init__(self, cultural_encoder, visual_encoder=None):
        super().__init__()
        self.cultural_encoder = cultural_encoder
        
        # Use provided visual encoder or create new one matching cultural dim
        if visual_encoder:
            self.visual_encoder = visual_encoder
        else:
            # Initialize with same embedding dim as cultural encoder
            # Default cultural dim is 256
            emb_dim = getattr(cultural_encoder, 'embedding_dim', 256)
            self.visual_encoder = HmongVisualEncoder(embedding_dim=emb_dim, pretrained=True)
            
        # Freeze visual encoder parameters to avoid "chasing a moving target"
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, generated_images, target_metadata=None, target_images=None):
        """
        Args:
            generated_images: Generated pattern images (B, C, H, W). Assumed to be in [-1, 1] range.
            target_metadata: Target cultural embeddings (B, dim) (Optional)
            target_images: Target real images (B, C, H, W) (Optional)
        Returns:
            loss: Cultural consistency loss
        """
        # 1. Denormalize from [-1, 1] to [0, 1]
        images = (generated_images.clone() + 1.0) * 0.5
        images = torch.clamp(images, 0.0, 1.0)
        
        # 2. Resize and Normalize for Visual Encoder
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        
        if images.shape[-2:] != (224, 224):
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            images_resized = images
            
        images_normalized = (images_resized - mean) / std
            
        # 3. Extract visual embeddings for Generated Images
        self.visual_encoder.eval()
        visual_emb = self.visual_encoder(images_normalized)
        
        # 4. Determine Target Embeddings
        # ARIORITIZE target_images (Perceptual Loss) over target_metadata
        # because visual_encoder is likely not aligned with cultural_metadata space yet.
        
        if target_images is not None:
            # Feature matching with Real Images
            with torch.no_grad():
                # Process target images exactly like generated ones
                t_images = (target_images.clone() + 1.0) * 0.5
                t_images = torch.clamp(t_images, 0.0, 1.0)
                
                if t_images.shape[-2:] != (224, 224):
                    t_images_resized = F.interpolate(t_images, size=(224, 224), mode='bilinear', align_corners=False)
                else:
                    t_images_resized = t_images
                
                t_images_normalized = (t_images_resized - mean) / std
                target_emb = self.visual_encoder(t_images_normalized)
                
        elif target_metadata is not None:
            # Fallback to metadata matching (only works if encoders are aligned)
            target_emb = target_metadata
        else:
            return torch.tensor(0.0, device=generated_images.device)
            
        # 5. Compute Cosine Similarity Loss
        target = torch.ones(visual_emb.shape[0], device=visual_emb.device)
        loss = self.cosine_loss(visual_emb, target_emb, target)
        
        return loss


class ColorPaletteLoss(nn.Module):
    """
    Loss to ensure generated patterns match target color palette
    """
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def extract_color_histogram(self, images, bins=256):
        """
        Extract color histogram from images
        
        Args:
            images: Tensor of shape (B, C, H, W)
            bins: Number of histogram bins
        Returns:
            histograms: Tensor of shape (B, C, bins)
        """
        batch_size, channels = images.shape[0], images.shape[1]
        histograms = []
        
        for i in range(batch_size):
            img_hists = []
            for c in range(channels):
                channel = images[i, c].flatten()
                hist = torch.histc(channel, bins=bins, min=0.0, max=1.0)
                hist = hist / hist.sum()  # Normalize
                img_hists.append(hist)
            histograms.append(torch.stack(img_hists))
        
        return torch.stack(histograms)
    
    def forward(self, generated_images, target_images):
        """
        Args:
            generated_images: Generated images (B, C, H, W)
            target_images: Target images (B, C, H, W)
        Returns:
            loss: Color palette matching loss
        """
        gen_hist = self.extract_color_histogram(generated_images)
        target_hist = self.extract_color_histogram(target_images)
        
        loss = self.l1_loss(gen_hist, target_hist)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for culturally-constrained generation
    L_total = λ₁·L_diffusion + λ₂·L_cultural + λ₃·L_color
    """
    
    def __init__(self, cultural_encoder=None, 
                 lambda_diffusion=1.0, lambda_cultural=0.3, lambda_color=0.2):
        super().__init__()
        
        self.lambda_diffusion = lambda_diffusion
        self.lambda_cultural = lambda_cultural
        self.lambda_color = lambda_color
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.cultural_loss = CulturalConsistencyLoss(cultural_encoder) if cultural_encoder else None
        self.color_loss = ColorPaletteLoss()
    
    def forward(self, predicted_noise, actual_noise, generated_images=None, 
                target_images=None, target_metadata=None):
        """
        Compute combined loss
        
        Args:
            predicted_noise: Predicted noise from diffusion model
            actual_noise: Actual noise added
            generated_images: Generated images (optional)
            target_images: Target images (optional)
            target_metadata: Target cultural metadata (optional)
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Standard diffusion loss (MSE between predicted and actual noise)
        loss_diffusion = self.mse_loss(predicted_noise, actual_noise)
        
        total_loss = self.lambda_diffusion * loss_diffusion
        loss_dict = {'diffusion': loss_diffusion.item()}
        
        # Cultural consistency loss (if applicable)
        if self.cultural_loss is not None:
             # Logic to determine arguments for cultural loss
             # We can pass target_images (real images) as reference for visual consistency
             # Or target_metadata if we trust the alignment
             
             # Prioritize passing target_images if available to enable Perceptual Loss
             c_loss = self.cultural_loss(
                 generated_images, 
                 target_metadata=target_metadata,
                 target_images=target_images
             )
             total_loss += self.lambda_cultural * c_loss
             loss_dict['cultural'] = c_loss.item()
        
        # Color palette loss (if applicable)
        if generated_images is not None and target_images is not None:
            loss_color = self.color_loss(generated_images, target_images)
            total_loss += self.lambda_color * loss_color
            loss_dict['color'] = loss_color.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("🧪 Testing Loss Functions...")
    
    # Test color palette loss
    color_loss_fn = ColorPaletteLoss()
    
    fake_gen = torch.rand(2, 3, 64, 64)
    fake_target = torch.rand(2, 3, 64, 64)
    
    color_loss = color_loss_fn(fake_gen, fake_target)
    print(f"✅ Color Loss Test: {color_loss.item():.4f}")
    
    # Test combined loss
    combined_loss_fn = CombinedLoss()
    
    pred_noise = torch.rand(2, 4, 64, 64)
    actual_noise = torch.rand(2, 4, 64, 64)
    
    total_loss, loss_dict = combined_loss_fn(
        pred_noise, actual_noise,
        generated_images=fake_gen,
        target_images=fake_target
    )
    
    print(f"✅ Combined Loss Test:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")

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

class CulturalConsistencyLoss(nn.Module):
    """
    Loss to enforce cultural consistency between generated and real patterns
    Compares cultural embeddings
    """
    
    def __init__(self, cultural_encoder):
        super().__init__()
        self.cultural_encoder = cultural_encoder
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, generated_images, target_metadata):
        """
        Args:
            generated_images: Generated pattern images
            target_metadata: Target cultural metadata
        Returns:
            loss: Cultural consistency loss
        """
        # Extract cultural features from generated images
        # (This would require a reverse encoder - simplified for now)
        
        # For now, use KL divergence on metadata embeddings
        # In practice, you'd extract features from generated images
        
        # Placeholder: return zero loss
        # Real implementation would compare embeddings
        return torch.tensor(0.0, device=generated_images.device)


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
        if self.cultural_loss is not None and generated_images is not None and target_metadata is not None:
            loss_cultural = self.cultural_loss(generated_images, target_metadata)
            total_loss += self.lambda_cultural * loss_cultural
            loss_dict['cultural'] = loss_cultural.item()
        
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

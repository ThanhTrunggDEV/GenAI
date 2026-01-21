#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Validator
Validates geometric properties: symmetry, repetition, proportions
"""

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

class StructureValidator:
    """
    Validates structural properties of patterns
    """
    
    def __init__(self, symmetry_threshold=0.8):
        self.symmetry_threshold = symmetry_threshold
    
    def check_symmetry(self, image):
        """
        Check different types of symmetry
        
        Args:
            image: PIL Image or numpy array
        Returns:
            symmetry_type: Detected symmetry type
            score: Symmetry score (0-1)
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))  # Grayscale
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = image.shape
        
        # Test bilateral symmetry (vertical axis)
        left_half = image[:, :w//2]
        right_half = image[:, w//2:]
        right_flipped = np.fliplr(right_half)
        
        # Resize if needed
        if left_half.shape != right_flipped.shape:
            min_w = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_w]
            right_flipped = right_flipped[:, :min_w]
        
        bilateral_score = np.mean(left_half == right_flipped)
        
        # Test rotational symmetry (180 degrees)
        rotated_180 = np.rot90(image, 2)
        rotational_180_score = np.mean(image == rotated_180)
        
        # Test rotational symmetry (90 degrees)
        rotated_90 = np.rot90(image, 1)
        rotational_90_score = np.mean(image == rotated_90)
        
        # Determine type
        scores = {
            'bilateral': bilateral_score,
            'rotational_180': rotational_180_score,
            'rotational_90': rotational_90_score
        }
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_type == 'rotational_90' or best_type == 'rotational_180':
            symmetry_type = 'rotational'
        else:
            symmetry_type = best_type
        
        return symmetry_type, best_score
    
    def check_repetition(self, image):
        """
        Check for repetitive patterns using FFT
        
        Args:
            image: Grayscale image array
        Returns:
            has_repetition: Boolean
            dominant_frequency: Dominant frequency detected
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Find peaks in frequency domain
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Mask out center (DC component)
        mask = np.ones_like(magnitude)
        cv2.circle(mask, (center_w, center_h), 20, 0, -1)
        magnitude_masked = magnitude * mask
        
        # Find dominant frequency
        max_val = np.max(magnitude_masked)
        threshold = max_val * 0.3
        
        has_repetition = np.sum(magnitude_masked > threshold) > 10
        
        return has_repetition, max_val
    
    def check_proportions(self, image):
        """
        Check if pattern follows golden ratio or other aesthetic proportions
        
        Args:
            image: Image array
        Returns:
            proportion_score: How well it follows aesthetic proportions
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Golden ratio ≈ 1.618
        golden_ratio = 1.618
        distance_from_golden = abs(aspect_ratio - golden_ratio)
        
        # Also check for square (1:1) which is common in patterns
        distance_from_square = abs(aspect_ratio - 1.0)
        
        # Score based on distance (closer = better)
        if distance_from_square < 0.1:
            proportion_score = 1.0
        elif distance_from_golden < 0.2:
            proportion_score = 0.9
        else:
            proportion_score = max(0.5, 1.0 - distance_from_golden)
        
        return proportion_score
    
    def validate(self, image, expected_metadata=None):
        """
        Main validation function
        
        Args:
            image: Generated image
            expected_metadata: Expected structure metadata
        Returns:
            is_valid: Boolean
            scores: Dict with validation scores
        """
        symmetry_type, symmetry_score = self.check_symmetry(image)
        has_repetition, freq_strength = self.check_repetition(image)
        proportion_score = self.check_proportions(image)
        
        # Check against expected if provided
        if expected_metadata:
            expected_sym = expected_metadata.get("visual_structure", {}).get("symmetry", "unknown")
            expected_rep = expected_metadata.get("visual_structure", {}).get("repetition", "unknown")
            
            sym_match = (expected_sym == "unknown" or expected_sym == symmetry_type)
            rep_match = (expected_rep == "unknown" or 
                        (expected_rep != "none" and has_repetition))
        else:
            sym_match = True
            rep_match = True
        
        # Overall validation
        is_valid = (
            symmetry_score >= self.symmetry_threshold and
            proportion_score >= 0.7 and
            sym_match and rep_match
        )
        
        scores = {
            "symmetry_type": symmetry_type,
            "symmetry_score": float(symmetry_score),
            "has_repetition": has_repetition,
            "frequency_strength": float(freq_strength),
            "proportion_score": float(proportion_score),
            "symmetry_match": sym_match,
            "repetition_match": rep_match
        }
        
        return is_valid, scores


if __name__ == "__main__":
    print("🔍 Testing Structure Validator...")
    
    validator = StructureValidator()
    
    # Create test image with symmetry
    test_img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 200), (312, 312), 255, -1)
    
    test_metadata = {
        "visual_structure": {
            "symmetry": "bilateral",
            "repetition": "grid"
        }
    }
    
    is_valid, scores = validator.validate(test_img, test_metadata)
    
    print(f"✅ Structure Test:")
    print(f"   Valid: {is_valid}")
    for k, v in scores.items():
        print(f"   {k}: {v}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motif Consistency Validator
Validates that generated patterns contain expected motifs
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

class MotifValidator:
    """
    Validates motif consistency in generated patterns
    """
    
    def __init__(self, visual_encoder=None):
        self.visual_encoder = visual_encoder
        self.threshold = 0.7  # 70% consistency threshold
    
    def detect_motifs_simple(self, image, expected_motifs):
        """
        Simple motif detection using edge detection and contours
        
        Args:
            image: PIL Image or numpy array
            expected_motifs: List of expected motif names
        Returns:
            detected_motifs: List of detected motif types
            confidence: Detection confidence score
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours to detect motif types
        detected = []
        
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Circularity check (for spiral/circular motifs)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
                
                if circularity > 0.7:
                    if "spiral" in expected_motifs or "snail" in expected_motifs:
                        detected.append("spiral")
            
            # Approximate polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Triangle check
            if len(approx) == 3:
                if "triangle" in expected_motifs:
                    detected.append("triangle")
            
            # Diamond/square check
            elif len(approx) == 4:
                if "diamond" in expected_motifs or "grid" in expected_motifs:
                    detected.append("diamond")
        
        # Calculate overlap with expected
        detected_set = set(detected)
        expected_set = set(expected_motifs)
        
        if len(expected_set) == 0:
            confidence = 1.0
        else:
            overlap = len(detected_set & expected_set)
            confidence = overlap / len(expected_set)
        
        return list(detected_set), confidence
    
    def validate(self, image, expected_metadata):
        """
        Main validation function
        
        Args:
            image: Generated image (PIL or numpy)
            expected_metadata: Dict with expected pattern info
        Returns:
            is_valid: Boolean indicating if validation passed
            score: Consistency score
            details: Dict with validation details
        """
        expected_motifs = expected_metadata.get("pattern_info", {}).get("specific_motifs", [])
        
        detected_motifs, confidence = self.detect_motifs_simple(image, expected_motifs)
        
        is_valid = confidence >= self.threshold
        
        details = {
            "expected_motifs": expected_motifs,
            "detected_motifs": detected_motifs,
            "confidence": confidence,
            "threshold": self.threshold
        }
        
        return is_valid, confidence, details


if __name__ == "__main__":
    print("🔍 Testing Motif Validator...")
    
    validator = MotifValidator()
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color='white')
    
    test_metadata = {
        "pattern_info": {
            "specific_motifs": ["spiral", "triangle", "diamond"]
        }
    }
    
    is_valid, score, details = validator.validate(test_img, test_metadata)
    
    print(f"✅ Validation Test:")
    print(f"   Valid: {is_valid}")
    print(f"   Score: {score:.2f}")
    print(f"   Expected: {details['expected_motifs']}")
    print(f"   Detected: {details['detected_motifs']}")

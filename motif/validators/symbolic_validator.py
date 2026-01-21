#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symbolic Correctness Validator
Ensures cultural symbols are used correctly according to traditional rules
"""

import numpy as np
from PIL import Image
import colorsys

# Cultural rules for Hmong patterns
CULTURAL_RULES = {
    "funeral": {
        "required_colors": ["white"],
        "forbidden_colors": ["red", "yellow"],
        "description": "Funeral garments must be primarily white, avoid bright colors"
    },
    "wedding": {
        "required_motifs": ["dragon", "flower", "bird"],
        "preferred_colors": ["red", "yellow"],
        "description": "Wedding patterns should include auspicious motifs"
    },
    "festival": {
        "preferred_colors": ["red", "yellow", "blue"],
        "min_colors": 3,
        "description": "Festival clothing should be colorful"
    },
    "daily_wear": {
        "color_scheme": "traditional",
        "allowed_techniques": ["batik", "embroidery", "applique"],
        "description": "Daily wear follows traditional color schemes"
    }
}

class SymbolicValidator:
    """
    Validates symbolic correctness of patterns
    """
    
    def __init__(self):
        self.rules = CULTURAL_RULES
    
    def extract_dominant_colors(self, image, n_colors=5):
        """
        Extract dominant colors from image
        
        Args:
            image: PIL Image
            n_colors: Number of dominant colors to extract
        Returns:
            color_names: List of color names
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Reshape to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Simple color quantization
        # In practice, use k-means or color histograms
        mean_color = pixels.mean(axis=0)
        
        # Map RGB to color names (simplified)
        color_names = []
        
        # Check for dominant colors
        r, g, b = mean_color
        
        if r > 200 and g > 200 and b > 200:
            color_names.append("white")
        elif r < 50 and g < 50 and b < 50:
            color_names.append("black")
        elif r > 150 and g < 100 and b < 100:
            color_names.append("red")
        elif r < 100 and g < 100 and b > 150:
            color_names.append("blue")
        elif r > 150 and g > 150 and b < 100:
            color_names.append("yellow")
        
        # Check for indigo (traditional Hmong color)
        if 40 < b < 120 and r < 50 and g < 50:
            color_names.append("indigo")
        
        return color_names if color_names else ["unknown"]
    
    def validate_ritual_use(self, image, ritual_use, detected_motifs=None):
        """
        Validate pattern for specific ritual use
        
        Args:
            image: PIL Image or numpy array
            ritual_use: String ritual use type
            detected_motifs: Optional list of detected motifs
        Returns:
            is_valid: Boolean
            violations: List of rule violations
        """
        if ritual_use not in self.rules:
            return True, []  # No rules to check
        
        rules = self.rules[ritual_use]
        violations = []
        
        # Extract colors from image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        detected_colors = self.extract_dominant_colors(image)
        
        # Check required colors
        if "required_colors" in rules:
            for req_color in rules["required_colors"]:
                if req_color not in detected_colors:
                    violations.append(f"Missing required color: {req_color}")
        
        # Check forbidden colors
        if "forbidden_colors" in rules:
            for forbidden in rules["forbidden_colors"]:
                if forbidden in detected_colors:
                    violations.append(f"Contains forbidden color: {forbidden}")
        
        # Check required motifs
        if "required_motifs" in rules and detected_motifs:
            for req_motif in rules["required_motifs"]:
                if req_motif not in detected_motifs:
                    violations.append(f"Missing required motif: {req_motif}")
        
        # Check minimum colors
        if "min_colors" in rules:
            if len(detected_colors) < rules["min_colors"]:
                violations.append(f"Insufficient colors: {len(detected_colors)} < {rules['min_colors']}")
        
        is_valid = len(violations) == 0
        
        return is_valid, violations
    
    def validate(self, image, metadata):
        """
        Main validation function
        
        Args:
            image: Generated image
            metadata: Pattern metadata with ritual_use
        Returns:
            is_valid: Boolean
            violations: List of violations
            details: Dict with validation details
        """
        ritual_use = metadata.get("cultural_meaning", {}).get("ritual_use", "unknown")
        detected_motifs = metadata.get("pattern_info", {}).get("specific_motifs", [])
        
        is_valid, violations = self.validate_ritual_use(image, ritual_use, detected_motifs)
        
        details = {
            "ritual_use": ritual_use,
            "rule": self.rules.get(ritual_use, {}),
            "violations": violations
        }
        
        return is_valid, violations, details


if __name__ == "__main__":
    print("🔍 Testing Symbolic Validator...")
    
    validator = SymbolicValidator()
    
    # Test with funeral pattern (should be white, no red)
    test_img = Image.new('RGB', (512, 512), color=(255, 255, 255))  # White
    
    test_metadata = {
        "cultural_meaning": {
            "ritual_use": "funeral"
        },
        "pattern_info": {
            "specific_motifs": ["spiral"]
        }
    }
    
    is_valid, violations, details = validator.validate(test_img, test_metadata)
    
    print(f"✅ Funeral Pattern Test:")
    print(f"   Valid: {is_valid}")
    print(f"   Violations: {violations}")
    
    # Test with red image (should fail for funeral)
    test_img_red = Image.new('RGB', (512, 512), color=(255, 0, 0))  # Red
    is_valid2, violations2, details2 = validator.validate(test_img_red, test_metadata)
    
    print(f"\n❌ Red Funeral Pattern Test (should fail):")
    print(f"   Valid: {is_valid2}")
    print(f"   Violations: {violations2}")

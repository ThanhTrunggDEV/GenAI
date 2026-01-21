#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Validation Pipeline
Runs all validators and makes accept/reject decisions
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from validators.motif_validator import MotifValidator
from validators.symbolic_validator import SymbolicValidator
from validators.structure_validator import StructureValidator

class ValidationPipeline:
    """
    Integrated validation pipeline for generated patterns
    """
    
    def __init__(self):
        self.motif_validator = MotifValidator()
        self.symbolic_validator = SymbolicValidator()
        self.structure_validator = StructureValidator()
    
    def validate_pattern(self, image, metadata):
        """
        Run all validators on a pattern
        
        Args:
            image: Generated image (PIL or numpy)
            metadata: Pattern metadata dict
        Returns:
            overall_valid: Boolean - passes all checks
            results: Dict with individual validation results
        """
        results = {}
        
        # 1. Motif Consistency Check
        motif_valid, motif_score, motif_details = self.motif_validator.validate(image, metadata)
        results['motif'] = {
            'valid': motif_valid,
            'score': motif_score,
            'details': motif_details
        }
        
        # 2. Symbolic Correctness Check
        symbolic_valid, violations, symbolic_details = self.symbolic_validator.validate(image, metadata)
        results['symbolic'] = {
            'valid': symbolic_valid,
            'violations': violations,
            'details': symbolic_details
        }
        
        # 3. Structure Check
        structure_valid, structure_scores = self.structure_validator.validate(image, metadata)
        results['structure'] = {
            'valid': structure_valid,
            'scores': structure_scores
        }
        
        # Overall validation
        overall_valid = motif_valid and symbolic_valid and structure_valid
        
        results['overall'] = {
            'valid': overall_valid,
            'passed_checks': sum([motif_valid, symbolic_valid, structure_valid]),
            'total_checks': 3
        }
        
        return overall_valid, results
    
    def validate_with_retry(self, generator_fn, metadata, max_attempts=5):
        """
        Generate and validate with automatic retry
        
        Args:
            generator_fn: Function that generates an image
            metadata: Target metadata
            max_attempts: Maximum generation attempts
        Returns:
            best_image: Best generated image
            best_results: Validation results for best image
        """
        best_image = None
        best_results = None
        best_score = -1
        
        for attempt in range(max_attempts):
            # Generate
            image = generator_fn(metadata)
            
            # Validate
            is_valid, results = self.validate_pattern(image, metadata)
            
            # Calculate overall score
            score = (
                results['motif']['score'] * 0.4 +
                (1.0 if results['symbolic']['valid'] else 0.0) * 0.3 +
                results['structure']['scores']['symmetry_score'] * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_image = image
                best_results = results
            
            # Early exit if perfect
            if is_valid:
                print(f"✅ Valid pattern generated on attempt {attempt + 1}")
                break
        
        return best_image, best_results


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    print("🔍 Testing Validation Pipeline...")
    
    pipeline = ValidationPipeline()
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color='white')
    
    test_metadata = {
        "pattern_info": {
            "specific_motifs": ["spiral", "zigzag"],
            "dominant_motif": "spiral"
        },
        "cultural_meaning": {
            "ritual_use": "daily_wear"
        },
        "visual_structure": {
            "symmetry": "rotational",
            "repetition": "grid"
        }
    }
    
    is_valid, results = pipeline.validate_pattern(test_img, test_metadata)
    
    print(f"\n📊 Validation Results:")
    print(f"   Overall Valid: {is_valid}")
    print(f"   Motif Check: {'✅' if results['motif']['valid'] else '❌'}")
    print(f"   Symbolic Check: {'✅' if results['symbolic']['valid'] else '❌'}")
    print(f"   Structure Check: {'✅' if results['structure']['valid'] else '❌'}")
    print(f"   Passed: {results['overall']['passed_checks']}/3")

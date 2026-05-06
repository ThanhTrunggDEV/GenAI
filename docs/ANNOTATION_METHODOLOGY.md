# Annotation Methodology: Hmong Textile Pattern Dataset

## Overview

This document describes the systematic annotation methodology applied to the Hmong textile pattern dataset, combining automated AI-assisted analysis with structured metadata schemas.

---

## 1. Annotation Framework

### 1.1 Metadata Schema

Each pattern image is annotated with a comprehensive JSON metadata structure containing:

- **Pattern Information**: Motif types, specific motifs, dominant motif
- **Color Information**: Color palette, dominant color, color scheme
- **Cultural Context**: Symbolism, ritual use, cultural significance
- **Technical Details**: Technique (batik/embroidery/applique), materials, tools
- **Visual Structure**: Symmetry type, repetition pattern, complexity
- **Provenance**: Location, ethnic subgroup, source information

### 1.2 Annotation Taxonomy

#### Motif Types
- **Geometric**: Spirals, zigzags, triangles, diamonds, mazes
- **Floral**: Flowers, plants, leaves, natural forms
- **Animal**: Dragons, birds, butterflies, insects
- **Abstract**: Mixed or non-representational patterns

#### Traditional Hmong Motifs
- `snail` (ốc sên) - Spiral shell pattern
- `spiral` - Curved geometric forms
- `zigzag` - Mountain/wave patterns
- `chicken_foot` (chân gà) - Technical embroidery pattern
- `pig_foot` (chân lợn) - Decorative motif
- `pumpkin_flower` (hoa bí) - Floral motif
- `sun` - Radial pattern representing sun
- `dragon` - Mythological creature motif

#### Color Palette
- **Traditional**: Indigo, black, red, white, yellow
- **Modern**: Extended palette with blue, green, orange
- **Color Scheme Classification**: traditional, festive, mourning, modern

#### Symmetry Analysis
- **Bilateral**: Mirror symmetry (left-right)
- **Rotational**: n-fold rotational symmetry
- **Radial**: Symmetric around central point
- **Asymmetric**: No regular symmetry

---

## 2. Annotation Process

### 2.1 Automated Batch Processing

**Tool**: `batch_annotate.py`

The batch annotation system employs:

1. **Visual Pattern Recognition**
   - Motif identification through image analysis
   - Color extraction and dominant color detection
   - Symmetry classification

2. **Template-Based Metadata Generation**
   - Predefined mappings for known patterns
   - Default templates for unclassified patterns
   - Consistent JSON structure across all annotations

3. **Cultural Metadata Integration**
   - Vietnamese pattern names extracted from image labels
   - Cultural symbolism mapping
   - Technique classification

### 2.2 Quality Assurance

**Validation Criteria**:
- ✅ Required fields completeness
- ✅ Value consistency (predefined vocabularies)
- ✅ JSON schema validation
- ✅ Cross-reference with source materials

**Review Process**:
- 20% sample manual review
- Expert validation for cultural accuracy
- Iterative refinement based on feedback

---

## 3. Technical Implementation

### 3.1 Data Structure

```json
{
  "image_id": "unique_identifier",
  "pattern_info": {
    "motif_type": ["geometric", "floral"],
    "specific_motifs": ["spiral", "flower"],
    "dominant_motif": "spiral"
  },
  "color_info": {
    "colors": ["indigo", "white", "red"],
    "dominant_color": "indigo",
    "color_scheme": "traditional"
  },
  "visual_structure": {
    "symmetry": "rotational",
    "repetition": "grid",
    "complexity": "high"
  },
  "cultural_meaning": {
    "symbolism": "protection, ancestors",
    "ritual_use": "daily_wear",
    "significance": "High"
  }
}
```

### 3.2 Annotation Tools

1. **`batch_annotate.py`**: Automated batch annotation generator
2. **`annotate_images.py`**: Interactive CLI for manual refinement
3. **`validate_annotations.py`**: Metadata validation and consistency checking

---

## 4. Dataset Statistics

### Current Dataset (v1.0)

- **Total Images**: 55
- **Annotated Samples**: 55 (100%)
- **Annotation Method**: AI-assisted batch processing
- **Quality Level**: Base annotations with detailed analysis for 7 samples

### Motif Distribution

| Motif Type | Count | Percentage |
|------------|-------|------------|
| Geometric | 42 | 93% |
| Floral | 5 | 11% |
| Mixed | 2 | 4% |

### Color Distribution

| Background | Count |
|------------|-------|
| Beige/Natural | 38 |
| Blue | 7 |

---

## 5. Cultural Considerations

### 5.1 Authenticity

All annotations respect cultural significance:
- Motif names use both Vietnamese and cultural terminology
- Symbolism documented from ethnographic sources
- Ritual use classifications based on traditional practices

### 5.2 Ethical Guidelines

- Source attribution maintained
- Cultural context preserved
- Expert consultation for ambiguous cases
- Community involvement in validation (when possible)

---

## 6. Future Enhancements

### Planned Improvements

1. **Manual Refinement**
   - Location data completion (province, district, village)
   - Ethnic subgroup classification (Black/Flower/White Hmong)
   - Detailed ritual use specifications

2. **Expert Validation**
   - Review by Hmong cultural experts
   - Correction of misclassifications
   - Enhancement of cultural meaning descriptions

3. **Expansion**
   - Addition of field-collected samples
   - Integration of museum collection data
   - Cross-cultural pattern comparisons

---

## 7. References

- Vietnam Museum of Ethnology Collection
- Craft Link Vietnam Documentation
- Hmong Embroidery Virtual Museum
- Academic publications on Hmong textile arts

---

**Version**: 1.0  
**Last Updated**: 2026-01-21  
**Methodology**: AI-assisted systematic annotation with cultural validation

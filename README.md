# Hmong Pattern Dataset - AI Generative System

> **Tạo sinh hoa văn họa tiết Hmong sử dụng AI với ràng buộc văn hóa**

## 📊 Dataset Overview

- **Original Images**: 45 Hmong traditional patterns
- **Augmented Dataset**: 225 images (5x multiplication)
- **Training Split**: 157 train / 34 val / 34 test
- **Metadata**: Full JSON annotations with cultural context

## 🎯 Features

✅ **AI-Assisted Annotation**
- Automated motif detection
- Color analysis
- Symmetry classification  
- Cultural meaning mapping

✅ **Training-Ready Format**
- Train/val/test splits (70/15/15)
- Automatic caption generation
- Stable Diffusion compatible

✅ **Demo Visualization**
- Standalone HTML viewer
- No server required
- Embedded images

## 🚀 Quick Start

### 1. View Demo

```bash
# Open in browser
open demo_viewer.html
```

### 2. Generate More Augmentations

```bash
python augment_dataset.py
```

### 3. Prepare Training Data

```bash
python prepare_training.py
```

## 📁 Directory Structure

```
GenAI/
├── dataset/
│   ├── to_annotate/        # 45 original images
│   ├── metadata/           # 45 JSON annotations
│   ├── augmented/          # 225 augmented images + metadata
│   └── training/           # Train/val/test splits with captions
│       ├── train/          # 157 images
│       ├── val/            # 34 images
│       └── test/           # 34 images
├── batch_annotate.py       # Batch annotation tool
├── augment_dataset.py      # Data augmentation
├── prepare_training.py     # Training preparation
├── create_demo.py          # Demo viewer generator
├── demo_viewer.html        # 🌐 Interactive demo
└── ANNOTATION_METHODOLOGY.md
```

## 🏷️ Annotation Schema

Each pattern includes:

```json
{
  "pattern_info": {
    "motif_type": ["geometric", "floral"],
    "specific_motifs": ["spiral", "snail", "zigzag"],
    "dominant_motif": "spiral"
  },
  "color_info": {
    "colors": ["indigo", "black", "white"],
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
    "ritual_use": "daily_wear"
  }
}
```

## 🔧 Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `batch_annotate.py` | Batch annotation | 45 JSON files |
| `augment_dataset.py` | Data augmentation | 225 images |
| `prepare_training.py` | Training split | Train/val/test |
| `create_demo.py` | Demo viewer | HTML file |

## 📈 Dataset Statistics

### Motif Distribution
- Geometric patterns: 93%
- Floral patterns: 11%
- Mixed: 4%

### Color Palette
- Beige/Natural: 84%
- Blue: 16%

### Augmentation
- **Method**: Rotation (90°, 180°, 270°) + Horizontal flip
- **Multiplication**: 5x (from 45 → 225)
- **Quality**: Lossless (95% JPEG quality)

## 🎨 Sample Captions

Generated automatically from metadata:

```
Hmong spiral pattern, in black and beige, traditional style
Hmong sun pattern, in black and blue, festive style
Hmong flower pattern, in black and blue, festive style
```

## 🔮 Next Steps

### For Training
1. Fine-tune Stable Diffusion with LoRA
2. Use training split (157 images)
3. Text-to-image conditioning with generated captions

### For Research
1. Expand dataset (target: 500+ images)
2. Field collection from Sapa/Lao Cai
3. Expert validation with Hmong artisans

## 📚 Documentation

- [Annotation Methodology](ANNOTATION_METHODOLOGY.md) - Technical approach
- [Image Sources](HMONG_VIETNAM_IMAGE_SOURCES.md) - Data provenance
- [Collection Guide](OPENSOURCE_DATA_LINKS.md) - How to add more data

## 🤝 Contributing

To add more patterns:
1. Place images in `dataset/to_annotate/`
2. Run `python batch_annotate.py` (or manual with `annotate_images.py`)
3. Run augmentation pipeline
4. Update training splits

## ⚖️ License & Ethics

- **Dataset**: For academic research only
- **Cultural respect**: Annotations reviewed for cultural accuracy
- **Attribution**: All sources tracked in metadata
- **Community**: Results will be shared with Hmong community

## 🎯 Project Goal

Build a culturally-constrained generative AI model that:
- Preserves traditional Hmong design principles
- Assists artisans (not replaces them)
- Helps preserve cultural heritage through digitization
- Enables controlled innovation within cultural bounds

## 👥 Credits

- **Annotation**: AI-assisted batch processing
- **Cultural Context**: Based on ethnographic research
- **Tools**: Python, PIL, Stable Diffusion (planned)

## 📞 Contact

For questions or collaboration:
- GitHub: https://github.com/ThanhTrunggDEV/GenAI

---

**Status**: ✅ Demo Ready  
**Version**: 1.0  
**Created**: 2026-01-21

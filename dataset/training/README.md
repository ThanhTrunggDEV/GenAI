# Hmong Pattern Training Dataset

## Dataset Statistics

- **Total Images**: 275
- **Train**: 193  
- **Validation**: 41 
- **Test**: 41

## Structure

```
training/
├── train/
│   ├── images/      # 193 training images
│   └── captions.txt # Image-caption pairs
├── val/
│   ├── images/      # 41 validation images
│   └── captions.txt
└── test/
    ├── images/      # 41 test images
    └── captions.txt
```

## Caption Format

Each line: `<filename>TAB<caption>`

Example captions:
- Hmong traditional pattern, in black and beige, traditional style, symbolizing Hmong traditional pattern
- Hmong traditional pattern, in black and beige, traditional style, symbolizing Hmong traditional pattern
- Hmong traditional pattern, in black and beige, traditional style, symbolizing Hmong traditional pattern

## Training

Use these for Stable Diffusion fine-tuning with LoRA/DreamBooth.

**Created**: 2026-01-21

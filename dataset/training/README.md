# Hmong Pattern Training Dataset

## Dataset Statistics

- **Total Images**: 225
- **Train**: 157  
- **Validation**: 33
- **Test**: 35

## Structure

```
training/
├── train/
│   ├── images/      # 157 training images
│   └── captions.txt # Image-caption pairs
├── val/
│   ├── images/      # 33 validation images
│   └── captions.txt
└── test/
    ├── images/      # 35 test images
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

# Hmong Pattern Generation - Deployment Guide

## ⚠️ Important: GPU Requirements

This system requires **GPU with 12GB+ VRAM** for training.

Recommended platforms:
- **Google Colab Pro** ($10/month, A100 access)
- **Kaggle Notebooks** (Free, P100/T4 GPUs, 30hrs/week)
- **Lambda Labs** ($0.50-$1.10/hour for A100)
- **RunPod** (Similar pricing to Lambda)

## 🚀 Quick Start (Cloud Deployment)

### Option 1: Google Colab

1. **Upload to Google Drive**:
   ```
   Upload entire GenAI folder to Google Drive
   ```

2. **Create Colab Notebook**:
   ```python
   # Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Navigate
   %cd /content/drive/MyDrive/GenAI
   
   # Install dependencies
   !pip install -q -r requirements.txt
   
   # Run Stage 1: Extract embeddings
   !python models/visual_encoder.py
   !python models/cultural_encoder.py
   !python models/combine_embeddings.py
   
   # Run Stage 2: Train model
   !python train_diffusion.py
   ```

3. **Training time**: ~8-12 hours on A100

### Option 2: Kaggle Notebooks

1. **Create Dataset**:
   - Upload GenAI folder as Kaggle Dataset
   - Enable GPU (P100 or T4)

2. **Create Notebook**:
   ```python
   # Install deps
   !pip install -q -r /kaggle/input/genai-hmong/requirements.txt
   
   # Copy data to workspace
   !cp -r /kaggle/input/genai-hmong/* ./
   
   # Run training
   !python train_diffusion.py
   ```

## 📦 Local Setup (if you have GPU)

```bash
# Create environment
conda create -n hmong-gen python=3.10
conda activate hmong-gen

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Run pipeline
python run_full_pipeline.py
```

## 🔧 Components Status

### ✅ Ready to Use (CPU)
- Data augmentation (`augment_dataset.py`)
- Training split preparation (`prepare_training.py`)
- Demo viewer (`create_demo.py`)
- Annotation tools

### ⚠️ Requires GPU
- Stage 1: Visual/Cultural encoders (1-2 hours)
- Stage 2: Stable Diffusion training (8-12 hours)
- Stage 3: Generation & validation

## 📊 Expected Results

After training completes:
- **Model checkpoint**: `outputs/hmong-pattern-lora/checkpoint-5000`
- **Generated samples**: `outputs/samples/`
- **Metrics**: FID, KID, cultural consistency scores
- **Training logs**: `logs/tensorboard/`

## 🎯 Next Steps After This Session

Since training requires GPU not available in this chat:

1. **Download repository** from GitHub
2. **Choose cloud platform** (Colab/Kaggle)
3. **Upload data** to chosen platform
4. **Run training** using provided scripts
5. **Generate patterns** with trained model
6. **Evaluate** using metrics scripts

## 📝 Training Commands Reference

```bash
# Full pipeline (automated)
python run_full_pipeline.py

# Or step by step:

# Stage 1: Embeddings
python models/visual_encoder.py
python models/cultural_encoder.py
python models/combine_embeddings.py

# Stage 2: Training
python train_diffusion.py --config config.yaml

# Stage 3: Generation
python generate_patterns.py --checkpoint outputs/hmong-pattern-lora/checkpoint-5000

# Evaluation
python evaluate.py --real dataset/test --generated outputs/samples
```

## ⏱️ Time Estimates

| Task | Time (A100) | Time (T4) |
|------|-------------|-----------|
| Stage 1 Embeddings | 10 min | 30 min |
| Stage 2 Training | 8 hours | 20 hours |
| Stage 3 Generation | 5 min | 15 min |
| Evaluation | 10 min | 20 min |

## 💾 Checkpoints

Model will be saved at:
- Every 500 steps: Quick checkpoint
- Every 1000 steps: Full checkpoint
- Best validation: Based on FID score

## 🐛 Troubleshooting

**Out of memory**:
- Reduce `train_batch_size` to 2 or 1
- Enable `gradient_checkpointing`
- Use `mixed_precision: "fp16"`

**Slow training**:
- Use A100 instead of T4/P100
- Increase `gradient_accumulation_steps`
- Reduce `resolution` to 256

**Poor quality**:
- Train longer (10K+ steps)
- Adjust loss weights in config.yaml
- Collect more training data

## 📧 Support

For issues or questions:
- Check implementation_plan.md
- Review logs in `logs/` directory
- GitHub issues (if repository is public)

---

**Status**: All code ready, GPU training pending
**Last Updated**: 2026-01-21

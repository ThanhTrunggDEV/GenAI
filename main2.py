import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt


# ---------------------------
# 1. Load pretrained model
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-2])  # feature map
model.eval().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---------------------------
# 2. Preprocessing
# ---------------------------
def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------------------------
# 3. Candidate Region Mining - Optimized with batch processing
# ---------------------------
def extract_candidates(img, patch_size=96, stride=48):
    """Extract patches more efficiently using vectorized operations"""
    h, w, _ = img.shape
    patches = []
    coords = []
    
    # Pre-calculate all coordinates
    y_coords = list(range(0, h - patch_size, stride))
    x_coords = list(range(0, w - patch_size, stride))
    
    # Extract patches
    for y in y_coords:
        for x in x_coords:
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
                coords.append((x, y))
    
    return patches, coords

# ---------------------------
# 4. Feature extraction - Batch processing
# ---------------------------
def get_features_batch(patches, batch_size=8):
    """Process multiple patches at once for GPU efficiency"""
    features = []
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            batch_tensors = torch.stack([transform(p) for p in batch]).to(device)
            batch_feats = model(batch_tensors)
            features.extend([f.cpu().numpy() for f in batch_feats])
    
    return features

# ---------------------------
# 5. Motif Verification (repetition) - Optimized with early stopping
# ---------------------------
def repetition_score(patch, gray_full, threshold=0.8):
    """Fast repetition scoring with optimized template matching"""
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Use normalized correlation for better performance
    result = cv2.matchTemplate(gray_full, gray_patch, cv2.TM_CCOEFF_NORMED)
    
    # Count peaks above threshold
    peaks = np.sum(result > threshold)
    
    # Also consider max correlation as quality indicator
    max_corr = np.max(result)
    
    # Combined score: repetition count weighted by quality
    return peaks * max_corr

# ---------------------------
# 6. Normalize motif
# ---------------------------
def normalize_motif(patch, size=128):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 10:
        return cv2.resize(patch, (size, size))

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    center = (patch.shape[1] // 2, patch.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))

    return cv2.resize(rotated, (size, size))

# ---------------------------
# 7. Full pipeline - Optimized
# ---------------------------
def level2_pipeline(image_path):
    print(f"Running pipeline on: {device}")
    
    img = cv2.imread(image_path)
    img = preprocess(img)
    
    # Convert to grayscale once for all template matching
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract candidates
    patches, coords = extract_candidates(img)
    print(f"Total candidates: {len(patches)}")
    
    if len(patches) == 0:
        print("❌ No candidates found")
        return None
    
    # Score all patches
    scores = []
    for i, patch in enumerate(patches):
        score = repetition_score(patch, gray_full)
        scores.append(score)
    
    scores = np.array(scores)
    print(f"Score range: {scores.min():.2f} to {scores.max():.2f}")

    # Select best patch
    best_idx = np.argmax(scores)
    best_patch = patches[best_idx]
    print(f"✓ Best score: {scores[best_idx]:.2f} at position {coords[best_idx]}")

    motif = normalize_motif(best_patch)

    return motif


def estimate_period_fft(gray, axis=1):
    """
    Sử dụng FFT để tìm chu kỳ lặp lại
    axis=1: tìm chu kỳ theo trục X
    axis=0: theo trục Y
    FFT nhanh hơn và chính xác hơn autocorrelation
    """
    signal = gray.mean(axis=axis)
    
    # Chuẩn hóa tín hiệu
    signal = signal - signal.mean()
    
    # Tính FFT
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    
    # Bỏ DC component (tần số 0)
    power[0] = 0
    
    # Chỉ lấy nửa đầu (do tính đối xứng của FFT thực)
    half_len = len(power) // 2
    power = power[1:half_len]
    
    # Tìm tần số dominant
    if len(power) == 0:
        return 0
    
    peak_freq_idx = np.argmax(power) + 1  # +1 vì đã bỏ phần tử 0
    
    # Chuyển từ tần số sang chu kỳ (period)
    period = len(signal) // peak_freq_idx if peak_freq_idx > 0 else 0
    
    return period

def extract_unit_fft(block_img):
    """Trích xuất đơn vị lặp lại sử dụng FFT"""
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

    px = estimate_period_fft(gray, axis=1)
    py = estimate_period_fft(gray, axis=0)

    h, w = gray.shape

    # Fallback nếu FFT không tìm được chu kỳ hợp lý
    px = px if 10 < px < w else w // 3
    py = py if 10 < py < h else h // 3

    # Crop đơn vị từ góc trên trái
    unit = block_img[0:py, 0:px]
    unit = cv2.resize(unit, (128, 128))

    return unit


# ---------------------------
# 8. Run
# ---------------------------
motif = level2_pipeline("hmong.jpg")

if motif is not None:
    cv2.imwrite("extracted_motif_level2.png", motif)
    
    # Extract unit pattern from the motif using FFT
    unit = extract_unit_fft(motif)
    cv2.imwrite("motif_unit_fft.png", unit)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(cv2.cvtColor(motif, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Extracted Motif Block")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(unit, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Unit Pattern (FFT)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Saved: extracted_motif_level2.png")
    print(f"✓ Saved: motif_unit_fft.png")

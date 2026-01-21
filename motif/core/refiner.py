import os

# Cấu hình môi trường (phải đặt trước khi import torch/cv2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt

class MotifRefiner:
    def __init__(self, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self._init_model()
        self._init_transforms()
        print(f"MotifRefiner initialized on {self.device}")

    def _init_model(self):
        """Khởi tạo và load model ResNet50"""
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # Sử dụng phần feature extractor (bỏ fully connected layers)
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.model.eval().to(self.device)

    def _init_transforms(self):
        """Khởi tạo các transform cần thiết"""
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img_path):
        """Đọc và tiền xử lý ảnh đầu vào (CLAHE)"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def extract_candidates(self, img, patch_size=96, stride=48):
        """Trích xuất các vùng ứng viên (candidates) từ ảnh"""
        h, w, _ = img.shape
        patches = []
        coords = []
        
        y_coords = range(0, h - patch_size, stride)
        x_coords = range(0, w - patch_size, stride)
        
        for y in y_coords:
            for x in x_coords:
                patch = img[y:y+patch_size, x:x+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)
                    coords.append((x, y))
        
        return patches, coords

    def calculate_repetition_score(self, patch, gray_full, threshold=0.8):
        """Tính điểm lặp lại (repetition score) của patch trên toàn bộ ảnh"""
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Template Matching để tìm các vùng giống nhau
        result = cv2.matchTemplate(gray_full, gray_patch, cv2.TM_CCOEFF_NORMED)
        
        peaks = np.sum(result > threshold)
        max_corr = np.max(result)
        
        # Điểm số kết hợp giữa số lượng đỉnh và độ tương quan cao nhất
        return peaks * max_corr

    def normalize_motif_orientation(self, patch, size=128):
        """Chuẩn hóa hướng của motif (xoay thẳng)"""
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

    def run_pipeline(self, image_path):
        """Chạy toàn bộ pipeline để tìm motif tốt nhất"""
        print(f"Processing: {image_path}")
        
        img = self.preprocess_image(image_path)
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        patches, coords = self.extract_candidates(img)
        print(f"Total candidates: {len(patches)}")
        
        if not patches:
            print("❌ No candidates found")
            return None
        
        # Tính điểm cho từng patch
        scores = [self.calculate_repetition_score(p, gray_full) for p in patches]
        scores = np.array(scores)
        
        best_idx = np.argmax(scores)
        best_patch = patches[best_idx]
        print(f"✓ Best score: {scores[best_idx]:.2f} at {coords[best_idx]}")

        return self.normalize_motif_orientation(best_patch)


class PatternAnalyzer:
    @staticmethod
    def estimate_period_fft(gray, axis=1):
        """Ước lượng chu kỳ lặp lại bằng FFT"""
        signal = gray.mean(axis=axis)
        signal = signal - signal.mean()
        
        fft = np.fft.fft(signal)
        power = np.abs(fft) ** 2
        power[0] = 0 # Bỏ DC component
        
        half_len = len(power) // 2
        power = power[1:half_len]
        
        if len(power) == 0:
            return 0
        
        peak_freq_idx = np.argmax(power) + 1
        period = len(signal) // peak_freq_idx if peak_freq_idx > 0 else 0
        
        return period

    @classmethod
    def extract_unit_pattern(cls, block_img):
        """Trích xuất đơn vị lặp lại nhỏ nhất (unit pattern)"""
        gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

        px = cls.estimate_period_fft(gray, axis=1)
        py = cls.estimate_period_fft(gray, axis=0)

        h, w = gray.shape
        # Fallback an toàn
        px = px if 10 < px < w else w // 3
        py = py if 10 < py < h else h // 3

        unit = block_img[0:py, 0:px]
        return cv2.resize(unit, (128, 128))


def save_and_visualize(motif, unit, output_dir, file_pk):
    """Lưu và hiển thị kết quả"""
    motif_path = os.path.join(output_dir, f"level2_motif_{file_pk}.png")
    unit_path = os.path.join(output_dir, f"level2_unit_{file_pk}.png")
    
    cv2.imwrite(motif_path, motif)
    cv2.imwrite(unit_path, unit)
    print(f"✓ Saved results to {output_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(cv2.cvtColor(motif, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Extracted Motif Block")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(unit, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Unit Pattern (FFT)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    input_dir = "dataset/to_annotate"
    output_dir = "outputs/refined"
    os.makedirs(output_dir, exist_ok=True)
    
    img_name = "hoa-van-trang-tri-hmong-2.jpg"
    img_path = os.path.join(input_dir, img_name)
    
    try:
        # 1. Pipeline trích xuất motif
        refiner = MotifRefiner()
        motif = refiner.run_pipeline(img_path)

        if motif is not None:
            # 2. Phân tích pattern
            unit = PatternAnalyzer.extract_unit_pattern(motif)
            
            # 3. Lưu và hiển thị
            save_and_visualize(motif, unit, output_dir, img_name.split('.')[0])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

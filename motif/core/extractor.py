import os

# Cấu hình môi trường (phải đặt trước khi import torch/cv2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

class MotifExtractor:
    def __init__(self, use_gpu=True):
        """
        Khởi tạo extractor, load model ResNet50 pretrained.
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Initializing MotifExtractor on device: {self.device}")
        
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        """
        Load ResNet50 backbone, bỏ lớp Fully Connected cuối cùng.
        """
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        # Lấy phần encoder: bỏ 2 lớp cuối (AvgPool và FC)
        encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
        encoder.to(self.device)
        encoder.eval()
        return encoder

    def preprocess(self, img_path, size=512):
        """
        Đọc và tiền xử lý ảnh đầu vào.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        
        # Chuyển sang Tensor và thêm batch dimension (1, C, H, W)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img, tensor

    def _get_feature_patches(self, feature_map, patch_size=4):
        """
        Chia feature map thành các patch nhỏ và vector hóa chúng.
        """
        # Average pooling để giảm kích thước và lấy đặc trưng đại diện cho vùng
        pooled = F.avg_pool2d(feature_map, kernel_size=patch_size, stride=patch_size)
        
        b, c, h, w = pooled.shape
        # Flatten: (Batch, C, H, W) -> (Batch, H, W, C) -> (N_patches, CVector)
        patches = pooled.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Tạo danh sách tọa độ (x, y) trên feature map gốc tương ứng với mỗi patch
        coords = [(x * patch_size, y * patch_size) for y in range(h) for x in range(w)]
        
        return patches, coords

    def _compute_scores(self, patches):
        """
        Tính toán điểm số dựa trên Similarity và Texture Score.
        """
        # 1. Cosine Similarity Score
        # Normalize vector về độ dài 1
        patches_norm = F.normalize(patches, p=2, dim=1)
        # Tính ma trận tương đồng (N x N)
        sim_matrix = torch.mm(patches_norm, patches_norm.t())
        sim_score = sim_matrix.mean(dim=1)

        # 2. Texture Score (Variance)
        texture_scores = torch.var(patches, dim=1)
        
        # Min-Max Scaling cho texture score
        t_min, t_max = texture_scores.min(), texture_scores.max()
        texture_scores = (texture_scores - t_min) / (t_max - t_min + 1e-6)

        # 3. Kết hợp (Element-wise multiplication)
        final_score = sim_score * texture_scores
        
        return final_score

    def find_dominant_motif(self, img_path, patch_size=4):
        """
        Hàm chính: Tìm vị trí của motif chủ đạo trong ảnh.
        Output: original_image, center_coordinate, scale_factor
        """
        original_img, tensor = self.preprocess(img_path)
        
        with torch.no_grad():
            features = self.model(tensor)
            
        patches, coords = self._get_feature_patches(features, patch_size)
        scores = self._compute_scores(patches)
        
        # Tìm patch có điểm số cao nhất
        best_idx = torch.argmax(scores).item()
        dominant_coord = coords[best_idx]
        
        # Tính tỷ lệ scale giữa ảnh gốc và feature map để map tọa độ ngược lại
        scale = original_img.shape[0] // features.shape[-1]
        
        return original_img, dominant_coord, scale

    @staticmethod
    def crop_motif(image, coord, scale, patch_size=4, block_ratio=3):
        """
        Cắt vùng motif từ ảnh gốc dựa trên tọa độ tìm được.
        """
        x, y = coord
        h, w, _ = image.shape
        
        # Tính kích thước vùng cắt
        tile_size = patch_size * scale
        crop_radius = (block_ratio * tile_size) // 2
        
        # Tính tâm vùng cắt
        cx = int((x + patch_size / 2) * scale)
        cy = int((y + patch_size / 2) * scale)

        # Giới hạn vùng cắt trong khung hình
        y1 = max(0, cy - crop_radius)
        y2 = min(h, cy + crop_radius)
        x1 = max(0, cx - crop_radius)
        x2 = min(w, cx + crop_radius)

        return image[y1:y2, x1:x2]

    @staticmethod
    def refine_motif(motif_img):
        """
        Tinh chỉnh lại vị trí motif để căn giữa (sử dụng projection profile).
        """
        if motif_img.size == 0:
            return motif_img
            
        gray = cv2.cvtColor(motif_img, cv2.COLOR_RGB2GRAY)
        
        # Tính trung bình cường độ sáng theo trục x và y
        proj_x = gray.mean(axis=0)
        proj_y = gray.mean(axis=1)

        # Tìm vị trí sáng nhất (giả định là tâm của motif)
        x_center = np.argmax(proj_x)
        y_center = np.argmax(proj_y)

        h, w = gray.shape
        size = min(h, w) // 2

        y1 = max(0, y_center - size)
        y2 = min(h, y_center + size)
        x1 = max(0, x_center - size)
        x2 = min(w, x_center + size)

        # Trả về vùng cắt đã căn chỉnh
        return motif_img[y1:y2, x1:x2]


def visualize_result(original, motif, save_path=None):
    """
    Hiển thị kết quả so sánh và lưu ảnh nếu có đường dẫn.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Extracted Motif")
    if motif.size > 0:
        plt.imshow(motif)
    else:
        plt.text(0.5, 0.5, "Empty Result", ha='center')
    plt.axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Result saved to: {save_path}")
        
    plt.show()


def main():
    # Đường dẫn ảnh đầu vào
    input_dir = "dataset/to_annotate"
    output_dir = "outputs/extracted"
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Use first available image if hmong.jpg doesn't exist
    img_name = "hoa-van-trang-tri-hmong-2.jpg"
    img_path = os.path.join(input_dir, img_name)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' does not exist.")
        return

    try:
        # 1. Khởi tạo
        extractor = MotifExtractor()
        
        # 2. Tìm kiếm motif
        original_img, coord, scale = extractor.find_dominant_motif(img_path)
        print(f"Motif found at feature map coord: {coord} with scale: {scale}")
        
        # 3. Cắt và tinh chỉnh
        motif_raw = extractor.crop_motif(original_img, coord, scale)
        motif_refined = extractor.refine_motif(motif_raw)
        
        # 4. Hiển thị và lưu kết quả
        save_path = os.path.join(output_dir, f"result_{img_name}")
        visualize_result(original_img, motif_refined, save_path=save_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
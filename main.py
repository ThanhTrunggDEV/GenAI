import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms



def preprocess_image(img_path, size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img).unsqueeze(0)
    return img, tensor



class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use weights instead of pretrained=True to avoid deprecation warnings
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.encoder(x)


def extract_patches(feature_map, patch_size=4):
    """
    Trích xuất feature vectors cho các patch sử dụng Average Pooling thay vì vòng lặp.
    Giữ dữ liệu trên GPU để tính toán nhanh hơn.
    """
    # feature_map: (1, C, H, W)
    # Average pooling tương đương với việc lấy mean trên từng cửa sổ patch_size
    # stride=patch_size giúp các patch không bị chồng lấn (tương tự vòng lặp cũ)
    pooled = F.avg_pool2d(feature_map, kernel_size=patch_size, stride=patch_size)
    # pooled shape: (1, C, H_new, W_new)
    
    b, c, h, w = pooled.shape
    
    # Flatten về (N_patches, C) để tính toán similarity
    # permute(0, 2, 3, 1) -> (1, H_new, W_new, C) -> reshape -> (H*W, C)
    patches = pooled.permute(0, 2, 3, 1).reshape(-1, c)
    
    # Tạo danh sách tọa độ tương ứng
    coords = []
    for y in range(h):
        for x in range(w):
            coords.append((x * patch_size, y * patch_size))
            
    return patches, coords


def compute_similarity_with_texture(patches):
    """
    Tính toán similarity và texture score hoàn toàn trên Torch Tensor (GPU/CPU).
    Nhanh hơn so với việc dùng sklearn và numpy list comprehension.
    """
    # 1. Cosine Similarity
    # Normalize các vector về độ dài 1 (L2 norm)
    patches_norm = F.normalize(patches, p=2, dim=1)
    
    # Tính ma trận similarity: A @ A.T (N x N)
    sim_matrix = torch.mm(patches_norm, patches_norm.t())
    
    # Lấy trung bình độ tương đồng của mỗi patch với các patch khác
    sim_score = sim_matrix.mean(dim=1)

    # 2. Texture Score (Variance trong từng feature vector)
    texture_scores = torch.var(patches, dim=1)
    
    # Min-Max Scaling cho texture_scores để đưa về [0, 1]
    t_min = texture_scores.min()
    t_max = texture_scores.max()
    texture_scores = (texture_scores - t_min) / (t_max - t_min + 1e-6)

    # 3. Kết hợp
    final_score = sim_score * texture_scores
    
    # Chuyển về numpy để xử lý bước tiếp theo (argmax)
    return final_score.cpu().numpy()


def find_dominant_patch(scores, coords):
    idx = np.argmax(scores)
    return coords[idx]


def crop_motif_block(original_img, coord, scale, patch_size=4, block=3):
    x, y = coord
    h, w, _ = original_img.shape

    tile = patch_size * scale
    size = block * tile // 2

    cx = int((x + patch_size / 2) * scale)
    cy = int((y + patch_size / 2) * scale)

    crop = original_img[
        max(0, cy-size):min(h, cy+size),
        max(0, cx-size):min(w, cx+size)
    ]
    return crop


def extract_motif(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running pipeline on: {device}")

    img, tensor = preprocess_image(img_path)
    tensor = tensor.to(device)

    model = ResNetEncoder().to(device)
    model.eval()

    with torch.no_grad():
        features = model(tensor)

    # Trích xuất patch (vẫn giữ trên GPU)
    patches, coords = extract_patches(features, patch_size=4)

    # Tính toán điểm số (vẫn giữ trên GPU cho đến bước cuối)
    scores = compute_similarity_with_texture(patches)
    
    dominant_coord = coords[np.argmax(scores)]

    scale = img.shape[0] // features.shape[-1]
    motif = crop_motif_block(img, dominant_coord, scale)

    return img, motif

def refine_motif_by_projection(motif):
    gray = cv2.cvtColor(motif, cv2.COLOR_RGB2GRAY)
    proj_x = gray.mean(axis=0)
    proj_y = gray.mean(axis=1)

    x_center = np.argmax(proj_x)
    y_center = np.argmax(proj_y)

    h, w = gray.shape
    size = min(h, w) // 2

    refined = motif[
        max(0, y_center-size):min(h, y_center+size),
        max(0, x_center-size):min(w, x_center+size)
    ]
    return refined





def main():
    img_path = "hmong.jpg"  

    original, motif = extract_motif(img_path)
    motif = refine_motif_by_projection(motif)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Extracted Motif")
    plt.imshow(motif)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()

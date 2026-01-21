# HƯỚNG DẪN GÁN NHÃN DỮ LIỆU HMONG

## 🎯 MỤC ĐÍCH

Gán nhãn chi tiết cho dataset họa tiết Hmong để huấn luyện mô hình AI có ràng buộc văn hóa.

---

## 📋 CẤU TRÚC NHÃN (Annotation Schema)

### Template JSON cho mỗi ảnh:

```json
{
  "image_id": "hmong_001",
  "filename": "hmong_001.jpg",
  "source": "wikimedia",
  "source_url": "https://commons.wikimedia.org/...",
  
  "location": {
    "province": "Lào Cai / Yên Bái / Hà Giang",
    "district": "Sa Pa / Mù Cang Chải / Quản Bạ",
    "village": "Chế Cu Nha / San Sả Hồ / Lùng Tám",
    "region": "Northwest Vietnam"
  },
  
  "ethnic_info": {
    "subgroup": "Black Hmong / Flower Hmong / White Hmong / Green Hmong",
    "local_name": "Hmong Đen / Hmong Hoa / Hmong Trắng"
  },
  
  "pattern_info": {
    "motif_type": ["geometric", "floral", "animal"],
    "specific_motifs": ["snail", "zigzag", "chicken_foot", "dragon"],
    "dominant_motif": "geometric"
  },
  
  "color_info": {
    "colors": ["indigo", "red", "black", "yellow", "white"],
    "dominant_color": "indigo",
    "color_scheme": "traditional / modern"
  },
  
  "cultural_meaning": {
    "symbolism": "mountains / fertility / protection / prosperity",
    "ritual_use": "daily wear / festival / funeral / wedding",
    "significance": "High / Medium / Low"
  },
  
  "technique": {
    "primary_technique": "batik / embroidery / applique / patchwork",
    "tools_used": ["beeswax pen", "indigo dye", "needle"],
    "material": "linen / cotton / hemp"
  },
  
  "visual_structure": {
    "symmetry": "rotational / bilateral / asymmetric",
    "repetition": "grid / linear / scattered / none",
    "complexity": "high / medium / low"
  },
  
  "quality": {
    "resolution": "1024x1024",
    "clarity": "high / medium / low",
    "completeness": "full pattern / partial / fragment",
    "condition": "new / worn / restored"
  },
  
  "date_info": {
    "photo_date": "2024-01-15",
    "estimated_creation": "traditional / modern / contemporary",
    "annotation_date": "2026-01-21",
    "annotator": "researcher_name"
  },
  
  "notes": "Additional observations or context"
}
```

---

## 🏷️ CHI TIẾT CÁC TRƯỜNG

### 1. MOTIF_TYPE (Loại họa tiết)

#### Geometric (Hình học)
- `zigzag` - Đường zigzag (núi non)
- `triangle` - Tam giác
- `diamond` - Hình thoi
- `spiral` - Xoắn ốc
- `maze` - Mê cung
- `grid` - Lưới ô vuông

#### Organic (Tự nhiên)
- `snail` (ốc sên) - Motif rất phổ biến
- `dragon` (rồng) - Biểu tượng quyền lực
- `bird` (chim) - Tự do, linh hồn
- `flower` (hoa) - Sinh sôi nảy nở
- `butterfly` (bướm) - Chuyển hóa

#### Technical (Kỹ thuật)
- `chicken_foot` (chân gà) - Pattern đặc trưng
- `pig_foot` (chân lợn) - Pattern thêu
- `hemp_tool` (khung lanh) - Công cụ
- `pumpkin_flower` (hoa bí) - Hoa văn phổ biến

### 2. COLORS (Màu sắc)

#### Màu truyền thống:
- `indigo` (chàm) - Màu chủ đạo của Black Hmong
- `black` (đen) - Nền, truyền thống
- `red` (đỏ) - May mắn, lễ hội
- `yellow` (vàng) - Giàu có, vương giả
- `white` (trắng) - Tinh khiết, tang lễ
- `green` (xanh lá) - Thiên nhiên
- `blue` (xanh dương) - Bầu trời

#### Color scheme:
- `traditional` - Chủ yếu indigo, đen, đỏ
- `modern` - Nhiều màu sắc hơn
- `festive` - Rực rỡ (đỏ, vàng)
- `mourning` - Tang lễ (trắng, đen)

### 3. TECHNIQUE (Kỹ thuật)

- `batik` - Vẽ sáp ong nhuộm chàm
- `embroidery` - Thêu tay
- `applique` - Đắp vải (thêu đắp)
- `patchwork` - Ghép vải
- `cross_stitch` - Thêu chữ thập
- `resist_dyeing` - Nhuộm cản

### 4. SYMMETRY (Đối xứng)

- `rotational` - Đối xứng xoay (4-fold, 8-fold)
- `bilateral` - Đối xứng gương (trái-phải)
- `radial` - Đối xứng tâm
- `asymmetric` - Không đối xứng

### 5. CULTURAL_MEANING (Ý nghĩa văn hóa)

#### Symbolism:
- `mountains` - Núi non (zigzag patterns)
- `fertility` - Sinh sôi (hoa, chim)
- `protection` - Bảo vệ (rồng, mê cung)
- `prosperity` - Thịnh vượng (hoa mai, vàng)
- `ancestors` - Tổ tiên (patterns cổ)
- `nature` - Thiên nhiên (động thực vật)

#### Ritual use:
- `daily_wear` - Hàng ngày
- `festival` - Lễ hội (Tết, cưới)
- `funeral` - Tang lễ (váy hoa bắt buộc)
- `wedding` - Đám cưới
- `ceremonial` - Nghi lễ

---

## 🛠️ QUY TRÌNH GÁN NHÃN

### Bước 1: Quan sát tổng thể
1. Xem toàn bộ ảnh
2. Xác định nguồn gốc nếu có thông tin
3. Đánh giá quality (độ nét, độ rõ)

### Bước 2: Phân tích pattern
1. Xác định motif chính (dominant motif)
2. Liệt kê các motif phụ
3. Đánh giá độ phức tạp

### Bước 3: Phân tích màu sắc
1. Đếm số màu có trong pattern
2. Xác định màu chủ đạo
3. Phân loại color scheme

### Bước 4: Xác định kỹ thuật
1. Nhận diện technique (batik/embroidery/applique)
2. Có thể kết hợp nhiều technique

### Bước 5: Phân tích cấu trúc
1. Kiểm tra symmetry
2. Xác định repetition pattern
3. Đo lường complexity

### Bước 6: Context văn hóa
1. Tìm kiếm thông tin về location
2. Xác định subgroup (Black/Flower/White Hmong)
3. Ghi nhận cultural meaning nếu có

### Bước 7: Quality check
1. Kiểm tra lại tất cả fields
2. Đảm bảo consistency
3. Thêm notes nếu cần

---

## 📝 MẪU GÁN NHÃN CỤ THỂ

### Ví dụ 1: Black Hmong Batik (Sapa)

```json
{
  "image_id": "hmong_sapa_001",
  "filename": "black_hmong_batik_sapa.jpg",
  "source": "wikimedia",
  
  "location": {
    "province": "Lào Cai",
    "district": "Sa Pa",
    "village": "Cat Cat",
    "region": "Northwest Vietnam"
  },
  
  "ethnic_info": {
    "subgroup": "Black Hmong",
    "local_name": "Hmong Đen"
  },
  
  "pattern_info": {
    "motif_type": ["geometric"],
    "specific_motifs": ["spiral", "zigzag", "maze"],
    "dominant_motif": "spiral"
  },
  
  "color_info": {
    "colors": ["indigo", "white"],
    "dominant_color": "indigo",
    "color_scheme": "traditional"
  },
  
  "cultural_meaning": {
    "symbolism": "protection, ancestors",
    "ritual_use": "daily_wear",
    "significance": "High"
  },
  
  "technique": {
    "primary_technique": "batik",
    "tools_used": ["beeswax pen", "indigo dye"],
    "material": "hemp"
  },
  
  "visual_structure": {
    "symmetry": "rotational",
    "repetition": "grid",
    "complexity": "high"
  },
  
  "quality": {
    "resolution": "1024x1024",
    "clarity": "high",
    "completeness": "full pattern",
    "condition": "traditional"
  },
  
  "notes": "Classic Black Hmong batik with fine spiral patterns. Indigo dye appears rich and authentic."
}
```

### Ví dụ 2: White Hmong Applique (Hà Giang)

```json
{
  "image_id": "hmong_hagiang_002",
  "filename": "white_hmong_collar.jpg",
  "source": "craftlink",
  
  "location": {
    "province": "Hà Giang",
    "district": "Quản Bạ",
    "village": "Lùng Tám",
    "region": "Northeast Vietnam"
  },
  
  "ethnic_info": {
    "subgroup": "White Hmong",
    "local_name": "Hmong Trắng"
  },
  
  "pattern_info": {
    "motif_type": ["geometric", "organic"],
    "specific_motifs": ["snail", "triangle"],
    "dominant_motif": "snail"
  },
  
  "color_info": {
    "colors": ["white", "red", "black"],
    "dominant_color": "black",
    "color_scheme": "traditional"
  },
  
  "cultural_meaning": {
    "symbolism": "fertility, continuity",
    "ritual_use": "festival",
    "significance": "High"
  },
  
  "technique": {
    "primary_technique": "applique",
    "tools_used": ["needle", "thread"],
    "material": "cotton"
  },
  
  "visual_structure": {
    "symmetry": "bilateral",
    "repetition": "linear",
    "complexity": "medium"
  },
  
  "quality": {
    "resolution": "512x512",
    "clarity": "medium",
    "completeness": "partial",
    "condition": "new"
  },
  
  "notes": "White Hmong collar with characteristic snail motif (ốc sên). Applique technique with white/red on black background."
}
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### Consistency (Nhất quán)
- Dùng cùng một spelling cho terms
- Lower case cho tất cả values
- Dấu gạch chân thay vì space trong keys

### Accuracy (Chính xác)
- Chỉ ghi thông tin đã biết chắc chắn
- Dùng "unknown" nếu không rõ
- Thêm context vào "notes" nếu cần

### Cultural Sensitivity (Nhạy cảm văn hóa)
- Tôn trọng ý nghĩa văn hóa
- Không phỏng đoán ý nghĩa nếu không chắc
- Tham khảo người Hmong nếu có thể

### Validation
- Cross-check ít nhất 20% dataset
- Có người thứ 2 review
- Đảm bảo không thiếu fields bắt buộc

---

## 🚀 CÔNG CỤ GÁN NHÃN

Sẽ tạo:
1. **Web app** - UI thân thiện để gán nhãn nhanh
2. **Python script** - Batch annotation với template
3. **Validation script** - Kiểm tra consistency

---

**Version**: 1.0
**Last Updated**: 2026-01-21
**Contact**: [Your email for questions]

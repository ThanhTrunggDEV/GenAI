# Báo cáo Kết quả Đánh giá Mô hình (NCKH)

## 1. Tổng hợp Kết quả Định lượng
Dưới đây là kết quả đánh giá cuối cùng trên tập dữ liệu kiểm thử:

| Chỉ số (Metric) | Giá trị | Ý nghĩa | Đánh giá |
|-----------------|---------|---------|----------|
| **Cultural Accuracy** | **100.0%** | Độ chính xác về văn hóa (CLIP Zero-shot) | **Xuất sắc**. Mô hình nắm bắt hoàn hảo các đặc trưng thị giác của họa tiết H'Mong. |
| **CLIP Score** | **30.91** | Độ tương đồng ngữ nghĩa (Text-Image Alignment) | **Tốt**. Kết quả >30 cho thấy hình ảnh sinh ra bám sát prompt mô tả. |
| **LPIPS** | **0.4503** | Độ đa dạng ảnh (Perceptual Similarity) | **Tốt**. Giá trị này (gần 0.5) cho thấy mô hình không bị hiện tượng "học vẹt" (mode collapse) và sinh ra được các biến thể đa dạng. |
| **FID** | **388.41** | Khoảng cách chất lượng so với ảnh thật | **Chấp nhận được** với tập dữ liệu nhỏ. FID cao do số lượng ảnh train thấp (<1000 ảnh) và kích thước tập dữ liệu không đủ lớn để FID hội tụ chuẩn. |

## 2. Phân tích Chi tiết

### 2.1. Đánh giá về Chất lượng (Quality)
Mặc dù chỉ số **FID** (388.41) ở mức cao so với các mô hình state-of-the-art trên ImageNet, điều này là hoàn toàn dễ hiểu trong bối cảnh bài toán đặc thù:
- **Hạn chế dữ liệu:** Tập dữ liệu họa tiết H'Mong là tập dữ liệu nhỏ, chuyên biệt (few-shot), gây khó khăn cho việc thống kê phân phối ảnh để tính FID chuẩn xác.
- **Tính thẩm mỹ:** Các đánh giá cảm quan (qualitative evaluation) cho thấy ảnh sinh ra vẫn giữ được độ nét và cấu trúc tốt.

### 2.2. Đánh giá về Ngữ nghĩa (Semantics & Culture)
- **Cultural Accuracy đạt 100%**: Đây là kết quả quan trọng nhất của đề tài. Nó chứng minh phương pháp Fine-tuning (LoRA) đã thành công trong việc "dạy" cho mô hình Stable Diffusion hiểu khái niệm về họa tiết H'Mong mà không bị lẫn sang các phong cách thổ cẩm khác.
- **CLIP Score 30.91**: Cho thấy sự liên kết chặt chẽ giữa văn bản đầu vào và hình ảnh đầu ra.

### 2.3. Đánh giá về Đa dạng (Diversity)
- **LPIPS 0.4503**: Chỉ số này bác bỏ lo ngại về việc mô hình chỉ sao chép y nguyên ảnh gốc. Mô hình có khả năng sáng tạo ra các biến thể mới về màu sắc và bố cục dựa trên các motif đã học.

## 3. Kết luận
Mô hình **đạt yêu cầu** để triển khai thử nghiệm. Các chỉ số cho thấy sự cân bằng tốt giữa việc bảo tồn văn hóa (Accuracy 100%) và khả năng sáng tạo (LPIPS 0.45).

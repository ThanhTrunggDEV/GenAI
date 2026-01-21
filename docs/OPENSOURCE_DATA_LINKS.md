# NGUỒN DỮ LIỆU MỞ - HMONG TEXTILE PATTERNS

## 🎯 MỤC TIÊU
Thu thập 200-500 ảnh từ các nguồn mở MIỄN PHÍ trước khi đi thu thập field data.

## 📦 NGUỒN 1: WIKIMEDIA COMMONS (Ưu tiên cao)

### Link trực tiếp để download thủ công:

#### Vietnam Museum of Ethnology Collection
🔗 https://commons.wikimedia.org/wiki/Category:Hmong_collection_(Vietnam_Museum_of_Ethnology)
- **Số lượng**: ~30-50 ảnh
- **Chất lượng**: Cao (museum quality)
- **License**: CC0 / CC-BY
- **Cách download**: 
  1. Click vào từng ảnh
  2. Click "More details" → "Original file"
  3. Click chuột phải → Save image

#### Hmong Textiles General
🔗 https://commons.wikimedia.org/wiki/Category:Hmong_textiles
- **Số lượng**: ~20-30 ảnh
- **License**: Mixed (kiểm tra từng ảnh)

#### Vietnamese Ethnic Clothing
🔗 https://commons.wikimedia.org/wiki/Category:Traditional_costumes_of_Vietnam
- **Số lượng**: ~50-100 ảnh (bao gồm nhiều dân tộc)

### ⚡ Tự động download với script
```bash
cd "E:\Bai bao\report fpf\GenAI"
python download_wikimedia.py
```
→ Sẽ download tự động vào folder `dataset/raw/wikimedia/`

---

## 📦 NGUỒN 2: FREEPIK (Free với attribution)

### Link search:
1. **Hmong pattern**: https://www.freepik.com/search?format=search&query=hmong%20pattern&type=vector
2. **Ethnic Vietnamese pattern**: https://www.freepik.com/search?format=search&query=vietnamese%20ethnic%20pattern
3. **Hmong textile**: https://www.freepik.com/search?format=search&query=hmong%20textile
4. **Hill tribe pattern**: https://www.freepik.com/search?format=search&query=hill%20tribe%20pattern%20vietnam

### Cách download:
1. Tạo tài khoản FREE tại https://www.freepik.com/sign-up
2. Filter: "Free" trong search results
3. Click "Download" → chọn kích thước (recommend: 1024px)
4. Attribution required: Lưu link nguồn trong metadata

**Ước tính**: 50-100 ảnh patterns free

---

## 📦 NGUỒN 3: VECTEEZY (Royalty-free)

🔗 https://www.vecteezy.com/free-vector/hmong-pattern

### Cách download:
1. Tạo tài khoản FREE
2. Search: "hmong pattern", "vietnamese ethnic", "tribal pattern"
3. Filter "Free" resources
4. Download (cần attribution)

**Ước tính**: 30-50 vectors

---

## 📦 NGUỒN 4: UNSPLASH / PEXELS (Stock photos)

### Unsplash:
🔗 Search: https://unsplash.com/s/photos/hmong-vietnam
🔗 Search: https://unsplash.com/s/photos/vietnam-ethnic-minority

### Pexels:
🔗 https://www.pexels.com/search/hmong/
🔗 https://www.pexels.com/search/vietnam%20ethnic%20costume/

**License**: CC0 (completely free, no attribution needed)
**Ước tính**: 20-40 ảnh

---

## 📦 NGUỒN 5: D-LAYERS DATASET

🔗 https://openxlab.org.cn/datasets/D-LAYERS/D-LAYERS

### Cách access:
1. Tạo account tại OpenXLab
2. Request access to dataset
3. Download subset có liên quan:
   - Vietnamese patterns
   - Southeast Asian ethnic patterns
   - Similar geometric patterns (Uyghur, Emirati có geometry giống)

**Ước tính**: 50-100 ảnh có thể sử dụng

---

## 📦 NGUỒN 6: PINTEREST (Informal reference)

⚠️ **CHÚ Ý**: Pinterest không có license rõ ràng, CHỈ dùng để tham khảo visual, KHÔNG dùng trực tiếp trong dataset training!

🔗 https://www.pinterest.com/search/pins/?q=hmong%20pattern
🔗 https://www.pinterest.com/search/pins/?q=vietnamese%20hmong%20textile

→ Dùng để:
- Tìm inspiration
- Reverse image search để tìm nguồn gốc
- Tìm museum collections khác

---

## 📦 NGUỒN 7: KAGGLE / HUGGING FACE DATASETS

### Kaggle:
Search: "textile pattern dataset", "ethnic pattern"
🔗 https://www.kaggle.com/datasets

### Hugging Face:
🔗 https://huggingface.co/datasets
Search: "textile", "pattern", "cultural heritage"

**Hiện tại**: Chưa có dataset Hmong cụ thể, nhưng có thể tìm được related patterns

---

## 🎯 KẾ HOẠCH DOWNLOAD THỨ TỰ

### TUẦN 1: Priority Sources (Nguồn có sẵn ngay)

**Ngày 1-2**: Wikimedia Commons
- [ ] Run script `download_wikimedia.py`
- [ ] Download thủ công các ảnh quality cao
- **Target**: 50-80 ảnh

**Ngày 3-4**: Freepik
- [ ] Tạo account
- [ ] Download free patterns
- [ ] Lưu attribution info
- **Target**: 50-100 ảnh

**Ngày 5**: Vecteezy
- [ ] Tạo account
- [ ] Download free vectors
- **Target**: 30-50 ảnh

**Ngày 6-7**: Unsplash + Pexels
- [ ] Search và download
- [ ] Crop ảnh để focus vào pattern regions
- **Target**: 20-40 ảnh

**🎯 Milestone tuần 1**: 150-270 ảnh từ nguồn mở

---

### TUẦN 2: Advanced Sources

**Ngày 8-9**: D-LAYERS Dataset
- [ ] Request access
- [ ] Download relevant subsets
- [ ] Filter patterns tương tự Hmong
- **Target**: 50-100 ảnh

**Ngày 10-11**: Organize & Clean
- [ ] Remove duplicates
- [ ] Quality check (resolution, clarity)
- [ ] Standardize to 512x512 or 1024x1024
- [ ] Create metadata JSON

**Ngày 12-14**: Augmentation
- [ ] Generate variations (rotation, flip)
- [ ] Color variations (within cultural bounds)
- [ ] Result: 2-3x original → **300-600 ảnh total**

**🎯 Milestone tuần 2**: 300-600 ảnh đã xử lý

---

## 📁 FOLDER STRUCTURE

```
E:\Bai bao\report fpf\GenAI\dataset\
├── raw/                          # Ảnh gốc download
│   ├── wikimedia/               # From Wikimedia Commons
│   ├── freepik/                 # From Freepik
│   ├── vecteezy/                # From Vecteezy
│   ├── stock/                   # Unsplash, Pexels
│   └── d_layers/                # D-LAYERS dataset
├── processed/                    # Đã clean và resize
│   ├── 512x512/
│   └── 1024x1024/
├── augmented/                    # Đã augment
└── metadata/                     # JSON files with sources
    └── sources.json             # Track all image sources
```

---

## ✅ CHECKLIST TRƯỚC KHI DOWNLOAD

- [ ] Cài đặt dependencies: `pip install requests pillow`
- [ ] Tạo folder structure (tự động bởi script)
- [ ] Tạo account Freepik (free)
- [ ] Tạo account Vecteezy (free)
- [ ] Chuẩn bị spreadsheet để track sources (hoặc dùng metadata JSON)

---

## 📊 TRACKING TEMPLATE

Tạo file Excel hoặc Google Sheets với columns:

| Image ID | Source | URL | License | Downloaded Date | Quality Check | Notes |
|----------|--------|-----|---------|----------------|---------------|-------|
| wiki_001 | Wikimedia | https://... | CC-BY | 2026-01-21 | ✓ Good | Museum quality |
| free_001 | Freepik | https://... | Freepik Free | 2026-01-21 | ✓ Good | Needs attribution |

---

## ⚠️ LƯU Ý QUAN TRỌNG

### Attribution Requirements:
- **Freepik**: Phải cite "Designed by Freepik" + link
- **Vecteezy**: Phải cite "Vector by Vecteezy" + link  
- **Wikimedia CC-BY**: Phải cite author name + link
- **Unsplash/Pexels**: Không bắt buộc nhưng nên credit photographer

### Legal:
- Tất cả nguồn trên đều OK cho **academic research**
- Nếu commercialize sau này → phải review licenses lại
- Luôn keep track của sources để citation trong paper

---

## 🚀 BẮT ĐẦU NGAY

**Action items cho hôm nay (Day 1)**:

1. ✅ Đã tạo script download_wikimedia.py
2. 🔄 Chạy script: 
   ```bash
   cd "E:\Bai bao\report fpf\GenAI"
   python download_wikimedia.py
   ```
3. ⏳ Trong khi script chạy, tạo account Freepik
4. ⏳ Bắt đầu download thủ công từ Freepik

**Mục tiêu hôm nay**: 50+ ảnh đầu tiên!

---

Bạn có muốn tôi tạo thêm script để automate Freepik download không? (phức tạp hơn vì cần login)

"""
Script để download Hmong pattern images từ Wikimedia Commons
Nguồn: Vietnam Museum of Ethnology Collection
License: CC0 / CC-BY (free for research)
"""

import requests
import os
from pathlib import Path
import json
import time

class WikimediaDownloader:
    def __init__(self, output_dir="dataset/raw/wikimedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://commons.wikimedia.org/w/api.php"
        
    def get_category_images(self, category, limit=100):
        """Lấy danh sách ảnh từ một category trên Wikimedia"""
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "cmtype": "file"
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if "query" in data and "categorymembers" in data["query"]:
            return data["query"]["categorymembers"]
        return []
    
    def get_image_url(self, filename):
        """Lấy URL download của một file ảnh"""
        params = {
            "action": "query",
            "format": "json",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "url|size|mime"
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "imageinfo" in page_data and len(page_data["imageinfo"]) > 0:
                return page_data["imageinfo"][0]
        return None
    
    def download_image(self, url, filename, metadata):
        """Download một ảnh và lưu metadata"""
        try:
            print(f"Downloading: {filename}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Lưu ảnh
            img_path = self.output_dir / filename
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Lưu metadata
            meta_path = self.output_dir / f"{filename}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    
    def download_category(self, category, max_images=50):
        """Download tất cả ảnh từ một category"""
        print(f"\n{'='*60}")
        print(f"Downloading from category: {category}")
        print(f"{'='*60}\n")
        
        members = self.get_category_images(category, limit=max_images)
        
        downloaded = 0
        for i, member in enumerate(members, 1):
            if downloaded >= max_images:
                break
                
            title = member.get("title", "").replace("File:", "")
            print(f"\n[{i}/{len(members)}] Processing: {title}")
            
            # Lấy thông tin ảnh
            img_info = self.get_image_url(title)
            if not img_info:
                print(f"✗ Could not get info for: {title}")
                continue
            
            # Chỉ download ảnh (không phải video)
            mime = img_info.get("mime", "")
            if not mime.startswith("image/"):
                print(f"⊘ Skipping non-image: {mime}")
                continue
            
            url = img_info.get("url")
            if not url:
                print(f"✗ No URL found for: {title}")
                continue
            
            # Chuẩn bị metadata
            metadata = {
                "source": "Wikimedia Commons",
                "category": category,
                "title": title,
                "url": img_info.get("descriptionurl"),
                "download_url": url,
                "size": img_info.get("size"),
                "mime": mime,
                "license": "Check Wikimedia Commons page"
            }
            
            # Download
            if self.download_image(url, title, metadata):
                downloaded += 1
            
            # Delay để không spam server
            time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"✓ Downloaded {downloaded} images from {category}")
        print(f"{'='*60}\n")
        return downloaded


def main():
    downloader = WikimediaDownloader()
    
    # Danh sách categories liên quan đến Hmong và ethnic Vietnamese
    categories = [
        "Hmong_collection_(Vietnam_Museum_of_Ethnology)",
        "Hmong_textiles",
        "Vietnamese_traditional_clothing",
        "Ethnic_minorities_in_Vietnam",
        "Traditional_costumes_of_Vietnam"
    ]
    
    total_downloaded = 0
    
    for category in categories:
        try:
            count = downloader.download_category(category, max_images=30)
            total_downloaded += count
            print(f"\nProgress: {total_downloaded} total images downloaded\n")
            time.sleep(2)  # Delay between categories
        except Exception as e:
            print(f"Error processing category {category}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images downloaded: {total_downloaded}")
    print(f"Output directory: {downloader.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   WIKIMEDIA COMMONS IMAGE DOWNLOADER                      ║
    ║   For Hmong Textile Pattern Research                      ║
    ║   License: CC0/CC-BY (Free for academic research)         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    main()

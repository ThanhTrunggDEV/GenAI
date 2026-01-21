#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Web Viewer for Hmong Pattern Dataset
Local demo to showcase annotated patterns
"""

import json
import base64
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hmong Pattern Dataset Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        
        .stat-item {
            text-align: center;
            color: white;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
        }
        
        .filter-bar {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .card-image {
            width: 100%;
            height: 250px;
            object-fit: cover;
            background: #f0f0f0;
        }
        
        .card-content {
            padding: 15px;
        }
        
        .card-title {
            font-size: 0.9rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        
        .card-motif {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.75rem;
            margin: 2px;
        }
        
        .card-colors {
            margin-top: 10px;
            display: flex;
            gap: 5px;
        }
        
        .color-dot {
            width: 25px;
            height: 25px;
            border-radius: 50%;
            border: 2px solid #ddd;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 800px;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .close-btn {
            float: right;
            font-size: 2rem;
            cursor: pointer;
            color: #666;
        }
        
        .color-indigo { background-color: #4B0082; }
        .color-black { background-color: #000000; }
        .color-red { background-color: #FF0000; }
        .color-white { background-color: #FFFFFF; border-color: #000; }
        .color-blue { background-color: #0000FF; }
        .color-yellow { background-color: #FFD700; }
        .color-green { background-color: #008000; }
        .color-beige { background-color: #F5F5DC; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 Hmong Pattern Dataset</h1>
            <p>AI-Assisted Annotation Demo</p>
        </header>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number" id="total-patterns">0</div>
                <div>Total Patterns</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="total-motifs">0</div>
                <div>Unique Motifs</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">225</div>
                <div>Training Images</div>
            </div>
        </div>
        
        <div class="gallery" id="gallery"></div>
    </div>
    
    <div class="modal" id="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <div id="modal-body"></div>
        </div>
    </div>
    
    <script>
        const DATASET_DATA = {{DATASET_JSON}};
        
        function init() {
            const gallery = document.getElementById('gallery');
            const patterns = DATASET_DATA.slice(0, 45); // Show original 45
            
            document.getElementById('total-patterns').textContent = patterns.length;
            
            const allMotifs = new Set();
            patterns.forEach(p => p.motifs.forEach(m => allMotifs.add(m)));
            document.getElementById('total-motifs').textContent = allMotifs.size;
            
            patterns.forEach((pattern, idx) => {
                const card = createCard(pattern);
                gallery.appendChild(card);
            });
        }
        
        function createCard(pattern) {
            const card = document.createElement('div');
            card.className = 'card';
            card.onclick = () => showDetails(pattern);
            
            card.innerHTML = `
                <img class="card-image" src="${pattern.image}" alt="${pattern.filename}">
                <div class="card-content">
                    <div class="card-title">${pattern.filename}</div>
                    <div>
                        ${pattern.motifs.map(m => `<span class="card-motif">${m}</span>`).join('')}
                    </div>
                    <div class="card-colors">
                        ${pattern.colors.map(c => `<div class="color-dot color-${c}"></div>`).join('')}
                    </div>
                </div>
            `;
            
            return card;
        }
        
        function showDetails(pattern) {
            const modal = document.getElementById('modal');
            const body = document.getElementById('modal-body');
            
            body.innerHTML = `
                <h2>${pattern.filename}</h2>
                <img src="${pattern.image}" style="max-width: 100%; border-radius: 10px; margin: 20px 0;">
                <p><strong>Motifs:</strong> ${pattern.motifs.join(', ')}</p>
                <p><strong>Colors:</strong> ${pattern.colors.join(', ')}</p>
                <p><strong>Symmetry:</strong> ${pattern.symmetry}</p>
                <p><strong>Description:</strong> ${pattern.description || 'N/A'}</p>
            `;
            
            modal.style.display = 'flex';
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        
        window.onclick = (e) => {
            const modal = document.getElementById('modal');
            if (e.target === modal) closeModal();
        };
        
        init();
    </script>
</body>
</html>
"""

def generate_demo_page():
    """Generate demo HTML page with embedded data"""
    
    # Load metadata
    meta_dir = Path("dataset/metadata")
    img_dir = Path("dataset/to_annotate")
    
    patterns = []
    
    for meta_file in sorted(meta_dir.glob("*.json"))[:45]:  # First 45 original
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        img_path = img_dir / metadata["filename"]
        if not img_path.exists():
            continue
        
        # Encode image to base64
        with open(img_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_src = f"data:image/jpeg;base64,{img_data}"
        
        patterns.append({
            "filename": metadata["filename"],
            "image": img_src,
            "motifs": metadata["pattern_info"]["specific_motifs"],
            "colors": metadata["color_info"]["colors"],
            "symmetry": metadata["visual_structure"]["symmetry"],
            "description": metadata.get("notes", "")
        })
    
    # Embed data into HTML
    html = HTML_TEMPLATE.replace("{{DATASET_JSON}}", json.dumps(patterns))
    
    # Save HTML
    output_path = Path("demo_viewer.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Demo page created: {output_path.absolute()}")
    return output_path

if __name__ == "__main__":
    output_file = generate_demo_page()
    print(f"\n🌐 Open in browser: {output_file.absolute()}")
    print("\nOr run: python -m http.server 8000")

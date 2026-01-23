from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path

# Ensure we can import the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from motif.pipeline.generate import generate_patterns
except ImportError as e:
    print(f"Error importing motif module: {e}")
    # Mock function for testing without dependencies if needed
    def generate_patterns(*args, **kwargs):
        # Create a mock response
        output_dir = kwargs.get('output_dir', 'static/generated/mock')
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        import time
        
        saved_paths = []
        num_samples = kwargs.get('num_samples', 4)
        
        # Use our dummy image
        dummy_src = "static/dummy.jpg"
        
        for i in range(num_samples):
            # Create a unique filename
            filename = f"mock_generated_{int(time.time())}_{i}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # If we have a dummy, copy it, otherwise create a blank file
            if os.path.exists(dummy_src):
                shutil.copy(dummy_src, save_path)
            else:
                with open(save_path, 'w') as f:
                    f.write("Mock Image Content")
                    
            saved_paths.append(str(save_path))
            
        return saved_paths

app = Flask(__name__)

# Config
OUTPUT_DIR = "static/generated"
CHECKPOINT_PATH = "outputs/hmong-pattern-lora"  # Correct path to model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        num_samples = int(data.get('num_samples', 4))
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})

        # Clean up old generated images? 
        # For now, we just keep adding to them. 
        # In a real app, maybe cleanup or use unique session subfolders.
        
        # Call the generation pipeline
        # We use a unique job ID or timestamp for filenames to avoid collisions if concurrent users
        # But generate_patterns currently uses simple enumeration.
        # Let's trust the single-person usage for now or accept overwrite risk in this demo.
        # Actually generate_patterns overwrites generated_hmong_001.png etc. 
        # This might be an issue for the browser caching.
        # We can implement a quick fix by moving files or renaming them after generation,
        # but let's see if we can pass a unique output dir per request?
        # generate_patterns takes output_dir. 
        
        import time
        timestamp = int(time.time())
        batch_dir = os.path.join(OUTPUT_DIR, str(timestamp))
        
        saved_paths = generate_patterns(
            checkpoint_path=CHECKPOINT_PATH,
            prompt=prompt,
            num_samples=num_samples,
            output_dir=batch_dir,
            motifs=None, # We could add this to UI later
            colors=None  # We could add this to UI later
        )

        # Convert absolute paths to relative URLs for Flask
        # saved_paths typically returns absolute paths or paths relative to CWD
        # We need URL relative to /static/
        
        image_urls = []
        for p in saved_paths:
            # p might be "static/generated/123456/generated_hmong_001.png"
            # we need "/static/generated/123456/generated_hmong_001.png"
            path_obj = Path(p)
            
            # Find the part relative to static folder
            # Assuming output_dir is inside static
            rel_path = os.path.relpath(path_obj, start=app.root_path)
            # Ensure forward slashes for URL
            url = "/" + rel_path.replace(os.sep, "/")
            image_urls.append(url)

        return jsonify({
            'success': True, 
            'images': image_urls
        })

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting Flask server...")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

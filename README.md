"""
Bengali Image Generator - Web Application

A Flask web app that:
1. Takes a prompt with Bengali text
2. Renders Bengali text as clean PNG images
3. Sends to OpenAI's gpt-image-1.5 with the text images as references
4. Returns the generated image

Run with:
    export OPENAI_API_KEY='your-key'
    python app.py

Then open http://localhost:5000
"""

import os
import re
import base64
import json
from io import BytesIO
from flask import Flask, render_template_string, request, jsonify

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Font paths for different operating systems
FONT_PATHS = [
    # Mac - if you download Noto Sans Bengali
    "/Library/Fonts/NotoSansBengali-SemiBold.ttf",
    "~/Library/Fonts/NotoSansBengali-SemiBold.ttf",
    # Mac - system fonts that might support Bengali
    "/System/Library/Fonts/Supplemental/Kohinoor.ttc",
    # Linux
    "/usr/share/fonts/truetype/noto/NotoSansBengali-SemiBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    # Windows
    "C:/Windows/Fonts/NotoSansBengali-Regular.ttf",
    # Current directory (portable)
    "./NotoSansBengali-SemiBold.ttf",
    "./fonts/NotoSansBengali-SemiBold.ttf",
]

def get_bengali_font_path():
    """Find a working Bengali font on this system."""
    import os
    for path in FONT_PATHS:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    return None

BENGALI_FONT_PATH = get_bengali_font_path()

# Bengali Unicode range
BENGALI_REGEX = re.compile(r'[\u0980-\u09FF]+[\u0980-\u09FF\s]*')


def extract_bengali_text(prompt: str) -> list[str]:
    """Extract all Bengali text segments from a prompt."""
    matches = BENGALI_REGEX.findall(prompt)
    return list(set(m.strip() for m in matches if m.strip()))


def render_bengali_text(text: str, font_size: int = 120) -> Image.Image:
    """Render Bengali text as a PNG with transparent background."""
    font = None
    
    if BENGALI_FONT_PATH:
        try:
            font = ImageFont.truetype(BENGALI_FONT_PATH, font_size)
        except (OSError, IOError) as e:
            print(f"Warning: Could not load font from {BENGALI_FONT_PATH}: {e}")
    
    if font is None:
        print("="*60)
        print("ERROR: No Bengali font found!")
        print("Please download Noto Sans Bengali from:")
        print("  https://fonts.google.com/noto/specimen/Noto+Sans+Bengali")
        print("")
        print("On Mac, put the .ttf file in one of these locations:")
        print("  ~/Library/Fonts/NotoSansBengali-SemiBold.ttf")
        print("  OR in the same folder as this script: ./NotoSansBengali-SemiBold.ttf")
        print("="*60)
        # Use default font as fallback (won't render Bengali correctly)
        font = ImageFont.load_default()
    
    # Measure text
    temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    
    padding = 30
    img_width = bbox[2] - bbox[0] + padding * 2
    img_height = bbox[3] - bbox[1] + padding * 2
    
    # Create image with transparent background
    img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill='black')
    
    return img


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_image(prompt: str, bengali_texts: list[str], rendered_images: list[Image.Image]) -> dict:
    """
    Generate image using OpenAI's API with Bengali text references.
    
    This is the core of your idea:
    - Pass the rendered Bengali text images as references
    - Let the model incorporate them naturally into the generated image
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Build enhanced prompt that tells the model about the reference images
    enhanced_prompt = prompt + "\n\n"
    enhanced_prompt += "CRITICAL INSTRUCTION: I am providing reference images containing Bengali text. "
    enhanced_prompt += "You MUST incorporate this exact text into the generated image. "
    enhanced_prompt += "The text must appear clearly and legibly, positioned appropriately based on the context (e.g., as a title at the top for a poster).\n\n"
    
    for i, text in enumerate(bengali_texts, 1):
        enhanced_prompt += f"Reference Image {i}: Contains Bengali text '{text}'\n"
    
    enhanced_prompt += "\nPreserve the Bengali characters exactly as shown in the reference images. "
    enhanced_prompt += "Do not attempt to re-render or modify the text."
    
    # Convert images to bytes for the API
    image_bytes = []
    for img in rendered_images:
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes.append(buffer)
    
    try:
        # Use the edit endpoint with reference images
        response = client.images.edit(
            model="gpt-image-1",  # or "gpt-image-1.5" if available
            image=image_bytes,
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )
        
        result_b64 = None
        if response.data:
            item = response.data[0]
            if hasattr(item, 'b64_json') and item.b64_json:
                result_b64 = item.b64_json
            elif hasattr(item, 'url') and item.url:
                import requests
                img_response = requests.get(item.url)
                result_b64 = base64.b64encode(img_response.content).decode('utf-8')
        
        return {
            "success": True,
            "image": result_b64,
            "prompt_used": enhanced_prompt
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # If edit endpoint fails, try generation with just the prompt
        # This is a fallback - won't have the reference images but shows the flow works
        try:
            print(f"Edit endpoint failed: {error_msg}")
            print("Trying generation endpoint as fallback...")
            
            response = client.images.generate(
                model="gpt-image-1",
                prompt=enhanced_prompt,
                n=1,
                size="1024x1024"
            )
            
            result_b64 = None
            if response.data:
                item = response.data[0]
                if hasattr(item, 'b64_json') and item.b64_json:
                    result_b64 = item.b64_json
                elif hasattr(item, 'url') and item.url:
                    import requests
                    img_response = requests.get(item.url)
                    result_b64 = base64.b64encode(img_response.content).decode('utf-8')
            
            return {
                "success": True,
                "image": result_b64,
                "prompt_used": enhanced_prompt,
                "note": "Used generation endpoint (fallback). Reference images not included."
            }
            
        except Exception as e2:
            return {
                "success": False,
                "error": f"Edit failed: {error_msg}. Generate failed: {str(e2)}"
            }


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>বাংলা ছবি - Bengali Image Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 2rem;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        h1 {
            font-family: 'Noto Sans Bengali', sans-serif;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #ff6b35, #f7c59f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: rgba(255,255,255,0.6);
            margin-bottom: 2rem;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        textarea {
            width: 100%;
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid rgba(255,255,255,0.1);
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-family: 'Noto Sans Bengali', 'Segoe UI', sans-serif;
            font-size: 1rem;
            resize: vertical;
            min-height: 100px;
        }
        
        textarea:focus {
            outline: none;
            border-color: #ff6b35;
        }
        
        button {
            padding: 1rem 2rem;
            border-radius: 8px;
            border: none;
            background: linear-gradient(135deg, #ff6b35, #f7c59f);
            color: #1a1a2e;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255,107,53,0.3);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .examples {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(255,107,53,0.1);
            border-radius: 8px;
        }
        
        .examples h4 {
            color: #ff6b35;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .examples ul {
            list-style: none;
        }
        
        .examples li {
            padding: 0.5rem;
            cursor: pointer;
            border-radius: 4px;
            transition: background 0.2s;
        }
        
        .examples li:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .result {
            text-align: center;
        }
        
        .result img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .bengali-preview {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        .bengali-preview img {
            height: 60px;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            padding: 0.5rem;
        }
        
        .status {
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .status.loading {
            background: rgba(255,193,7,0.2);
            color: #ffc107;
        }
        
        .status.error {
            background: rgba(244,67,54,0.2);
            color: #f44336;
        }
        
        .status.success {
            background: rgba(76,175,80,0.2);
            color: #4caf50;
        }
        
        .prompt-used {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .how-it-works {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }
        
        .step {
            background: rgba(255,255,255,0.03);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }
        
        .step-num {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #ff6b35, #f7c59f);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            font-weight: bold;
            color: #1a1a2e;
            font-family: 'Noto Sans Bengali', sans-serif;
        }
        
        .step h4 {
            margin-bottom: 0.5rem;
        }
        
        .step p {
            color: rgba(255,255,255,0.6);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>বাংলা ছবি</h1>
        <p class="subtitle">Bengali Text in AI Images — Your Original Idea</p>
        
        <div class="card">
            <label for="prompt">Enter your prompt (include Bengali text):</label>
            <textarea id="prompt" placeholder="Example: A movie poster with 'বাংলা চলচ্চিত্র' as the title, dramatic lighting"></textarea>
            
            <div class="examples">
                <h4>Try these examples:</h4>
                <ul>
                    <li onclick="setPrompt(this)">🎬 A movie poster with "বাংলা চলচ্চিত্র" as the main title, cinematic dramatic lighting</li>
                    <li onclick="setPrompt(this)">🎂 A birthday card with "শুভ জন্মদিন" written beautifully, balloons and confetti</li>
                    <li onclick="setPrompt(this)">📚 A book cover with title "রবীন্দ্রনাথের কবিতা", elegant vintage design</li>
                    <li onclick="setPrompt(this)">🏪 A shop sign that says "মিষ্টির দোকান", traditional Bengali sweet shop</li>
                </ul>
            </div>
            
            <button onclick="generateImage()" id="generateBtn">Generate Image →</button>
            
            <div id="bengaliPreview" class="bengali-preview"></div>
            <div id="status"></div>
        </div>
        
        <div id="resultCard" class="card result" style="display: none;">
            <h3>Generated Image</h3>
            <img id="resultImage" src="" alt="Generated image">
            <details style="margin-top: 1rem; text-align: left;">
                <summary style="cursor: pointer; color: rgba(255,255,255,0.6);">Show prompt sent to API</summary>
                <div id="promptUsed" class="prompt-used"></div>
            </details>
        </div>
        
        <div class="how-it-works">
            <div class="step">
                <div class="step-num">১</div>
                <h4>Extract Bengali</h4>
                <p>Detect Bengali text in your prompt</p>
            </div>
            <div class="step">
                <div class="step-num">২</div>
                <h4>Render Text</h4>
                <p>Create clean PNG with proper fonts</p>
            </div>
            <div class="step">
                <div class="step-num">৩</div>
                <h4>Send to OpenAI</h4>
                <p>Include text image as reference</p>
            </div>
            <div class="step">
                <div class="step-num">৪</div>
                <h4>Get Result</h4>
                <p>AI places text naturally</p>
            </div>
        </div>
    </div>
    
    <script>
        function setPrompt(el) {
            document.getElementById('prompt').value = el.textContent.substring(2).trim();
        }
        
        async function generateImage() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            const status = document.getElementById('status');
            const resultCard = document.getElementById('resultCard');
            const bengaliPreview = document.getElementById('bengaliPreview');
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            status.className = 'status loading';
            status.textContent = '⏳ Extracting Bengali text and rendering...';
            resultCard.style.display = 'none';
            bengaliPreview.innerHTML = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                
                const data = await response.json();
                
                // Show rendered Bengali text previews
                if (data.rendered_texts) {
                    bengaliPreview.innerHTML = '<strong>Rendered Bengali text:</strong><br>';
                    data.rendered_texts.forEach(b64 => {
                        const img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + b64;
                        bengaliPreview.appendChild(img);
                    });
                }
                
                if (data.success) {
                    status.className = 'status success';
                    status.textContent = '✓ Image generated successfully' + (data.note ? ' (' + data.note + ')' : '');
                    
                    document.getElementById('resultImage').src = 'data:image/png;base64,' + data.image;
                    document.getElementById('promptUsed').textContent = data.prompt_used || '';
                    resultCard.style.display = 'block';
                } else {
                    status.className = 'status error';
                    status.textContent = '✗ Error: ' + data.error;
                }
                
            } catch (err) {
                status.className = 'status error';
                status.textContent = '✗ Error: ' + err.message;
            }
            
            btn.disabled = false;
            btn.textContent = 'Generate Image →';
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"})
    
    if not OPENAI_API_KEY:
        return jsonify({"success": False, "error": "OPENAI_API_KEY not set"})
    
    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)
    
    # Render Bengali text as images
    rendered_images = []
    rendered_b64 = []
    for text in bengali_texts:
        img = render_bengali_text(text)
        rendered_images.append(img)
        rendered_b64.append(image_to_base64(img))
    
    if not bengali_texts:
        # No Bengali text found - just do normal generation
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            result_b64 = None
            if response.data:
                item = response.data[0]
                if hasattr(item, 'b64_json') and item.b64_json:
                    result_b64 = item.b64_json
                elif hasattr(item, 'url') and item.url:
                    import requests as req
                    img_response = req.get(item.url)
                    result_b64 = base64.b64encode(img_response.content).decode('utf-8')
            
            return jsonify({
                "success": True,
                "image": result_b64,
                "prompt_used": prompt,
                "rendered_texts": [],
                "note": "No Bengali text detected - used standard generation"
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    # Generate with Bengali text references
    result = generate_image(prompt, bengali_texts, rendered_images)
    result["rendered_texts"] = rendered_b64
    
    return jsonify(result)


if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print("="*60)
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print("="*60)
    
    print("\n🚀 Starting Bengali Image Generator...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

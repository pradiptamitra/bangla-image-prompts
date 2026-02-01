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

if BENGALI_FONT_PATH:
    print(f"Found Bengali font: {BENGALI_FONT_PATH}")
else:
    print("WARNING: No Bengali font found! Text will not render correctly.")
    print("Download Noto Sans Bengali from: https://fonts.google.com/noto/specimen/Noto+Sans+Bengali")
    print("Then place the .ttf file in one of these locations:")
    for p in FONT_PATHS:
        print(f"  - {p}")

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

    # Remove Bengali text from the prompt - we only want to send the rendered image
    cleaned_prompt = prompt
    for text in bengali_texts:
        cleaned_prompt = cleaned_prompt.replace(text, "[TEXT FROM IMAGE]")

    # Also remove quoted versions
    cleaned_prompt = re.sub(r'["\']?\[TEXT FROM IMAGE\]["\']?', '[TEXT FROM IMAGE]', cleaned_prompt)

    # Build enhanced prompt that references the image without including Bengali unicode
    enhanced_prompt = cleaned_prompt + "\n\n"
    enhanced_prompt += "CRITICAL INSTRUCTION: I am providing a reference image containing text. "
    enhanced_prompt += "You MUST copy this exact text into the generated image - do not recreate or modify it. "
    enhanced_prompt += "Use the text from the reference image exactly as rendered, placing it appropriately based on context (e.g., as a title at the top for a poster)."
    
    # Convert images to bytes for the API
    # Note: BytesIO needs a .name attribute with .png extension for OpenAI to detect mimetype
    image_bytes = []
    for i, img in enumerate(rendered_images):
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        buffer.name = f"bengali_text_{i}.png"  # Required for OpenAI to detect mimetype
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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Bengali:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0a0a0b;
            --bg-secondary: #141416;
            --bg-card: rgba(255, 255, 255, 0.03);
            --bg-card-hover: rgba(255, 255, 255, 0.05);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-accent: rgba(255, 255, 255, 0.1);
            --text-primary: #fafafa;
            --text-secondary: rgba(255, 255, 255, 0.5);
            --text-tertiary: rgba(255, 255, 255, 0.35);
            --accent: #6366f1;
            --accent-light: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --success: #22c55e;
            --warning: #eab308;
            --error: #ef4444;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 20px;
            --radius-xl: 24px;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .noise {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.015;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        }

        .gradient-bg {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(ellipse 80% 50% at 50% -20%, var(--accent-glow), transparent),
                radial-gradient(ellipse 60% 40% at 100% 0%, rgba(139, 92, 246, 0.08), transparent),
                radial-gradient(ellipse 50% 30% at 0% 100%, rgba(59, 130, 246, 0.05), transparent);
            pointer-events: none;
        }

        .container {
            position: relative;
            max-width: 720px;
            margin: 0 auto;
            padding: 3rem 1.5rem 4rem;
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .logo {
            font-family: 'Noto Sans Bengali', sans-serif;
            font-size: 3rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .tagline {
            color: var(--text-tertiary);
            font-size: 0.95rem;
            font-weight: 400;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .card:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-accent);
        }

        .input-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        textarea {
            width: 100%;
            padding: 1rem 1.25rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            font-size: 1rem;
            resize: none;
            min-height: 120px;
            transition: all 0.2s ease;
        }

        textarea::placeholder {
            color: var(--text-tertiary);
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .examples-section {
            margin-bottom: 1.5rem;
        }

        .examples-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .examples-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }

        .example-chip {
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .example-chip:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-accent);
            color: var(--text-primary);
        }

        .example-chip .bengali {
            font-family: 'Noto Sans Bengali', sans-serif;
            color: var(--accent-light);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            width: 100%;
            padding: 1rem 1.5rem;
            border-radius: var(--radius-md);
            border: none;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: var(--accent);
            color: white;
        }

        .btn-primary:hover {
            background: var(--accent-light);
            transform: translateY(-1px);
            box-shadow: 0 8px 30px -10px rgba(99, 102, 241, 0.5);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-secondary {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-accent);
        }

        .btn-secondary:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
            border-color: var(--text-tertiary);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        .btn-icon {
            width: 18px;
            height: 18px;
            transition: transform 0.2s ease;
        }

        .btn:hover .btn-icon {
            transform: translateX(2px);
        }

        .preview-section {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-subtle);
        }

        .preview-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .bengali-preview {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .bengali-preview img {
            height: 48px;
            background: white;
            border-radius: var(--radius-sm);
            padding: 0.5rem 0.75rem;
        }

        .status {
            margin-top: 1rem;
            padding: 0.875rem 1rem;
            border-radius: var(--radius-sm);
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status.loading {
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid rgba(234, 179, 8, 0.2);
            color: var(--warning);
        }

        .status.error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: var(--error);
        }

        .status.success {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            color: var(--success);
        }

        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top-color: currentColor;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .result-card {
            text-align: center;
        }

        .result-card h3 {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 1.25rem;
        }

        .result-image-wrapper {
            position: relative;
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: var(--bg-secondary);
        }

        .result-image-wrapper img {
            width: 100%;
            display: block;
        }

        .result-actions {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
        }

        details {
            margin-top: 1.5rem;
            text-align: left;
        }

        summary {
            font-size: 0.8rem;
            color: var(--text-tertiary);
            cursor: pointer;
            transition: color 0.2s;
        }

        summary:hover {
            color: var(--text-secondary);
        }

        .prompt-used {
            margin-top: 0.75rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            white-space: pre-wrap;
            max-height: 160px;
            overflow-y: auto;
            line-height: 1.5;
        }

        .steps-section {
            margin-top: 2rem;
        }

        .steps-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: center;
            margin-bottom: 1rem;
        }

        .steps {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .step {
            flex: 1;
            text-align: center;
            padding: 1.25rem 0.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            transition: all 0.2s ease;
        }

        .step:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-accent);
        }

        .step-num {
            width: 28px;
            height: 28px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.75rem;
            font-family: 'Noto Sans Bengali', sans-serif;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-light);
        }

        .step-title {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .step-desc {
            font-size: 0.7rem;
            color: var(--text-tertiary);
        }

        @media (max-width: 600px) {
            .container {
                padding: 2rem 1rem 3rem;
            }

            .logo {
                font-size: 2.25rem;
            }

            .card {
                padding: 1.5rem;
            }

            .examples-grid {
                grid-template-columns: 1fr;
            }

            .steps {
                flex-wrap: wrap;
            }

            .step {
                flex: 1 1 45%;
            }
        }
    </style>
</head>
<body>
    <div class="noise"></div>
    <div class="gradient-bg"></div>

    <div class="container">
        <header>
            <h1 class="logo">বাংলা ছবি</h1>
            <p class="tagline">Generate AI images with perfect Bengali text</p>
        </header>

        <div class="card">
            <div class="input-group">
                <label for="prompt">Describe your image</label>
                <textarea id="prompt" placeholder="A movie poster with 'বাংলা চলচ্চিত্র' as the title..."></textarea>
            </div>

            <div class="examples-section">
                <div class="examples-label">Quick examples</div>
                <div class="examples-grid">
                    <div class="example-chip" onclick="setPrompt('A movie poster with \\'বাংলা চলচ্চিত্র\\' as the main title, cinematic dramatic lighting')">
                        Movie poster · <span class="bengali">বাংলা চলচ্চিত্র</span>
                    </div>
                    <div class="example-chip" onclick="setPrompt('A birthday card with \\'শুভ জন্মদিন\\' written beautifully, balloons and confetti')">
                        Birthday card · <span class="bengali">শুভ জন্মদিন</span>
                    </div>
                    <div class="example-chip" onclick="setPrompt('A book cover with title \\'রবীন্দ্রনাথের কবিতা\\', elegant vintage design')">
                        Book cover · <span class="bengali">রবীন্দ্রনাথের কবিতা</span>
                    </div>
                    <div class="example-chip" onclick="setPrompt('A shop sign that says \\'মিষ্টির দোকান\\', traditional Bengali sweet shop')">
                        Shop sign · <span class="bengali">মিষ্টির দোকান</span>
                    </div>
                </div>
            </div>

            <div style="display: flex; gap: 0.75rem;">
                <button class="btn btn-primary" onclick="generateImage()" id="generateBtn" style="flex: 1;">
                    Generate
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                    </svg>
                </button>
                <button class="btn btn-secondary" onclick="renderOnly()" id="renderBtn" title="Test Bengali text rendering without calling OpenAI">
                    Render
                </button>
                <button class="btn btn-secondary" onclick="previewPrompt()" id="promptBtn" title="Show the prompt that would be sent to OpenAI">
                    Prompt
                </button>
            </div>

            <div id="bengaliPreviewSection" class="preview-section" style="display: none;">
                <div class="preview-label">Rendered text</div>
                <div id="bengaliPreview" class="bengali-preview"></div>
            </div>

            <div id="status"></div>
        </div>

        <div id="resultCard" class="card result-card" style="display: none;">
            <h3>Generated Image</h3>
            <div class="result-image-wrapper">
                <img id="resultImage" src="" alt="Generated image">
            </div>
            <details>
                <summary>View prompt sent to API</summary>
                <div id="promptUsed" class="prompt-used"></div>
            </details>
        </div>

        <div class="steps-section">
            <div class="steps-label">How it works</div>
            <div class="steps">
                <div class="step">
                    <div class="step-num">১</div>
                    <div class="step-title">Extract</div>
                    <div class="step-desc">Detect Bengali text</div>
                </div>
                <div class="step">
                    <div class="step-num">২</div>
                    <div class="step-title">Render</div>
                    <div class="step-desc">Create text PNG</div>
                </div>
                <div class="step">
                    <div class="step-num">৩</div>
                    <div class="step-title">Generate</div>
                    <div class="step-desc">Send to OpenAI</div>
                </div>
                <div class="step">
                    <div class="step-num">৪</div>
                    <div class="step-title">Result</div>
                    <div class="step-desc">AI places text</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        async function previewPrompt() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            const btn = document.getElementById('promptBtn');
            const status = document.getElementById('status');
            const resultCard = document.getElementById('resultCard');

            btn.disabled = true;
            btn.textContent = '...';
            resultCard.style.display = 'none';

            try {
                const response = await fetch('/preview-prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                const data = await response.json();

                status.className = 'status success';
                status.innerHTML = '<strong>Prompt preview:</strong><pre style="margin-top:0.5rem;white-space:pre-wrap;font-size:0.8rem;color:var(--text-secondary);">' +
                    data.openai_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';

            } catch (err) {
                status.className = 'status error';
                status.textContent = err.message;
            }

            btn.disabled = false;
            btn.textContent = 'Prompt';
        }

        async function renderOnly() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            const btn = document.getElementById('renderBtn');
            const status = document.getElementById('status');
            const bengaliPreview = document.getElementById('bengaliPreview');
            const bengaliPreviewSection = document.getElementById('bengaliPreviewSection');
            const resultCard = document.getElementById('resultCard');

            btn.disabled = true;
            btn.textContent = 'Rendering...';
            resultCard.style.display = 'none';

            try {
                const response = await fetch('/render', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                const data = await response.json();

                if (data.rendered_texts && data.rendered_texts.length > 0) {
                    bengaliPreviewSection.style.display = 'block';
                    bengaliPreview.innerHTML = '';
                    data.rendered_texts.forEach(b64 => {
                        const img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + b64;
                        bengaliPreview.appendChild(img);
                    });
                    status.className = 'status success';
                    status.textContent = 'Rendered ' + data.bengali_texts.length + ' Bengali text segment(s)';
                } else {
                    bengaliPreviewSection.style.display = 'none';
                    status.className = 'status error';
                    status.textContent = 'No Bengali text found in prompt';
                }

            } catch (err) {
                status.className = 'status error';
                status.textContent = err.message;
            }

            btn.disabled = false;
            btn.textContent = 'Render Only';
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
            const bengaliPreviewSection = document.getElementById('bengaliPreviewSection');

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Generating...';
            status.className = 'status loading';
            status.innerHTML = '<span class="spinner"></span> Rendering Bengali text...';
            resultCard.style.display = 'none';
            bengaliPreviewSection.style.display = 'none';
            bengaliPreview.innerHTML = '';

            try {
                // Step 1: Quickly render Bengali text and show preview
                const renderResponse = await fetch('/render', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                const renderData = await renderResponse.json();

                // Show rendered Bengali text previews immediately
                if (renderData.rendered_texts && renderData.rendered_texts.length > 0) {
                    bengaliPreviewSection.style.display = 'block';
                    bengaliPreview.innerHTML = '';
                    renderData.rendered_texts.forEach(b64 => {
                        const img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + b64;
                        bengaliPreview.appendChild(img);
                    });
                }

                // Step 2: Now call OpenAI (the slow part)
                status.innerHTML = '<span class="spinner"></span> Generating image with OpenAI...';

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                const data = await response.json();

                if (data.success) {
                    status.className = 'status success';
                    status.textContent = 'Image generated successfully' + (data.note ? ' — ' + data.note : '');

                    document.getElementById('resultImage').src = 'data:image/png;base64,' + data.image;
                    document.getElementById('promptUsed').textContent = data.prompt_used || '';
                    resultCard.style.display = 'block';
                    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    status.className = 'status error';
                    status.textContent = data.error;
                }

            } catch (err) {
                status.className = 'status error';
                status.textContent = err.message;
            }

            btn.disabled = false;
            btn.innerHTML = 'Generate <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>';
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/render', methods=['POST'])
def render():
    """Quick endpoint to just render Bengali text - called first for immediate preview."""
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"bengali_texts": [], "rendered_texts": []})

    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)

    # Render Bengali text as images
    rendered_b64 = []
    for text in bengali_texts:
        img = render_bengali_text(text)
        rendered_b64.append(image_to_base64(img))

    return jsonify({
        "bengali_texts": bengali_texts,
        "rendered_texts": rendered_b64
    })


@app.route('/preview-prompt', methods=['POST'])
def preview_prompt():
    """Show the prompt that would be sent to OpenAI without calling the API."""
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"prompt": ""})

    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)

    # Build the same prompt that generate_image would build
    cleaned_prompt = prompt
    for text in bengali_texts:
        cleaned_prompt = cleaned_prompt.replace(text, "[TEXT FROM IMAGE]")

    cleaned_prompt = re.sub(r'["\']?\[TEXT FROM IMAGE\]["\']?', '[TEXT FROM IMAGE]', cleaned_prompt)

    enhanced_prompt = cleaned_prompt + "\n\n"
    enhanced_prompt += "CRITICAL INSTRUCTION: I am providing a reference image containing text. "
    enhanced_prompt += "You MUST copy this exact text into the generated image - do not recreate or modify it. "
    enhanced_prompt += "Use the text from the reference image exactly as rendered, placing it appropriately based on context (e.g., as a title at the top for a poster)."

    return jsonify({
        "original_prompt": prompt,
        "bengali_texts": bengali_texts,
        "openai_prompt": enhanced_prompt
    })


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

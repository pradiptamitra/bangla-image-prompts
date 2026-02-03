"""
Bengali Image Generator - Web Application

A Flask web app that:
1. Takes a prompt with Bengali text
2. Renders Bengali text as clean PNG images
3. Sends to Gemini 2.5 Flash with the text images as references
4. Returns the generated image

Uses OpenAI gpt-4o-mini for prompt rewriting, Gemini for image generation.

Run with:
    export OPENAI_API_KEY='your-key'
    export GEMINI_API_KEY='your-key'
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
from google import genai
from google.genai import types

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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


def render_bengali_text(text: str, font_size: int = 120, mask: bool = False) -> Image.Image:
    """Render Bengali text as an image.

    If mask=False: plain black text on white background.
    If mask=True:  text on a bordered plaque with colored circle overlay
                   to disguise it as a graphic and disrupt text recognition.
    """
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
        font = ImageFont.load_default()

    # Measure text
    temp_img = Image.new('RGB', (1, 1), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if not mask:
        # Plain mode: black text on white background
        padding = 30
        img_width = text_w + padding * 2
        img_height = text_h + padding * 2
        img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill='black')
        return img

    # Masked mode: plaque + circle overlay on white background
    pad_x = int(text_h * 0.6)
    pad_y = int(text_h * 0.45)
    border_width = max(3, font_size // 30)
    corner_radius = int(text_h * 0.35)

    img_width = text_w + pad_x * 2
    img_height = text_h + pad_y * 2

    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Center the text
    text_x = (img_width - text_w) // 2 - bbox[0]
    text_y = (img_height - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill='black')

    # Scatter non-overlapping semi-transparent rectangles to disrupt text recognition
    import random
    random.seed(hash(text))
    overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    palette = [(180, 155, 120), (150, 135, 115)]  # light brown / tan
    min_w = max(10, int(text_h * 0.15))
    max_w = max(25, int(text_h * 0.4))
    min_h = max(8, int(text_h * 0.1))
    max_h = max(20, int(text_h * 0.3))
    num_rects = max(6, (img_width * img_height) // (max_w * max_h * 10))
    placed = []
    attempts = 0
    while len(placed) < num_rects and attempts < num_rects * 20:
        attempts += 1
        rw = random.randint(min_w, max_w)
        rh = random.randint(min_h, max_h)
        rx = random.randint(0, img_width - rw)
        ry = random.randint(0, img_height - rh)
        # Check for overlap with already placed rectangles
        overlap = False
        for (px, py, pw, ph) in placed:
            if rx < px + pw and rx + rw > px and ry < py + ph and ry + rh > py:
                overlap = True
                break
        if overlap:
            continue
        placed.append((rx, ry, rw, rh))
        color = random.choice(palette)
        alpha = random.randint(80, 140)
        ov_draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(*color, alpha))

    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')

    return img


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


REWRITE_SYSTEM_PROMPT = """You are an image-generation prompt rewriter. Your job is to transform the user's raw prompt into a clean, precise prompt for an AI image generator.

You will be given:
- The user's raw prompt (which may contain Bengali text references)
- The number of user-uploaded images
- The number of app-rendered Bengali text images (these are graphic overlays, NOT text)

IMAGE ORDERING CONVENTION:
- User-uploaded images come FIRST (positions 1..N)
- App-rendered graphic overlays come AFTER (positions N+1..M)
- Refer to images by ordinal position: "the first attached image", "the second attached image", etc.

RULES FOR USER-UPLOADED IMAGES:
- Describe them as the raw prompt does. If the user attaches as image and describes them as background, do the same.
- Example: "The first attached image is the background scene"

RULES FOR GRAPHIC OVERLAY IMAGES (the app-rendered ones):
- These are pre-rendered GRAPHIC SHAPES on a transparent background. 
- NEVER use the words "text", "title", "writing", "script", "characters", "letters", "words", or "font" when referring to these images. The image generator will try to re-render text if you describe them that way.
- Instead, describe them as: "graphic in the first attachement" etc. Otherwise capture the meaning of the raw prompt. Examples:
    * 'Use [bengali text] as the title' becomes 'Use the image in first attachment as the title'
    * 'Render [bengali text] in small font next to the flower' becomes 'Render the image in first attachment in a small size next to the flower'
- Rewrite the prompt to retain raw user instruction regarding placement (e.g. "on top") etc, but rephrase as to
  reference the input as a image, graphic etc.
- The graphic images have a white background. Always include this instruction: "Blend the white background of the graphic seamlessly into the scene — treat the white area as transparent and integrate only the dark shapes into the composition."

OUTPUT RULES:
- Output ONLY the rewritten prompt — no explanations, no preamble
- CRITICAL: Your output must be pure ASCII/English. Do NOT include any Bengali/Bangla script (Unicode range U+0980–U+09FF) anywhere in your output. Never transliterate, quote, or reproduce any non-Latin script.
- NEVER describe the overlay images as containing "text" or "writing" — always call them graphics/shapes/artwork.
- Reference all images by their ordinal number
- Be concise but precise about placement, styling, and composition"""


MAX_REWRITE_RETRIES = 3


def _contains_bengali(text: str) -> bool:
    """Check if a string contains any Bengali Unicode characters."""
    return bool(re.search(r'[\u0980-\u09FF]', text))


def rewrite_prompt(raw_prompt: str, num_user_images: int, bengali_texts: list[str]) -> str:
    """Call gpt-4o-mini to rewrite the user's prompt for image generation.
    Retries up to MAX_REWRITE_RETRIES times if Bengali text leaks into the output."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    num_text_images = len(bengali_texts)

    user_message = f"""Raw prompt: {raw_prompt}

Number of user-uploaded images: {num_user_images}
Number of app-rendered Bengali text images: {num_text_images}
Bengali text strings extracted: {json.dumps(bengali_texts, ensure_ascii=False)}

Rewrite this into a clean image-generation prompt following the rules."""

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for attempt in range(MAX_REWRITE_RETRIES):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )

        result = response.choices[0].message.content.strip()

        if not _contains_bengali(result):
            return result

        print(f"Rewrite attempt {attempt + 1}: Bengali text leaked, retrying...")
        # Append the bad response and a correction request so the LLM learns from its mistake
        messages.append({"role": "assistant", "content": result})
        messages.append({"role": "user", "content": "Your output contains Bengali script characters. Rewrite it again using ONLY English/ASCII. Never include any Bengali characters. Refer to the overlay images only as 'the graphic from the Nth attached image' — never as text, title, or writing."})

    # Final attempt still had Bengali — strip it out as a last resort
    print("WARNING: Bengali text persisted after retries, stripping it from output")
    return re.sub(r'[\u0980-\u09FF]+', '', result).strip()


def generate_image(prompt: str, bengali_texts: list[str], rendered_images: list[Image.Image], user_images: list[Image.Image] = None, rewritten_prompt: str = None) -> dict:
    """
    Generate image using Gemini's API with Bengali text references and user images.

    Pipeline:
    1. Rewrite the prompt using gpt-4o-mini (or use cached rewritten_prompt if provided)
    2. Send rewritten prompt + all images (user uploads first, then rendered text) to Gemini
    """
    if user_images is None:
        user_images = []

    # Use provided rewritten prompt, or call the LLM rewriter
    if not rewritten_prompt:
        try:
            rewritten_prompt = rewrite_prompt(prompt, len(user_images), bengali_texts)
        except Exception as e:
            print(f"Prompt rewrite failed, falling back to manual prompt: {e}")
            rewritten_prompt = prompt
            for text in bengali_texts:
                rewritten_prompt = rewritten_prompt.replace(text, "[TEXT FROM IMAGE]")
            rewritten_prompt = re.sub(r'["\']?\[TEXT FROM IMAGE\]["\']?', '[TEXT FROM IMAGE]', rewritten_prompt)
            rewritten_prompt += "\n\nCRITICAL INSTRUCTION: I am providing a reference image containing a graphic design element. "
            rewritten_prompt += "Composite this graphic into the generated image exactly as-is — preserve its exact shapes. "
            rewritten_prompt += "Do not redraw, reinterpret, or alter the shapes. Place the graphic appropriately based on context. "
            rewritten_prompt += "Blend the white background of the graphic seamlessly into the scene — treat the white area as transparent and integrate only the dark shapes into the composition."

    # Build contents list: user uploads first, then rendered Bengali text, then prompt
    contents = []
    for img in user_images:
        contents.append(img)
    for img in rendered_images:
        contents.append(img)
    contents.append(rewritten_prompt)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )

        result_b64 = None
        for part in response.parts:
            if part.inline_data is not None:
                result_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                break

        return {
            "success": True,
            "image": result_b64,
            "prompt_used": rewritten_prompt
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
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

        .upload-group {
            margin-bottom: 1.5rem;
        }

        .upload-area {
            position: relative;
            border: 1px dashed var(--border-accent);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: var(--bg-secondary);
        }

        .upload-area:hover {
            border-color: var(--accent);
            background: var(--accent-glow);
        }

        .upload-area input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .upload-area-text {
            font-size: 0.85rem;
            color: var(--text-tertiary);
        }

        .upload-area-text strong {
            color: var(--accent-light);
        }

        .upload-thumbnails {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.75rem;
        }

        .upload-thumb {
            position: relative;
            width: 64px;
            height: 64px;
            border-radius: var(--radius-sm);
            overflow: hidden;
            border: 1px solid var(--border-accent);
        }

        .upload-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .upload-thumb-remove {
            position: absolute;
            top: 2px;
            right: 2px;
            width: 18px;
            height: 18px;
            background: rgba(0,0,0,0.7);
            border: none;
            border-radius: 50%;
            color: white;
            font-size: 11px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
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
                    <div class="example-chip" onclick="setPrompt('A book cover with title \\'চর্যাপদ\\', elegant vintage design')">
                        Book cover · <span class="bengali">চর্যাপদ</span>
                    </div>
                    <div class="example-chip" onclick="setPrompt('A shop sign that says \\'মিষ্টির দোকান\\', traditional Bengali sweet shop')">
                        Shop sign · <span class="bengali">মিষ্টির দোকান</span>
                    </div>
                </div>
            </div>

            <div class="upload-group">
                <label>Attach your own images (optional)</label>
                <div class="upload-area" id="uploadArea">
                    <input type="file" multiple accept="image/*" id="fileInput" onchange="handleFileSelect(event)">
                    <div class="upload-area-text"><strong>Click to upload</strong> or drag and drop</div>
                </div>
                <div class="upload-thumbnails" id="uploadThumbnails"></div>
            </div>

            <div style="margin-bottom: 1.25rem;">
                <label style="display: inline-flex; align-items: center; gap: 0.5rem; cursor: pointer; user-select: none;">
                    <input type="checkbox" id="maskText" checked style="width: 16px; height: 16px; accent-color: var(--accent);">
                    <span style="font-size: 0.85rem; color: var(--text-secondary);">Mask text (overlays shapes to prevent the model from reading and re-rendering the text)</span>
                </label>
            </div>

            <div style="display: flex; gap: 0.75rem;">
                <button class="btn btn-primary" onclick="generateImage()" id="generateBtn" style="flex: 1;">
                    Generate
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                    </svg>
                </button>
                <button class="btn btn-secondary" onclick="renderOnly()" id="renderBtn" title="Test Bengali text rendering without calling the API">
                    Render
                </button>
                <button class="btn btn-secondary" onclick="previewPrompt()" id="promptBtn" title="Show the rewritten prompt that would be sent to the API">
                    Rewritten Prompt
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
                    <div class="step-desc">Send to AI</div>
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
        // Track uploaded files as base64 data URLs
        let uploadedFiles = [];

        // Cache for rewritten prompts — avoids redundant gpt-4o-mini calls
        let rewriteCache = { prompt: null, numImages: null, rewrittenPrompt: null };

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        function handleFileSelect(event) {
            const files = Array.from(event.target.files);
            files.forEach(file => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedFiles.push(e.target.result);
                    renderUploadThumbnails();
                };
                reader.readAsDataURL(file);
            });
            // Reset input so the same file can be re-selected
            event.target.value = '';
        }

        function removeUpload(index) {
            uploadedFiles.splice(index, 1);
            renderUploadThumbnails();
        }

        function renderUploadThumbnails() {
            const container = document.getElementById('uploadThumbnails');
            container.innerHTML = '';
            uploadedFiles.forEach((dataUrl, i) => {
                const thumb = document.createElement('div');
                thumb.className = 'upload-thumb';
                thumb.innerHTML = '<img src="' + dataUrl + '">' +
                    '<button class="upload-thumb-remove" onclick="removeUpload(' + i + ')">&#215;</button>';
                container.appendChild(thumb);
            });
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
                    body: JSON.stringify({
                        prompt: prompt,
                        num_user_images: uploadedFiles.length
                    })
                });

                const data = await response.json();

                if (data.error) {
                    status.className = 'status error';
                    status.textContent = data.error;
                } else {
                    // Cache the rewritten prompt
                    rewriteCache = { prompt: prompt, numImages: uploadedFiles.length, rewrittenPrompt: data.openai_prompt };

                    const method = data.rewrite_method === 'llm' ? ' (LLM rewritten)' : ' (fallback)';
                    status.className = 'status success';
                    status.innerHTML = '<strong>Prompt preview' + method + ':</strong>' +
                        '<pre style="margin-top:0.5rem;white-space:pre-wrap;font-size:0.8rem;color:var(--text-secondary);">' +
                        data.openai_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';
                }

            } catch (err) {
                status.className = 'status error';
                status.textContent = err.message;
            }

            btn.disabled = false;
            btn.textContent = 'Rewritten Prompt';
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
                const mask_text = document.getElementById('maskText').checked;
                const response = await fetch('/render', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, mask_text })
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
            btn.textContent = 'Render';
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
                const mask_text = document.getElementById('maskText').checked;

                // Step 1: Quickly render Bengali text and show preview
                const renderResponse = await fetch('/render', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, mask_text })
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

                // Step 2: Call generate with prompt + user images
                // Use cached rewritten prompt if inputs haven't changed
                const payload = { prompt: prompt, user_images: uploadedFiles, mask_text };
                if (rewriteCache.prompt === prompt && rewriteCache.numImages === uploadedFiles.length && rewriteCache.rewrittenPrompt) {
                    payload.rewritten_prompt = rewriteCache.rewrittenPrompt;
                    status.innerHTML = '<span class="spinner"></span> Generating image (using cached prompt)...';
                } else {
                    status.innerHTML = '<span class="spinner"></span> Rewriting prompt & generating image...';
                }

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
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
    mask = data.get('mask_text', False)

    if not prompt:
        return jsonify({"bengali_texts": [], "rendered_texts": []})

    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)

    # Render Bengali text as images
    rendered_b64 = []
    for text in bengali_texts:
        img = render_bengali_text(text, mask=mask)
        rendered_b64.append(image_to_base64(img))

    return jsonify({
        "bengali_texts": bengali_texts,
        "rendered_texts": rendered_b64
    })


@app.route('/rewrite-prompt', methods=['POST'])
def rewrite_prompt_endpoint():
    """Rewrite a prompt using gpt-4o-mini."""
    data = request.json
    prompt = data.get('prompt', '')
    num_user_images = data.get('num_user_images', 0)
    bengali_texts = data.get('bengali_texts', None)

    if not prompt:
        return jsonify({"rewritten_prompt": ""})

    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 400

    # Extract Bengali text if not provided
    if bengali_texts is None:
        bengali_texts = extract_bengali_text(prompt)

    try:
        rewritten = rewrite_prompt(prompt, num_user_images, bengali_texts)
        return jsonify({
            "rewritten_prompt": rewritten,
            "bengali_texts": bengali_texts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/preview-prompt', methods=['POST'])
def preview_prompt():
    """Show the prompt that would be sent to OpenAI without calling the API."""
    data = request.json
    prompt = data.get('prompt', '')
    num_user_images = data.get('num_user_images', 0)

    if not prompt:
        return jsonify({"prompt": ""})

    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)

    # Try LLM rewrite if API key is available
    if OPENAI_API_KEY:
        try:
            rewritten = rewrite_prompt(prompt, num_user_images, bengali_texts)
            return jsonify({
                "original_prompt": prompt,
                "bengali_texts": bengali_texts,
                "openai_prompt": rewritten,
                "rewrite_method": "llm"
            })
        except Exception as e:
            print(f"LLM rewrite failed in preview: {e}")

    # Fallback: build prompt the old way
    cleaned_prompt = prompt
    for text in bengali_texts:
        cleaned_prompt = cleaned_prompt.replace(text, "[TEXT FROM IMAGE]")

    cleaned_prompt = re.sub(r'["\']?\[TEXT FROM IMAGE\]["\']?', '[TEXT FROM IMAGE]', cleaned_prompt)

    enhanced_prompt = cleaned_prompt + "\n\n"
    enhanced_prompt += "CRITICAL INSTRUCTION: I am providing a reference image containing a graphic design element. "
    enhanced_prompt += "Composite this graphic into the generated image exactly as-is — preserve its exact shapes. "
    enhanced_prompt += "Do not redraw, reinterpret, or alter the shapes. Place the graphic appropriately based on context (e.g., centered on a poster, top of a cover). "
    enhanced_prompt += "Blend the white background of the graphic seamlessly into the scene — treat the white area as transparent and integrate only the dark shapes into the composition."

    return jsonify({
        "original_prompt": prompt,
        "bengali_texts": bengali_texts,
        "openai_prompt": enhanced_prompt,
        "rewrite_method": "fallback"
    })


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    user_images_b64 = data.get('user_images', [])
    cached_rewritten_prompt = data.get('rewritten_prompt', None)
    mask = data.get('mask_text', False)

    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"})

    if not GEMINI_API_KEY:
        return jsonify({"success": False, "error": "GEMINI_API_KEY not set"})

    # Decode user-uploaded images
    user_images = []
    for b64_str in user_images_b64:
        # Strip data URL prefix if present
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        img_data = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_data)).convert('RGBA')
        user_images.append(img)

    # Extract Bengali text
    bengali_texts = extract_bengali_text(prompt)

    # Render Bengali text as images
    rendered_images = []
    rendered_b64 = []
    for text in bengali_texts:
        img = render_bengali_text(text, mask=mask)
        rendered_images.append(img)
        rendered_b64.append(image_to_base64(img))

    if not bengali_texts and not user_images:
        # No Bengali text and no user images - just do normal generation
        try:
            # Use cached prompt if available, otherwise rewrite
            if cached_rewritten_prompt:
                rewritten = cached_rewritten_prompt
            else:
                try:
                    rewritten = rewrite_prompt(prompt, 0, [])
                except Exception:
                    rewritten = prompt

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[rewritten],
            )

            result_b64 = None
            for part in response.parts:
                if part.inline_data is not None:
                    result_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    break

            return jsonify({
                "success": True,
                "image": result_b64,
                "prompt_used": rewritten,
                "rendered_texts": [],
                "note": "No Bengali text detected and no user images - used standard generation"
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # Generate with Bengali text references and user images
    result = generate_image(prompt, bengali_texts, rendered_images, user_images, rewritten_prompt=cached_rewritten_prompt)
    result["rendered_texts"] = rendered_b64

    return jsonify(result)


if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("="*60)
        print("WARNING: GEMINI_API_KEY environment variable not set!")
        print("Image generation will not work.")
        print("Set it with: export GEMINI_API_KEY='your-key'")
        print("="*60)
    if not OPENAI_API_KEY:
        print("="*60)
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Prompt rewriting will fall back to manual mode.")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print("="*60)

    print("\nStarting Bengali Image Generator...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

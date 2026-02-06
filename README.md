# Bengali Image Generator

AI image generators (ChatGPT, DALL-E, Midjourney, Gemini) cannot render Bengali text correctly. They mangle the script, especially conjunct characters (যুক্তাক্ষর).

This app works around that limitation:
1. You write a prompt containing Bengali text
2. The app extracts and renders the Bengali text as a PNG using proper fonts
3. The app sends the rendered image + rewritten prompt to Gemini
4. Gemini places your pre-rendered text into the generated image

**Live demo:** [your-railway-url.up.railway.app](https://your-railway-url.up.railway.app)

## Quick Start (Local)

```bash
# Install dependencies
brew install libraqm fribidi  # macOS - required for Bengali text shaping
pip install flask pillow openai google-genai

# Build Pillow from source with raqm support
pip install pillow --no-cache-dir --no-binary pillow

# Verify raqm works
python -c "from PIL import features; print(features.check('raqm'))"  # Should print True

# Run
export GEMINI_API_KEY='your-gemini-key'
export OPENAI_API_KEY='your-openai-key'  # for prompt rewriting
python app.py
```

Open http://localhost:5000

## Deploy to Railway

1. Fork this repo
2. Go to [railway.app](https://railway.app), sign in with GitHub
3. New Project → Deploy from GitHub Repo → select your fork
4. Add environment variables in the Variables tab:
   - `GEMINI_API_KEY` - your Gemini API key
   - `OPENAI_API_KEY` - your OpenAI API key
   - `ACCESS_CODE` - (optional) require this code to access the app
5. Settings → Networking → Generate Domain

Railway auto-deploys on every push to main.

## API Keys

| Key | Purpose | Get it from |
|-----|---------|-------------|
| `GEMINI_API_KEY` | Image generation | [Google AI Studio](https://aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | Prompt rewriting | [OpenAI Platform](https://platform.openai.com/api-keys) |
| `ACCESS_CODE` | (Optional) Gate access to the app | You choose |

## Code Structure

The entire app is in `app.py` (~1500 lines). Key sections:

| Lines | What it does |
|-------|--------------|
| 1-75 | Imports, config, font detection |
| 78-190 | `render_bengali_text()` - renders Bengali as PNG with optional masking |
| 192-275 | `rewrite_prompt()` - calls GPT-4o-mini to reframe prompt (strips Bengali, calls images "graphics") |
| 277-335 | `generate_image()` - sends images + prompt to Gemini |
| 338-1228 | `HTML_TEMPLATE` - the entire frontend (HTML/CSS/JS inline) |
| 1230-1320 | Flask routes: `/`, `/render`, `/generate`, `/preview-prompt` |

## Gemini API Reference

Image generation uses `google-genai` SDK:

```python
from google import genai
from google.genai.types import GenerateContentConfig, ImageConfig

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[pil_image1, pil_image2, "your prompt here"],
    config=GenerateContentConfig(
        image_config=ImageConfig(aspect_ratio="1:1")  # 1:1, 4:3, 3:4, 16:9, 9:16
    ),
)

# Extract generated image
for part in response.parts:
    if part.inline_data is not None:
        image_bytes = part.inline_data.data
        break
```

- **Model:** `gemini-2.5-flash-image`
- **Input:** Mix of PIL Images and text in the `contents` list
- **Aspect ratios:** `1:1` (default), `4:3`, `3:4`, `16:9`, `9:16`
- **Docs:** [Gemini Image Generation](https://ai.google.dev/gemini-api/docs/image-generation)

## Modifying the Code

**Change aspect ratio:**
Search for `aspect_ratio="1:1"` in `app.py` (appears twice in `generate_content` calls)

**Change the model:**
Search for `gemini-2.5-flash-image` and replace with another model ID

**Change the font:**
Put your `.ttf` file in the repo root and update `FONT_PATHS` list at the top of `app.py`

**Change the masking style:**
Edit `render_bengali_text()` around line 130 — the `mask=True` branch adds rectangles over text

**Change prompt rewriting:**
Edit `REWRITE_SYSTEM_PROMPT` (line ~192) or `rewrite_prompt()` function

**Disable access code:**
Don't set `ACCESS_CODE` env var, or remove the login logic from the `/` route

## Files

```
app.py                              # The entire application
Dockerfile                          # Railway/Docker deployment
requirements.txt                    # Python dependencies
NotoSansBengali-SemiBold.ttf        # Bengali font (required)
CLAUDE.md                           # Project context for Claude Code
JOURNAL.md                          # Development notes
```

## How the Text Masking Works

When "Mask text" is enabled (default), the app overlays semi-transparent rectangles on the rendered Bengali text. This prevents Gemini from recognizing it as text and trying to re-render it (which would produce garbled Bengali).

Without masking, short text (1-2 words) usually works. Longer text benefits from masking.

## License

MIT

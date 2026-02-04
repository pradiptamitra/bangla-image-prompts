FROM python:3.12-slim

# Install system dependencies for libraqm (Bengali text shaping)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libraqm-dev \
        libfribidi-dev \
        libharfbuzz-dev \
        libfreetype6-dev \
        build-essential \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python deps — build Pillow from source so it links against libraqm
RUN pip install --no-cache-dir gunicorn flask openai google-genai && \
    pip install --no-cache-dir --no-binary pillow pillow

# Verify raqm support
RUN python -c "from PIL import features; assert features.check('raqm'), 'raqm not available!'"

COPY . .

# Railway sets PORT env var
ENV PORT=5000
EXPOSE ${PORT}

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 app:app

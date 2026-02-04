FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Official Pillow Linux wheels bundle libraqm, so no source build needed
RUN pip install --no-cache-dir -r requirements.txt

# Verify raqm support (bundled in Pillow's manylinux wheel)
RUN python -c "from PIL import features; assert features.check('raqm'), 'raqm not available!'"

COPY . .

# Railway sets PORT env var
ENV PORT=5000
EXPOSE ${PORT}

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 app:app

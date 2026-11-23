# SnapShop API image (CPU). The code and libraries are baked in; the big files (weights in runs/,
# embeddings + catalogue photos in data/processed/) are mounted at run time, so the image stays small
# and the same image works with retrained models.
#
#   docker build -t snapshop-api .
#   docker run --rm -p 8000:8000 -v "%cd%\runs:/app/runs:ro" -v "%cd%\data\processed:/app/data/processed:ro" snapshop-api
#   (macOS/Linux: -v "$PWD/runs:/app/runs:ro" -v "$PWD/data/processed:/app/data/processed:ro")
#   then open http://127.0.0.1:8000/docs

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=1

# OpenCV (pulled in by Ultralytics) needs these system libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# run as a normal user, not root
RUN useradd --create-home app
WORKDIR /app

# layers that change rarely go first, so code edits don't re-install everything
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

USER app
# our ResNet-50s start from torchvision's ImageNet weights when they are built: download them once,
# into the image, instead of on every container start
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"

COPY --chown=app:app src/ src/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

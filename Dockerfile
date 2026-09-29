FROM python:3.11-slim

LABEL org.opencontainers.image.title="ImageCraft" \
      org.opencontainers.image.description="Text-to-image synthesis with a GloVe-conditioned DCGAN (Oxford-102 Flowers)" \
      org.opencontainers.image.source="https://github.com/daivikrao/imagecraft-text-to-image-gan"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    FLOWERS_ROOT=/data/flowers \
    GLOVE_PATH=/data/glove.6B.300d.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Mount the dataset, GloVe file and checkpoints at /data, then run any script:
#   docker run -v "$PWD:/data" ghcr.io/daivikrao/imagecraft-text-to-image-gan src/train.py --epochs 500
ENTRYPOINT ["python"]
CMD ["src/generate.py", "--help"]

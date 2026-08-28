# CPU-first runtime image for the SAM2 contour service.
#
# The model checkpoint is deliberately not baked into this image: mount the
# downloaded tiny checkpoint at /models/sam2.1_hiera_tiny.pt at runtime.
FROM python:3.11-slim-bookworm

ARG SAM2_REPOSITORY=https://github.com/facebookresearch/sam2.git
ARG SAM2_REVISION=2b90b9f5ceec907a1c18123530e92e794ad901a4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/sam2 \
    SAM2_BUILD_CUDA=0

# libGL/libgomp are needed by OpenCV. git is only used while building the image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch before SAM2 so setup.py never attempts a CUDA build.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
    && python -m pip install \
        Flask==3.1.0 \
        gunicorn==23.0.0 \
        opencv-python-headless==4.10.0.84

# Pin SAM2 to a commit so a rebuild does not silently receive a different
# upstream implementation. Change SAM2_REVISION deliberately when upgrading.
RUN git clone "${SAM2_REPOSITORY}" /opt/sam2 \
    && git -C /opt/sam2 checkout --detach "${SAM2_REVISION}" \
    && SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation /opt/sam2 \
    && rm -rf /opt/sam2/.git

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir --parents /app/data \
    && chown --recursive app:app /app

WORKDIR /app
COPY --chown=app:app app.py contours.py grounding.py ./
COPY --chown=app:app templates ./templates
COPY --chown=app:app static ./static

USER app
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "--worker-class", "gthread", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:8000", "--timeout", "300", "--graceful-timeout", "30", "--keep-alive", "5", "--worker-tmp-dir", "/dev/shm", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

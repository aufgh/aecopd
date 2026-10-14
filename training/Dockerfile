FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime@sha256:c263514402c85b4203311883fbef75cc48c4dcc01bbc65bcd2435348788a34e7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN python -m pip install --break-system-packages --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "matplotlib==3.11.2" \
    "tensorboard==2.21.0"

CMD ["bash"]

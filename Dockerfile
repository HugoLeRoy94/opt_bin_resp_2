# torch, CUDA and cuDNN come from the base image, which is why they are not in
# requirements.txt: installing torch from pip here would shadow the tuned build.
FROM nvcr.io/nvidia/pytorch:24.01-py3

# UID 1000 matches the host user, so files a run writes are not root-owned.
RUN groupadd -g 1000 devuser && \
    useradd -u 1000 -g 1000 -m -s /bin/bash devuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The source is mounted at run time, not copied, so edits take effect without a
# rebuild. That means the package cannot be pip-installed at build time either,
# and PYTHONPATH is what makes `import model` resolve from any directory.
ENV PYTHONPATH=/app

CMD ["python3", "-c", "import model, estimate, train, analysis; print('ready')"]

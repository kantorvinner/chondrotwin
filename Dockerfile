FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/chondrotwin

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY chondrotwin ./chondrotwin
COPY recipes ./recipes
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "chondrotwin.gallery.train"]
CMD ["+experiment=main"]

# Use base shared Python image
FROM python:3.9

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    netcat-openbsd \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils\
    python3-opencv \
    tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade pip setuptools wheel

COPY . /src

WORKDIR /src

# RUN pip install --no-cache-dir -r /src/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


ARG APP_VERSION=0.0.0
ENV APP_VERSION=$APP_VERSION
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore::DeprecationWarning"

RUN chmod +x /src/start.sh
CMD ["/src/start.sh"]
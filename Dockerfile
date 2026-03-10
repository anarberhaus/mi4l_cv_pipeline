# 1. Use an official Python 3.10 image (Stable for Mediapipe 0.10.32)
FROM python:3.10-slim

# 2. Install essential Linux system libraries for CV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up the working directory
WORKDIR /app

# 4. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 6. Set PYTHONPATH so 'scripts/run_mi4l.py' can find the 'src' folder
ENV PYTHONPATH="/app:/app/src"

# 7. Start Streamlit (Render provides the $PORT)
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

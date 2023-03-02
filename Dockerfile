# Get Fast API image with Python version 3.9
FROM tiangolo/uvicorn-gunicorn:python3.9

# Create container dir
WORKDIR /app
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api_template.py ./

# Copy models and embeddings
COPY ./final_models/ ./final_models/
COPY ./image_embeddings.json ./image_embeddings.json
COPY ./image_cnn.py ./image_cnn.py
COPY ./image_processor.py ./image_processor.py
COPY ./clean_data.py ./clean_data.py
COPY ./config.py ./config.py
COPY ./data/Labels.csv ./data/Labels.csv

ENTRYPOINT ["uvicorn", "api_template:api", "--host", "0.0.0.0", "--port", "80"]
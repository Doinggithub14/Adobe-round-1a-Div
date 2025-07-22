FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Copy and install only needed libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Run the main prediction script
CMD ["python", "main.py"]
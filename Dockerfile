FROM fw407/2.6-r36.4.3-cu126-22.04

WORKDIR /app

COPY requirements-jetson.txt /app/
RUN pip install --no-cache-dir -r requirements-jetson.txt


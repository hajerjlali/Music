FROM tiangolo/uwsgi-nginx-flask:python3.9

EXPOSE 5000
# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --upgrade numpy
RUN pip install pytest
# URL under which static (not modified by Python) files will be requested
# They will be served by Nginx directly, without being handled by uWSGI
ENV STATIC_URL /static
# Absolute path in where the static files wil be
ENV STATIC_PATH /app/static



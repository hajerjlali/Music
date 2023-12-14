FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7
RUN apk --update add bash nano
ENV apk --update add bash nano 
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/TestApp/app/static
COPY ./requirements.txt /var/www/TestApp/app/requirements.txt
RUN pip install -r /var/www/TestApp/app/requirements.txt

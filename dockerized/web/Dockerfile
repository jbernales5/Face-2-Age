FROM nginx:latest

WORKDIR /etc/nginx

RUN mkdir sites-available
COPY default.conf /etc/nginx/conf.d
# RUN ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled

FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080/tcp

# CMD [ "python", "./server.py" ]
CMD uwsgi --socket 0.0.0.0:8080 --protocol=http -w wsgi:app

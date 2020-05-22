FROM python:3.7-slim
COPY server.py /deploy/src/
COPY requirements.txt /deploy/src

COPY trained_models/real_age/age.prototxt /deploy/models/trained_models/real_age/age.prototxt
COPY trained_models/real_age/dex_imdb_wiki.caffemodel /deploy/models/trained_models/real_age/dex_imdb_wiki.caffemodel

COPY trained_models/apparent_age/apparent_age.prototxt /deploy/models/trained_models/apparent_age/apparent_age.prototxt
COPY trained_models/apparent_age/dex_chalearn_iccv2015.caffemodel /deploy/models/trained_models/apparent_age/dex_chalearn_iccv2015.caffemodel

COPY trained_models/gender/gender.prototxt /deploy/models/trained_models/gender/gender.prototxt
COPY trained_models/gender/gender.caffemodel /deploy/models/trained_models/gender/gender.caffemodel

WORKDIR /deploy/src/

RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "server.py"]


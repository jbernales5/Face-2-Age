# How to use the container version of this project

Download the "Real age" and "Gender" caffemodel files from this link: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Save the files in the "back" directory as dex_imdb_wiki.caffemodel, and gender.caffemodel.

Update the back/default.conf file and set the appropriate Access-Control-Allow-Origin header

Run "docker-compose up" and you're good to go!

Hit your API as: localhost:8080/predict and send an "image" parameter with the image content.

For example, with curl:

curl -X POST \
  localhost:8080/predict \
  -F "image=@/Users/jonathan/Desktop/foto.jpg"

# How to use the container version of this project

Run the downloadModels.sh script, it will download the appropriate files in the back-end directory (don't forget to set the appropriate privileges).

Update the back/default.conf file and set the appropriate Access-Control-Allow-Origin header

Run "docker-compose up" and you're good to go!

Hit your API as: localhost:8080/predict and send an "image" parameter with the image content.

For example, with curl:

curl -X POST \
  localhost:8080/predict \
  -F "image=@/Users/jonathan/Desktop/foto.jpg"

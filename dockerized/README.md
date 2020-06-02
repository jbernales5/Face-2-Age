# How to use the container version of this project

Run the downloadModels.sh script, it will download the appropriate files in the back-end directory (don't forget to set the appropriate privileges).

##HTTP only
Update the back/default.conf file and set the appropriate Access-Control-Allow-Origin header and server_name

Run "docker-compose up" and you're good to go!

Hit your API as: localhost:8080/predict and send an "image" parameter with the image content.

For example, with curl:

curl -X POST \
  localhost:8080/predict \
  -F "image=@<your/image/path>"

##Enabling HTTPS
If you wish to enable HTTPS to your server, please run docker-compose up and wait until the containers are up and running.

Connect to the container running the dockerized_front image with the following command:
* docker exec -it <container_id> /bin/bash
* Run "apt-get update" and then run "apt-get install certbot python-certbot-nginx".
* Once the process is complete, run "certbot --nginx" and follow the steps. Be sure that you have a valid domain pointing to your server's IP address.

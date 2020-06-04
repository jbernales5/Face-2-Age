# Face to Age

Hello there! Welcome to the Face2Age repository. Please

# Running your very own Face2Age locally
## Requirements
In order to run the servers locally you'll need:

 - Docker: https://docs.docker.com/engine/install/
 - Docker-Compose: https://docs.docker.com/compose/install/
 - NPM: https://www.npmjs.com/get-npm

... And that's it ! You're good to go 🥳

## Building the back-end with Docker 🐋

 1. Go to the dockerized directory and execute the downloadModels script
 2. Then, feel free to follow the dedicated instructions from the ReadMe file 📘 in that directory 

## Building the front-end with Angular 🅰️
Simply go to the front-end directory and execute the following commands:

    npm install
 
Once installed, you should edit the app.component.ts which is located in src/app:

    export class AppComponent {
	  faHeart = faHeart;
	  faAngular = faAngular;
	  faAWS = faAws;
	  faEnvelope = faEnvelope;

	  apiUrl = '<your_server_ip>/predict'; //<-- Change THIS to "http://localhost/predict"
	 }

Then, you're ready to run your server and start guessing ages 👶👴! Simply run:

    ng serve -o

Your web server should be up and running at the address localhost:42000 🕺

# Sources 🎓

 - Model: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
 - Front-end dependencies:
	 - Angular Material: http://material.angular.io
	 - Sweet Alert 2: https://sweetalert2.github.io
	 - NGX Dropzone: https://www.npmjs.com/package/ngx-dropzone

import { Component, ElementRef } from '@angular/core';
import { faBrain } from '@fortawesome/free-solid-svg-icons';
import { HttpClient } from '@angular/common/http';
import { Observable, Observer, ReplaySubject } from 'rxjs';
import Swal, { SweetAlertIcon } from 'sweetalert2';
import { NgxSpinnerService } from 'ngx-spinner';

import { faHeart, faEnvelope } from '@fortawesome/free-solid-svg-icons';
import { faAngular, faAws } from '@fortawesome/free-brands-svg-icons';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent {
  faHeart = faHeart;
  faAngular = faAngular;
  faAWS = faAws;
  faEnvelope = faEnvelope;

  apiUrl = '<your_server_ip>/predict';

  // LOCAL TESTING
  // apiUrl = 'http://127.0.0.1/predict';

  title = 'Face to Age!';

  files: File[] = [];
  faBrain = faBrain;
  loading = false;

  constructor(private httpClient: HttpClient, private spinner: NgxSpinnerService, private elementRef: ElementRef) { }

  ngAfterViewInit() {
    this.elementRef.nativeElement.ownerDocument.body.style.backgroundColor = '#F2F3F3';
  }

  onSelect(event) {
    this.files.push(...event.addedFiles);
  }

  onRemove(event) {
    this.files.splice(this.files.indexOf(event), 1);
  }

  uploadFiles() {
    const formData = new FormData();
    formData.append('image', this.files[0]);
    this.loading = true;
    this.spinner.show();
    this.httpClient.post(this.apiUrl, formData).subscribe(
      (response) => {
        this.loading = false;
        this.spinner.hide();
        const aparentAge = response['apparent_age'];
        const gender = response['gender'];
        if (response.hasOwnProperty('apparent_age')) {
          this.readImage(this.files[0]).subscribe(result => {
            const tmpResult = String(result);
            this.launchImageSWAL(
              'You look ' + aparentAge + '!',
              'You are a ' + gender + ', and our super AI believes you look like ' + aparentAge,
              tmpResult).then(() => {
              this.files.splice(0);
            });
          });
        } else {
          this.readImage(this.files[0]).subscribe(result => {
            const tmpResult = String(result);
            this.launchImageSWAL(
              'You don\'t look human!',
              'We could not recognize your face from the provided picture',
              tmpResult).then(() => {
              this.files.splice(0);
            });
          });
        }
      },
      (error) => {
        this.loading = false;
        this.spinner.hide();
        this.launchSWAL(
          'Error!',
          'Something wrong happened, maybe the server is down!',
          'error',
          'Ok :('
        );
      }
    );
  }

  readImage(file: File): Observable<string | ArrayBuffer> {
    const fileReader = new FileReader();
    const urlObservable = new ReplaySubject<string | ArrayBuffer>(1);
    fileReader.onload = event => {
      urlObservable.next(fileReader.result);
    };
    fileReader.readAsDataURL(this.files[0]); // read file as data url
    return urlObservable;
  }

  launchSWAL(title2: string, text2: string, icon: SweetAlertIcon, confirmButton2: string) {
    return Swal.fire({
      title: title2,
      text: text2,
      icon: icon,
      showCancelButton: false,
      confirmButtonText: confirmButton2,
      cancelButtonText: 'No'
    });
  }

  launchImageSWAL(myTitle: string, myText: string, myImageUrl: string) {
    return Swal.fire({
      title: myTitle,
      text: myText,
      imageUrl: myImageUrl,
      imageWidth: 500
    });
  }

}

import { Component } from '@angular/core';
import { faBrain } from '@fortawesome/free-solid-svg-icons';
import { HttpClient } from '@angular/common/http';
import { Observable, Observer, ReplaySubject } from 'rxjs';
import Swal, { SweetAlertIcon } from 'sweetalert2';
import { NgxSpinnerService } from 'ngx-spinner';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent {

  constructor(private httpClient: HttpClient, private spinner: NgxSpinnerService) { }

  apiUrl = 'http://<your-server-ip-address>/predict';

  // LOCAL TESTING
  // apiUrl = 'http://127.0.0.1/predict';

  title = 'age-guess';

  files: File[] = [];
  faBrain = faBrain;
  loading = false;

  onSelect(event) {
    console.log(event);
    this.files.push(...event.addedFiles);
  }

  onRemove(event) {
    console.log(event);
    this.files.splice(this.files.indexOf(event), 1);
  }

  uploadFiles() {
    const formData = new FormData();
    formData.append('image', this.files[0]);
    this.loading = true;
    this.spinner.show();
    this.httpClient.post(this.apiUrl, formData).subscribe(
      (response) => {
        console.log(response);
        this.loading = false;
        this.spinner.hide();
        const aparentAge = response['real_age'];
        const gender = response['gender'];
        this.readImage(this.files[0]).subscribe(result => {
          const tmpResult = String(result);
          Swal.fire({
            title: 'You look ' + aparentAge + '!',
            text: 'You are a ' + gender + ', and our super AI believes you look like ' + aparentAge,
            imageUrl: tmpResult,
            imageWidth: 500,
            imageHeight: 400
          }).then((result) => {
            this.files.splice(0);
          });
        });
      },
      (error) => {
        this.loading = false;
        this.spinner.hide();
        console.log(error);
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

}

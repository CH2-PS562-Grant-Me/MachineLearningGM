# Documentation

## data_awardee_topsis.csv
Terdiri dari 120+ record dengan 10 attribute dtype float, yaitu:
> - IPK : nilai IPK awardee terakhir
> - Sertifikasi : jumlah sertifikasi general yang dimiliki awardee
> - Sertifikasi Professional : jumlah sertifikasi professional yang dimiliki awardee
> - Prestasi Nasional : jumlah prestasi skala nasional (include runner up dalam kompetisi)
> - Kompetisi Top 3 Nasional : jumlah kompetisi/turnamen skala nasional dan dapat posisi top 3
> - Prestasi Internasional : jumlah prestasi skala internasional (include runner up dalam kompetisi)
> - Kompetisi Top 3 Internasional : jumlah kompetisi/turnamen skala internasional dan dapat posisi top 3
> - Intern : durasi (bulan) intern/magang yang telah dijalani
> - Volunteer : durasi (bulan) volunteer/kepanitiaan yang telah dijalani

## grant_me_model.py
Model dibuat menggunakan TensorFlow dengan 2 layer dense, Adam optimizer, dan Huber loss parameter. Merupakan regression model, oleh karena itu metrik MAE digunakan dengan nilai minimal MAE < 0.05. Training menggunakan data_awardee_topsis.csv.

Contoh input untuk predict : [[3.83, 0., 0., 2., 3., 0., 0., 3., 14.]]
Index kiri sampai ke kanan mengikuti urutan data awardee dari section sebelumnya, dari atas sampai bawah.
> ### input akan di convert dari json array ke list in list seperti di atas di api.py

## model_topsis.h5
Model yang telah di train oleh grant_me_model.py dan disimpan dengan format .h5. Model akan digunakan oleh api.py

## api.py
API di buat dengan menggunakan framework Flask dan metode POST. Bisa di test di postman dengan sample dari contoh_input.txt
> :warning: **Model Path**: wajib diganti mengikuti path yang kalian gunakan

## contoh_input.txt
Contoh input berupa .json array dengan value dtype float yang dipakai untuk test API model. Contoh:
> {
>  "IPK": 3.83,
>  "Sertifikasi": 0,
>  "Sertifikasi Professional": 0,
>  "Prestasi Nasional": 2,
>  "Kompetisi Top 3 Nasional": 3,
>  "Prestasi Internasional": 0,
>  "Kompetisi Top 3 Internasional": 0,
>  "Intern": 3,
>  "Volunteer": 14
> }

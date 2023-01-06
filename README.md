---
title: Indo Twitter Sentiment App
emoji: 👀
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.15.2
app_file: app.py
pinned: false
---

# Twitter Sentiment Analysis App

Halo Semuanya, ini adalah salah satu mini projek saya mengenai NLP(Natural Language Processing), mini project ini berkaitan dengan machine learning dan deep learning, disini saya mencoba untuk melakukan deployment model dan juga mengatur demo agar user diluar bidang ini dapat menggunakan model yang telah saya buat.
Mini projek ini berupa aplikasi untuk meprediksi sentimen pada postingan di media twitter. Aplikasi ini akan melakukan scraping data di media sosial twitter berdasarkan tag atau akun tertentu sesuai input dari user. Data yang diperoleh kemudian dilakukan analisis berupa analisis sentimen. 

Pada aplikasi ini saya menggunakan 2 metode utama, yakni text classification dan juga topic modelling. Metode text classification saya gunakan untuk mengklasifikasikan apakah tweet tersebut berisi sentimen positif, negatif, atau netral. Metode kedua, yakni topic modelling saya gunakan untuk mengesktrak atau mengetahui topik apa saja yang dibahas pada setiap sentimen positif, negatif dan netral. Untuk pembahasan mengenai metode yang saya gunakan, ada pada section dibawah.

Demo aplikasi : https://huggingface.co/spaces/dafqi/indo_twitter_sentiment_app 

## bias dan limitasi

Menurut saya model untuk sentimen analisis lebih baik dikhususkan untuk domain tertentu, dengan kata lain jika ingin menganalisis di bidang finance, maka gunakan model yang memang dilatih pada data yang berisi tweet finance dengan begitu model akan lebih representatif. Pada aplikasi ini model klasifikasi dilatih menggunakan data dengan domain yang lebih umum, seperti review aplikasi dan juga komentar tentang event tertentu. Oleh karena itu, jika topik yang dibahas diluar itu, mungkin saja hasilnya kurang representatif. 
Keterbatasan dari aplikasi ini adalah hanya bisa melakukan analisis tweet yang berbahasa indonesia, hal ini dikarenakan keterbatasan storage yang digunakan untuk menyimpan model dan juga dataset yang diperoleh belum mencakup bahasa selain indonesia.  

## Text Classification

Model yang digunakan : Indobert-lite-base-p1
Dataset yang digunakan : data tweet IndoNLU  SMSA 
Model benchmark : 
Gambar benchmar dari indoNLU
Detail mengenai proses training dan pembuatan model ada pada notebook/text_classification.ipynb   

## Topic Modelling 

Algoritma : Bertopic
Sentence_transformers model : indobert-sentence-transformers
Model lainnya menggunakan default model pada bertopic



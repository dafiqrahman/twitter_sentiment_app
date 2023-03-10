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

library yang digunakan:  <br>
modelling : transformers, bertopic, scikit-learn <br>
visualisasi : plotly, wordcloud <br>
deployment : streamlit <br>

<b> important note : Jika ingin mengcopy/menggunakan versi aplikasi serta file aplikasi terbaru silahkan menggunakan repo pada link <a href = "https://huggingface.co/spaces/dafqi/indo_twitter_sentiment_app "> demo aplikasi </a> </b>



## bias dan limitasi

Menurut saya model untuk sentimen analisis lebih baik dikhususkan untuk domain tertentu, dengan kata lain jika ingin menganalisis di bidang finance, maka gunakan model yang memang dilatih pada data yang berisi tweet finance dengan begitu model akan lebih representatif. Pada aplikasi ini model klasifikasi dilatih menggunakan data dengan domain yang lebih umum, seperti review aplikasi dan juga komentar tentang event tertentu. Oleh karena itu, jika topik yang dibahas diluar itu, mungkin saja hasilnya kurang representatif. <br>
Keterbatasan dari aplikasi ini adalah <b> hanya bisa melakukan analisis tweet yang berbahasa indonesia </b>, hal ini dikarenakan keterbatasan storage yang digunakan untuk menyimpan model dan juga dataset yang diperoleh belum mencakup bahasa selain indonesia.  

## Text Classification

Model yang digunakan : Indobert-lite-base-p1 <br>
Dataset yang digunakan : data tweet IndoNLU  SMSA <br> 
Model benchmark in validation subset :  <br>
<img src="assets/img/model_benchmark.png" /> <br>
Detail mengenai proses training dan pembuatan model ada pada notebook/text_classification.ipynb   

## Topic Modelling 

Algoritma : Bertopic <br>
Sentence_transformers model : indobert-sentence-transformers <br>
Model lainnya menggunakan default model pada bertopic

## Preview Aplikasi

<img src="assets/img/showcase_1.png" width = "512"/> <br>
<img src="assets/img/showcase_2.png" width = "512" /> <br>


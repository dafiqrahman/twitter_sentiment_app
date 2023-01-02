import streamlit as st 
import pandas as pd
import script.functions as fn
import plotly.express as px
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
# import text_proc in script folder
import script.text_proc as tp

# Load data
# add tiwtter logo inside title 
st.markdown("<h1 style='text-align: center; color: black;'>🗨️Twitter Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.write("Aplikasi sederhana untuk melakukan analisis sentimen terhadap tweet yang diinputkan dan mengekstrak topik dari setiap sentimen")

# streamlit selectbox simple and advanced

sb1,sb2 = st.columns([1,4])
with sb1:
    option = st.selectbox('Pilih Mode Pencarian',('Simple','Advanced'))

if option == 'Simple':
# create col1 and col2
    col1, col2 = st.columns([3,2])
    with col1:
        input = st.text_input("Masukkan User/Hastag", "@BPJSKesehatanRI")
    with col2:
        length = st.number_input("Jumlah Tweet", 10, 500, 100)
else :
    col1, col2 = st.columns([3,1])
    with col1:
        input = st.text_input("Masukkan Parameter Pencarian", "(@undipmenfess AND @BPJSKesehatanRI) -filter:links filter:replies")
    with col2:
        length = st.number_input("Jumlah Tweet", 10, 500, 100)
    st.caption("anda bisa menggunakan parameter pencarian yang lebih spesifik, parameter ini sama dengan paremeter pencarian di twitter")

submit = st.button("🔍Cari Tweet")

st.caption("semakin banyak tweet yang diambil maka semakin lama proses analisis sentimen")

if submit:
    with st.spinner('Mengambil data dari twitter... (1/2)'):
        df = fn.get_tweets(input, length, option)
    with st.spinner('Melakukan Prediksi Sentimen... (2/2)'):
        df = fn.get_sentiment(df)
        df.to_csv('assets/data.csv',index=False)
    # plot
    st.write("<b>Preview Dataset</b>",unsafe_allow_html=True)
    st.dataframe(df,use_container_width=True,height = 200)
    st.write ("Jumlah Tweet: ",df.shape[0])
    # download datasets 
    

    st.write("<h3>📊 Analisis Sentimen</h3>",unsafe_allow_html=True)
    col_fig1, col_fig2 = st.columns([4,3])
    with col_fig1:
         with st.spinner('Sedang Membuat Grafik...'):
            st.write("<b>Jumlah Tweet Tiap Sentiment</b>",unsafe_allow_html=True)
            fig_1 = fn.get_bar_chart(df)
            st.plotly_chart(fig_1,use_container_width=True,theme="streamlit")
    with col_fig2:
        st.write("<b>Wordcloud Tiap Sentiment</b>",unsafe_allow_html=True)
        tab1,tab2,tab3 = st.tabs(["negatif","netral","positif"])
        with tab1:
            wordcloud_pos = tp.get_wordcloud(df,"negatif")
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_pos, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
        with tab2:
            wordcloud_neg = tp.get_wordcloud(df,"netral")
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_neg, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
        with tab3:
            wordcloud_net = tp.get_wordcloud(df,"positif")
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_net, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
    st.write("<h3>✨ Sentiment Clustering</h3>",unsafe_allow_html=True)
    @st.experimental_singleton
    def load_sentence_model():
        embedding_model = SentenceTransformer("sentence_bert")
        return embedding_model
    embedding_model = load_sentence_model()
    tab4,tab5,tab6 = st.tabs(["Negatif","Netral","Positif"])
    with tab4:
        if len(df[df["sentiment"]=="negatif"]) < 5:
            st.write("Tweet Terlalu Sedikit, Tidak dapat melakukan clustering")
        else:
            with st.spinner('Sedang Membuat Grafik...'):
                text,data,fig = tp.plot_text(df,"negatif",embedding_model)
                st.plotly_chart(fig,use_container_width=True,theme=None)
            fig,topic_modelling = tp.topic_modelling(text,data)
            st.plotly_chart(fig,use_container_width=True,theme="streamlit")
    with tab5:
        if len(df[df["sentiment"]=="netral"]) < 5:
            st.write("Tweet Terlalu Sedikit, Tidak dapat melakukan clustering")
        else:
            with st.spinner('Sedang Membuat Grafik...'):
                text,data,fig = tp.plot_text(df,"netral",embedding_model)
                st.plotly_chart(fig,use_container_width=True,theme=None)
            fig,topic_modelling = tp.topic_modelling(text,data)
            st.plotly_chart(fig,use_container_width=True,theme="streamlit")
    with tab6:
        if len(df[df["sentiment"]=="positif"]) < 5:
            st.write("Tweet Terlalu Sedikit, Tidak dapat melakukan clustering")
        else:
            with st.spinner('Sedang Membuat Grafik...'):
                text,data,fig = tp.plot_text(df,"positif",embedding_model)
                st.plotly_chart(fig,use_container_width=True,theme=None)
            fig,topic_modelling = tp.topic_modelling(text,data)
            st.plotly_chart(fig,use_container_width=True,theme="streamlit")
    
            
    


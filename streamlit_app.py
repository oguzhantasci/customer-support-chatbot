import streamlit as st
import requests

st.title("💳 AI Banking Chatbot")

customer_id = st.text_input("Müşteri ID'nizi girin:")
user_input = st.text_area("Mesajınızı yazın:")

if st.button("Gönder"):
    if not customer_id.isdigit():
        st.error("Lütfen geçerli bir müşteri ID girin!")
    else:
        response = requests.post("http://127.0.0.1:8000/chat/", json={"customer_id": int(customer_id), "message": user_input})
        bot_response = response.json().get("response", "❌ Hata oluştu.")
        st.text_area("💬 Bot Yanıtı:", bot_response, height=100)

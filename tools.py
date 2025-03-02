from langchain_core.tools import tool
import pandas as pd
import re
from config import df
from langchain_openai import ChatOpenAI
import datetime


def classify_intent(text: str) -> str:
    """AI-driven intent classification using an LLM with explanations for each intent."""
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = llm.invoke(
        f"Bu mesajın niyetini belirle: {text}. "
        "Aşağıdaki niyetlerden biriyle yanıt ver ve bir cümlede açıkla:\n"
        "BALANCE_INQUIRY: Kullanıcı hesap bakiyesini öğrenmek istiyor.\n"
        "TRANSACTION_REQUEST: Kullanıcı para transferi yapmak istiyor.\n"
        "TRANSACTION_HISTORY: Kullanıcı geçmiş işlemlerini görmek istiyor.\n"
        "COMPLAINT: Kullanıcı bir sorun veya şikayet bildirmek istiyor.\n"
        "GENERAL_QUERY: Kullanıcı genel bir soru soruyor.\n"
        "Yanıt şu formatta olmalıdır: \"NİYET - Açıklama\"."
    )
    intent_response = response.content.strip()

    if " - " in intent_response:
        intent, explanation = intent_response.split(" - ", 1)
    else:
        intent, explanation = "GENERAL_QUERY", "Kullanıcı genel bir soru sormaktadır."

    if intent not in ["BALANCE_INQUIRY", "TRANSACTION_REQUEST", "COMPLAINT", "GENERAL_QUERY", "TRANSACTION_HISTORY"]:
        return "GENERAL_QUERY"

    print(f"🛠️ Tespit Edilen Niyet: {intent} - {explanation}")  # Debugging log
    return intent


@tool
def get_balance(customer_id: int) -> str:
    """Returns the account balance for the given customer ID in a structured format."""
    customer = df[df['CustomerId'] == customer_id]
    if customer.empty:
        return "❌ Hata: Müşteri bulunamadı. Lütfen geçerli bir müşteri numarası girin."

    balance = customer.iloc[0]['Balance']
    return f"💳 Hesap Bakiyeniz: ${balance:,.2f} USD\n📅 Sorgulama Tarihi: Güncel.\nℹ️ Daha fazla bilgi için müşteri hizmetleriyle iletişime geçebilirsiniz."


@tool
def process_transaction(sender_customer_id: int, receiver_customer_id: int, amount: float) -> str:
    """Processes a money transfer between two customers and returns a structured confirmation."""
    sender = df[df['CustomerId'] == sender_customer_id]
    receiver = df[df['CustomerId'] == receiver_customer_id]

    if sender.empty or receiver.empty:
        return "❌ Hata: Gönderici veya alıcı bulunamadı. Lütfen bilgilerinizi kontrol edin."

    if sender.iloc[0]['Balance'] < amount:
        return "⚠️ İşlem Başarısız: Yetersiz bakiye. Lütfen bakiyenizi kontrol ediniz."

    # ✅ Update balances
    df.at[sender.index[0], 'Balance'] -= amount
    df.at[receiver.index[0], 'Balance'] += amount
    df.to_csv("Customer-Churn-Records.csv", index=False)

    # ✅ FIX: Call log_transaction using invoke()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_transaction.invoke({
        "customer_id": sender_customer_id,
        "transaction_details": f"[{timestamp}] {amount} USD gönderildi → {receiver_customer_id}"
    })

    return f"""✅ İşlem Başarılı!
💸 Transfer Edilen Tutar: ${amount:,.2f} USD
📤 Gönderici ID: {sender_customer_id}
📥 Alıcı ID: {receiver_customer_id}
📅 İşlem Tarihi: Güncel.
🔔 İşlem detayları için hesap ekstrenizi kontrol edebilirsiniz."""


@tool
def log_transaction(customer_id: int, transaction_details: str) -> str:
    """Logs a customer's transaction with a timestamp."""
    with open("transactions_log.txt", "a", encoding="utf-8") as file:
        file.write(f"Müşteri ID: {customer_id} - {transaction_details}\n")
    return "✅ İşlem kaydedildi."

@tool
def get_transaction_history(customer_id: int) -> str:
    """Retrieves a customer's transaction history from the log file."""
    try:
        with open("transactions_log.txt", "r", encoding="utf-8") as file:
            transactions = [line.strip() for line in file.readlines() if line.startswith(f"Müşteri ID: {customer_id}")]
            if not transactions:
                return "📜 Henüz işlem geçmişiniz bulunmamaktadır."
            return "📜 İşlem Geçmişiniz:\n" + "\n".join(transactions[-5:])  # Show last 5 transactions
    except FileNotFoundError:
        return "❌ İşlem geçmişi bulunamadı."

@tool
def log_complaint(customer_id: int, complaint_text: str) -> str:
    """Logs a customer's complaint and provides a confirmation response."""
    with open("complaints_log.txt", "a", encoding="utf-8") as file:
        file.write(f"Müşteri ID: {customer_id} - Şikayet: {complaint_text}\n")
    return "📩 Şikayetiniz başarıyla kaydedildi. Yetkililerimiz en kısa sürede sizinle iletişime geçecektir."

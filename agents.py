from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from tools import get_balance, process_transaction, log_complaint, log_transaction, get_transaction_history
import datetime

def financial_agent(state):
    """Handles balance inquiries and returns the correct balance."""
    customer_id = state.get("customer_id", None)

    if not customer_id:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Lütfen müşteri numaranızı belirtin.")],
            "next": "END"
        }

    # ✅ Retrieve balance from `get_balance`
    balance_response = get_balance.invoke({"customer_id": customer_id})

    # ✅ Ensure AIMessage response is correctly returned
    return {
        "messages": state["messages"] + [AIMessage(content=f"💳 {balance_response}")],
        "next": "END"
    }


def transaction_agent(state):
    """Handles transaction requests using AI to extract details dynamically and logs transactions."""
    last_message = state["messages"][-1].content
    sender_customer_id = state.get("customer_id", None)

    if not sender_customer_id:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Lütfen müşteri numaranızı belirtin.")],
            "next": "END"
        }

    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = llm.invoke(
        f"Kullanıcının mesajındaki işlem detaylarını çıkar: {last_message}. "
        "Yanıt şu formatta olmalıdır: 'AMOUNT: <tutar>, RECEIVER_ID: <alıcı_id>'. "
        "Eğer mesajda gerekli bilgiler eksikse 'ERROR: Bilgiler eksik' şeklinde yanıt ver."
    )
    response_text = response.content.strip()

    if "ERROR" in response_text:
        return {
            "messages": state["messages"] + [
                AIMessage(content="⚠️ Hata: Lütfen gönderilecek miktarı ve alıcı hesap numarasını belirtin.")],
            "next": "END"
        }

    try:
        extracted_data = {pair.split(": ")[0]: pair.split(": ")[1] for pair in response_text.split(", ")}
        amount = float(extracted_data["AMOUNT"])
        receiver_customer_id = int(extracted_data["RECEIVER_ID"])
    except (KeyError, ValueError):
        return {
            "messages": state["messages"] + [AIMessage(content="⚠️ Hata: İşlem detayları doğru formatta değil.")],
            "next": "END"
        }

    transaction_response = process_transaction.invoke({
        "sender_customer_id": sender_customer_id,
        "receiver_customer_id": receiver_customer_id,
        "amount": amount
    })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_transaction.invoke(
        {"customer_id": sender_customer_id, "transaction_details": f"[{timestamp}] {transaction_response}"})

    return {
        "messages": state["messages"] + [AIMessage(content=transaction_response)],
        "next": "END"
    }


def transaction_history_agent(state):
    """Retrieves the customer's transaction history."""
    customer_id = state.get("customer_id", None)

    if not customer_id:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Lütfen müşteri numaranızı belirtin.")],
            "next": "END"
        }

    # ✅ Fetch transaction history properly
    history_response = get_transaction_history.invoke({"customer_id": customer_id})

    return {
        "messages": state["messages"] + [AIMessage(content=history_response)],
        "next": "END"
    }


def complaint_agent(state):
    """Handles customer complaints using AI-driven extraction."""
    last_message = state["messages"][-1].content
    customer_id = state.get("customer_id", None)

    if not customer_id:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Lütfen müşteri numaranızı belirtin.")],
            "next": "END"
        }

    # ✅ Use AI to extract complaint details dynamically
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = llm.invoke(
        f"Kullanıcının mesajındaki şikayeti anlamlandır: {last_message}. "
        "Yanıt şu formatta olmalıdır: 'COMPLAINT: <şikayet_detayı>'. "
        "Eğer mesajda bir şikayet tespit edilemiyorsa 'ERROR: Şikayet tespit edilmedi' şeklinde yanıt ver."
    )
    response_text = response.content.strip()

    if "ERROR" in response_text:
        return {
            "messages": state["messages"] + [
                AIMessage(content="⚠️ Hata: Şikayet algılanamadı. Lütfen detaylı açıklayın.")],
            "next": "END"
        }

    # ✅ Extract complaint details
    try:
        complaint_text = response_text.split(": ")[1]
    except IndexError:
        return {
            "messages": state["messages"] + [AIMessage(content="⚠️ Hata: Şikayet detayları doğru formatta değil.")],
            "next": "END"
        }

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_complaint.invoke({"customer_id": customer_id, "complaint_text": f"[{timestamp}] {complaint_text}"})

    return {
        "messages": state["messages"] + [AIMessage(
            content="📩 Şikayetiniz başarıyla kaydedildi. Yetkililerimiz en kısa sürede sizinle iletişime geçecektir.")],
        "next": "END"
    }


def query_agent(state):
    """Handles general queries using an AI model. If unrelated, suggests relevant topics."""
    last_message = state["messages"][-1].content
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = llm.invoke(f"Kullanıcının sorduğu genel soruyu yanıtla: {last_message}")

    ai_response = response.content.strip()

    if "ERROR" in ai_response or "anlamıyorum" in ai_response.lower():
        ai_response = ("📌 Bu sistem aşağıdaki konular hakkında yardımcı olabilir:\n"
                       "- Bakiye sorgulama\n"
                       "- Para transferi işlemleri\n"
                       "- İşlem geçmişi\n"
                       "- Şikayet bildirimi\n"
                       "Lütfen bu konulardan birini seçiniz.")

    return {
        "messages": state["messages"] + [AIMessage(content=ai_response)],
        "next": "END"
    }
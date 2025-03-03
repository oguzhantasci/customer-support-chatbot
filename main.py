from langchain_openai import ChatOpenAI
from config import df
from workflow import build_workflow
from langchain.schema import HumanMessage, AIMessage


# Function to generate customer name dynamically
def get_customer_name_from_data(customer_id):
    customer_row = df[df["CustomerId"] == customer_id]
    if not customer_row.empty:
        surname = customer_row.iloc[0]["Surname"]
        gender = customer_row.iloc[0]["Gender"]
        title = "Bey" if gender == "Male" else "Hanım"
        return f"{surname} {title}"
    return "Değerli Müşterimiz"


# Initialize LLM for conversation analysis
llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)


def detect_conversation_end(messages):
    """Uses LLM to determine if the conversation should end."""
    analysis_prompt = "Aşağıdaki konuşmayı analiz et ve kullanıcının başka bir ihtiyacı olup olmadığını tespit et.\n\n"
    analysis_prompt += "\n\n".join([f"Kullanıcı: {msg.content}" for msg in messages if isinstance(msg, HumanMessage)])
    analysis_prompt += "\n\nChatbot'un konuşmayı sonlandırması gerekiyor mu? Eğer evet, 'EVET' yaz, eğer hayır, 'HAYIR' yaz."

    response = llm.predict(analysis_prompt)
    return "EVET" in response.upper()


def main():
    print("\n💬 Chatbot hazır! Çıkmak için 'çıkış' veya 'exit' yazın.\n")

    customer_id = input("🔑 Lütfen müşteri numaranızı girin: ")

    if not customer_id.isdigit():
        print("❌ Geçersiz müşteri numarası. Lütfen sayısal bir değer girin.")
        return

    customer_id = int(customer_id)
    customer_name = get_customer_name_from_data(customer_id)  # Generate customer name dynamically

    print(f"\n🤖 Bot: Merhaba {customer_name}, size nasıl yardımcı olabilirim?")

    workflow = build_workflow()
    compiled_workflow = workflow.compile()
    state = {"messages": [], "customer_id": customer_id, "customer_name": customer_name}

    while True:
        user_input = input("\n👤 Siz: ")
        if user_input.lower() in ["çıkış", "exit"]:
            print(f"\n🤖 Bot: {customer_name}, iyi günler dilerim! 👋")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = compiled_workflow.invoke(state)
        messages = state["messages"]

        bot_response = next((msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
                            "Üzgünüm, isteğinizi anlayamadım.")

        print(f"\n🤖 Bot: {bot_response.replace('Müşteri', customer_name)}")

        # Check if conversation should end using LLM
        if detect_conversation_end(state["messages"]):
            print(f"\n🤖 Bot: {customer_name}, size yardımcı olabildiysem ne mutlu! İyi günler dilerim! 👋")
            break


if __name__ == "__main__":
    main()

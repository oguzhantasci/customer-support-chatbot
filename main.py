from workflow import build_workflow
from langchain.schema import HumanMessage, AIMessage

def main():
    print("\n💬 Chatbot hazır! Çıkmak için 'çıkış' veya 'exit' yazın.\n")

    customer_id = input("🔑 Lütfen müşteri numaranızı girin: ")

    if not customer_id.isdigit():
        print("❌ Geçersiz müşteri numarası. Lütfen sayısal bir değer girin.")
        return

    customer_id = int(customer_id)

    workflow = build_workflow()
    compiled_workflow = workflow.compile()

    state = {"messages": [], "customer_id": customer_id}

    while True:
        user_input = input("\n👤 Siz: ")
        if user_input.lower() in ["çıkış", "exit"]:
            print("\n👋 Görüşmek üzere!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = compiled_workflow.invoke(state)
        messages = state["messages"]

        # ✅ Ensure proper extraction of AI response
        bot_response = next((msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
                            "Üzgünüm, isteğinizi anlayamadım.")

        print(f"\n🤖 Bot: {bot_response}")

if __name__ == "__main__":
    main()
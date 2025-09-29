from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0)

template = "You are a helpful AI assistant. Answer this question: {question} \n This is the conversation history for context:\n {history}"
prompt = PromptTemplate(template=template, input_variables=["question", "history"])

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

chain = prompt | llm

inputs = {
    "question": "",
    "history": []
}

print("Simple LLM Chatbot. Type 'exit' to quit.")
while True:
    inputs["question"] = input("You: ")
    if inputs["question"].lower() == "exit":
        print("Good Bye!")
        break

    response = chain.invoke(inputs)
    print(f"Bot: {response.content}")

    memory.save_context(inputs, {"response": response.content})
    inputs["history"] = memory.dict()["chat_memory"]["messages"]

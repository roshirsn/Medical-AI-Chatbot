from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

def is_casual_message(user_input: str) -> str | None:
    input_lower = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if input_lower in greetings:
        return "Hello! How can I assist you with your medical questions today?"

    casual_responses = {
        "how are you": "I'm doing well! How can I help you with medical information today?",
        "who are you": "I'm a medical assistant chatbot built to answer healthcare questions.",
        "where are you from": "I'm running on a cloud server to help users from anywhere!",
        "thank you": "You're welcome!",
        "thanks": "Glad to help!",
        "fuck you":"fuck you too",
        "you are an idiot":"you too"
    }

    return casual_responses.get(input_lower, None)

def is_offensive(user_input: str) -> bool:
    offensive_keywords = [
        "idiot", "stupid", "dumb", "fool", "nonsense", 
        "shut up", "kill yourself", "hate you", "moron","fuck","bitch","bastard","myre"
    ]
    input_lower = user_input.lower()
    return any(word in input_lower for word in offensive_keywords)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
TOGETHER_API_KEY=os.environ.get('TOGETHER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
#models : mistralai/Mixtral-8x7B-Instruct-v0.1, 

llm = ChatOpenAI(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
,  # You can use other supported models too
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=TOGETHER_API_KEY,
    temperature=0.4,
    max_tokens=500,
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User:", input)

    if is_offensive(input):
        warning = "Let's keep the conversation respectful. I'm here to help with medical information."
        print("Bot (warning):", warning)
        return str(warning)

    # Step 1: Casual message check
    casual_response = is_casual_message(input)
    if casual_response:
        print("Bot (casual):", casual_response)
        return str(casual_response)

    # Step 2: Go to RAG chain
    response = rag_chain.invoke({"input": input})
    print("Bot (rag):", response["answer"])
    return str(response["answer"])





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
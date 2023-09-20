from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests 

import gradio as gr
import time

requests.adapters.DEFAULT_TIMEOUT = 60
# hf_api_key = os.environ['HF_API_KEY']

def setup_and_get_qa_chain():
    persist_directory = './chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    # print(vectordb._collection.count())

    llm_name = "gpt-3.5-turbo-0301"
    llm = ChatOpenAI(model_name=llm_name, temperature=0.1)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. The context and information provided below is extracted from the resume of Ricky Cheng. Imagine you are Ricky Cheng and all the following information is about you. Recruiter are going to ask you questions regarding the resume. Keep your answer concise and complete. 
    Resume Context: 
    {context}

    Question: {question}
    You must pretend you are Ricky Cheng and give your answer in first person perspective.
    Answer 'I'm sorry, but I cannot answer that question as it is not related to my professional experience or qualifications.' if the question is not related to the resume or professional background.
    Answer: """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vectordb.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": QA_CHAIN_PROMPT
        }
    )

    return qa_chain

def respond(message, chat_history):
    result = qa_chain({"question": message})
    bot_message = result["answer"]
    return bot_message

if __name__ == "__main__":
    qa_chain = setup_and_get_qa_chain()

    gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=720),
        textbox=gr.Textbox(value="Tell me about yourself.", label="Chat with my resume!"), #placeholder="Ask me a yes or no question", container=False, scale=7),
        title="Ricky Cheng's Resume GPT",
        description="Chat with my resume! Ask anything you want to know better about my resume.\nI've built this project using LangChain, vector database, and OpenAI API.",
        theme="default",
        examples=[
            "Tell me about yourself.",
            "What is your strength?",
            "What is your most notable achievement?",
            "How can I contact you?"
        ],
        # cache_examples=True,
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
    ).launch()
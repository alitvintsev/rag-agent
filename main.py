import streamlit as st
from dataclasses import dataclass

from langchain_gigachat.chat_models import GigaChat
from langchain.chains import RetrievalQA

from agent.rag import Retriever
from agent.llm import gigachat


@dataclass
class Message:
    is_user: bool
    text: str


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def setup_agent():
    retriever = Retriever()
    retriever.load_document(path_to_document="data/mbulgakov_ivan.txt")
    retriever.split_document()
    db = retriever.create_vector_store()
    agent = RetrievalQA.from_chain_type(gigachat, retriever=db.as_retriever())
    return agent


def init_agent():
    if "agent" not in st.session_state:
        st.session_state.agent = setup_agent()
    return st.session_state.agent


def main():
    st.title("Агент с RAG механикой")
    st.write("по пьесе М. Булгакова \"Иван Васильевич\"")
    # Инициализация агента и истории чата

    messages = init_chat_history()
    agent = init_agent()

    for message in messages:
        with st.chat_message("user" if message.is_user else "assistant"):
            st.markdown(message.text)

    user_input = st.chat_input("Введите ваш запрос...")
    if user_input:
        messages.append(Message(is_user=True, text=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Генерация ответа..."):
                response = agent.invoke(user_input)
                st.markdown(response["result"])
                messages.append(Message(is_user=False, text=response["result"]))


if __name__ == "__main__":
    main()
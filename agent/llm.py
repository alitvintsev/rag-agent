from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from agent.config import get_config


config = get_config()
token = config["gigachat"]["token"]

gigachat = GigaChat(
    credentials=token,
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    temperature=0.1
)

embeddings = GigaChatEmbeddings(
    credentials=token,
    verify_ssl_certs=False
)
import os
import boto3
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------
# 0. RÉCUPÉRER LA CLÉ OPENAI DEPUIS AWS SECRETS MANAGER
# ------------------------------------------------------------
def get_openai_key_from_aws():
    client = boto3.client("secretsmanager", region_name="eu-north-1")

    response = client.get_secret_value(SecretId="my_rag_secrets")
    secret = json.loads(response["SecretString"])

    return secret["OPENAI_API_KEY"]


# ------------------------------------------------------------
# 1. CHARGER UN PDF
# ------------------------------------------------------------
def load_pdf(path_pdf):
    loader = PyPDFLoader(path_pdf)
    return loader.load()


# ------------------------------------------------------------
# 2. SPLIT EN CHUNKS (amélioré)
# ------------------------------------------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # légèrement plus petit → meilleur matching
        chunk_overlap=120      # garde du contexte
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 3. VECTORSTORE + EMBEDDINGS OPENAI (ULTRA IMPORTANT)
# ------------------------------------------------------------
def create_vectorstore(splits):
    api_key = get_openai_key_from_aws()

    # 🚀 Embeddings puissants
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 7})


# ------------------------------------------------------------
# 4. RAG CHAIN
# ------------------------------------------------------------
def create_rag(retriever):
    api_key = get_openai_key_from_aws()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    system_prompt = (
        "Tu es un assistant spécialisé dans la recherche d'information.\n"
        "Règles strictes :\n"
        "- Tu réponds UNIQUEMENT à partir du contexte RAG.\n"
        "- Si l'information existe dans les documents, tu la donnes de manière claire.\n"
        "- Ne dis pas 'je ne sais pas' avant d’avoir vérifié tous les extraits.\n"
        "- Si ça n’apparaît dans aucun extrait, tu dis que ce n’est pas dans les documents.\n"
        "\nContexte :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    return (
        RunnableMap({
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

import os
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json


# ------------------------------------------------------------
# 0. RÉCUPÉRER LA CLÉ OPENAI DEPUIS AWS SECRETS MANAGER
# ------------------------------------------------------------
def get_openai_key_from_aws():
    client = boto3.client("secretsmanager", region_name="eu-north-1")


    response = client.get_secret_value(SecretId="my_rag_secrets")

    secret = json.loads(response["SecretString"])
    return secret["OPENAI_API_KEY"]


# ------------------------------------------------------------
# 1. CHARGER LE PDF
# ------------------------------------------------------------
def load_pdf(path_pdf):
    loader = PyPDFLoader(path_pdf)
    docs = loader.load()
    return docs


# ------------------------------------------------------------
# 2. SPLIT EN CHUNKS
# ------------------------------------------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 3. CRÉER/CHARGER CHROMA VECTORSTORE
# ------------------------------------------------------------
def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 7})


# ------------------------------------------------------------
# 4. CRÉER LE RAG CHAIN
# ------------------------------------------------------------
def create_rag(retriever):

    # 📌 La clé OpenAI vient maintenant D’AWS
    api_key = get_openai_key_from_aws()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    system_prompt = (
        "Tu es un assistant spécialisé dans la recherche d'information.\n" 
        "Tu dois TOUJOURS te baser sur les documents fournis dans le contexte.\n"
        "Si une réponse se trouve dans les documents, tu la DONNES clairement.\n"
        "Si l’utilisateur pose une question précise, tu dois chercher dans les extraits RETOURNÉS PAR LE RAG.\n"
        "Règles :Utilise uniquement les passages retrouvés. Donne des réponses claires, structurées.Si l’information n'est pas trouvée dans les documents, dis-le ET propose des questions pertinentes. Ne dis pas je ne sais pas trop vite : vérifie d'abord les extraits.\n"
        "Contexte :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        RunnableMap({
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

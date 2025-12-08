import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
        chunk_size=1000,
        chunk_overlap=200
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
        persist_directory="chroma_db"  # Chroma est sauvegardé
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


# ------------------------------------------------------------
# 4. CRÉER LE RAG CHAIN
# ------------------------------------------------------------
def create_rag(retriever, api_key):
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    system_prompt = (
        "Tu es un assistant qui répond uniquement à partir du contexte fourni.\n"
        "Ton rôle est de résumer, expliquer, ou répondre à la question de l’utilisateur.\n"
        "Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.\n\n"
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

import streamlit as st
import os
from rag_engine import load_pdf, split_documents, create_vectorstore, create_rag

st.set_page_config(page_title="RAG Multi-PDF", page_icon="📚", layout="wide")

st.title("📚 RAG Multi-PDF — Streamlit App")
st.write("Pose une question sur les documents chargés.")


# ------------------------------------------------------------
# 1. Charger les PDF + Créer RAG
# ------------------------------------------------------------
if "rag_chain" not in st.session_state:

    with st.spinner("🔄 Chargement des PDF et création des embeddings..."):

        all_docs = []

        pdf_folder = "pdfs"
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        st.write(f"📄 PDFs détectés : {pdf_files}")

        for pdf in pdf_files:
            path = os.path.join(pdf_folder, pdf)
            st.write(f"➡️ Chargement : {pdf}")
            docs = load_pdf(path)
            all_docs.extend(docs)

        splits = split_documents(all_docs)
        retriever = create_vectorstore(splits)

        # 👉 SESSION ID UNIQUE PAR UTILISATEUR
        st.session_state["rag_chain"] = create_rag(retriever, session_id="client_1")

    st.success("✅ RAG prêt ! Pose une question.")


# ------------------------------------------------------------
# 2. UI question
# ------------------------------------------------------------
question = st.text_input("Votre question :", placeholder="Ex: Quelle est la TVA ?")


# ------------------------------------------------------------
# 3. Réponse
# ------------------------------------------------------------
if st.button("Envoyer") and question:

    with st.spinner("🧠 Analyse des documents et de la mémoire..."):
        answer = st.session_state["rag_chain"](question)

    st.subheader("📌 Réponse")
    st.write(answer)

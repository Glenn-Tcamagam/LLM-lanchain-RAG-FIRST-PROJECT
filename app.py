import streamlit as st
import os
from rag_engine import load_pdf, split_documents, create_vectorstore, create_rag

st.set_page_config(page_title="RAG Multi-PDF", page_icon="📚")

st.title("📚 RAG Multi-PDF — Streamlit App")
st.write("Pose une question sur les documents chargés.")


# ------------------------------------------------------------
# 1. Préparer RAG au démarrage (pas de clé API ici)
# ------------------------------------------------------------
if "rag_chain" not in st.session_state:

    st.info("🔄 Chargement des PDF et création des embeddings...")

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

    rag_chain = create_rag(retriever)
    st.session_state["rag_chain"] = rag_chain

    st.success("✅ RAG prêt ! Vous pouvez poser une question.")


# ------------------------------------------------------------
# 2. Pose de question
# ------------------------------------------------------------
question = st.text_input("Votre question :")

if st.button("Envoyer") and question:
    with st.spinner("Analyse en cours..."):
        answer = st.session_state["rag_chain"].invoke(question)

    st.subheader("📌 Réponse")
    st.write(answer)

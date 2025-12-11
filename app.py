import streamlit as st
from rag_engine import get_rag_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ----------- CSS STYLE WHATSAPP/MESSENGER -------------
st.markdown("""
<style>

.chat-container {
    max-height: 650px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 12px;
    background-color: #f5f6f8;
    border: 1px solid #ddd;
}

.user-bubble {
    background-color: #DCF8C6;
    color: #000;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 5px 0;
    max-width: 70%;
    float: right;
    clear: both;
}

.bot-bubble {
    background-color: #ffffff;
    color: #000;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 5px 0;
    max-width: 70%;
    float: left;
    clear: both;
    border: 1px solid #ececec;
}

</style>
""", unsafe_allow_html=True)


# ---------- INITIALISATION RAG + MEMORY ----------
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # { "role": "user"/"assistant", "content": "..." }


st.title("🤖 Chatbot RAG avec Mémoire")


# ---------- DISPLAY CHAT HISTORY ----------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{message['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------- USER INPUT ----------
user_input = st.chat_input("Pose ta question...")

if user_input:
    # Ajouter message user
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Récupération réponse via RAG
    answer = st.session_state["rag_chain"](user_input)

    # Ajouter message bot
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Rafraîchir pour scroll automatique
    st.rerun()

import os
import boto3
import json
import time
from datetime import datetime
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -------------------------
# Configuration mémoire
# -------------------------
MEMORY_FILE = "memory.json"         # fichier local pour stocker la mémoire
MEMORY_MAX_ITEMS = 8               # nombre de paires Q/A à garder (récent -> ancien)


# ------------------------------------------------------------
# 0. RÉCUPÉRER LA CLÉ OPENAI DEPUIS AWS SECRETS MANAGER
# ------------------------------------------------------------
def get_openai_key_from_aws(region_name: str = "eu-north-1", secret_name: str = "my_rag_secrets"):
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])
    return secret["OPENAI_API_KEY"]


# ------------------------------------------------------------
# Mémoire : helpers pour sauvegarder / charger / ajouter
# ------------------------------------------------------------
def load_memory() -> List[dict]:
    """
    Charge le fichier memory.json si présent.
    Retourne une liste de dicts : {"question":..., "answer":..., "ts": ...}
    """
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_memory(mem: List[dict]):
    """Écrit la mémoire sur disque (atomique approximatif)."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem[:MEMORY_MAX_ITEMS], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def append_memory(question: str, answer: str):
    """Ajoute une paire question/answer en tête (récent d'abord) puis sauvegarde."""
    mem = load_memory()
    entry = {"question": question, "answer": answer, "ts": datetime.utcnow().isoformat()}
    # insérer en tête
    mem.insert(0, entry)
    # tronquer
    mem = mem[:MEMORY_MAX_ITEMS]
    save_memory(mem)


def memory_to_text(max_items: int = MEMORY_MAX_ITEMS) -> str:
    """
    Convertit la mémoire en texte structuré pour être injecté dans le prompt.
    On met d'abord les plus récents.
    """
    mem = load_memory()[:max_items]
    if not mem:
        return ""
    parts = []
    for i, e in enumerate(mem, 1):
        q = e.get("question", "").strip()
        a = e.get("answer", "").strip()
        ts = e.get("ts", "")
        parts.append(f"[Mémoire {i} - {ts}]\nQ: {q}\nR: {a}")
    return "\n\n".join(parts)


# ------------------------------------------------------------
# 1. CHARGER UN PDF
# ------------------------------------------------------------
def load_pdf(path_pdf: str):
    loader = PyPDFLoader(path_pdf)
    return loader.load()


# ------------------------------------------------------------
# 2. SPLIT EN CHUNKS (amélioré)
# ------------------------------------------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 3. VECTORSTORE + EMBEDDINGS OPENAI (ULTRA IMPORTANT)
# ------------------------------------------------------------
def create_vectorstore(splits, persist_directory: str = "chroma_db"):
    api_key = get_openai_key_from_aws()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # on garde k raisonnable (multi-query / mémoire aidera)
    return vectorstore.as_retriever(search_kwargs={"k": 7})


# ------------------------------------------------------------
# Utility : récupérer docs à partir du retriever (fallbacks)
# Compatible avec plusieurs versions de LangChain / VectorStore
# ------------------------------------------------------------
def retrieve_with_fallback(retriever, query: str, k: int = 7):
    """
    Retourne une liste de documents pertinents pour la query.
    Essaie différentes méthodes selon les versions/implémentations.
    """
    # 1) Prefer invoke (nouvelles API)
    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
            if docs:
                return list(docs)
    except Exception:
        pass

    # 2) get_relevant_documents (souvent présent)
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
            if docs:
                return list(docs)
    except Exception:
        pass

    # 3) get_relevant_nodes
    try:
        if hasattr(retriever, "get_relevant_nodes"):
            docs = retriever.get_relevant_nodes(query)
            if docs:
                return list(docs)
    except Exception:
        pass

    # 4) similarity_search sur vectorstore (fallback)
    try:
        if hasattr(retriever, "vectorstore") and hasattr(retriever.vectorstore, "similarity_search"):
            docs = retriever.vectorstore.similarity_search(query, k=k)
            if docs:
                return list(docs)
    except Exception:
        pass

    # 5) similarity_search sur retriever
    try:
        if hasattr(retriever, "similarity_search"):
            docs = retriever.similarity_search(query, k=k)
            if docs:
                return list(docs)
    except Exception:
        pass

    # 6) callable fallback
    try:
        if callable(retriever):
            docs = retriever(query)
            if docs:
                return list(docs)
    except Exception:
        pass

    # default -> vide
    return []


# ------------------------------------------------------------
# 4. CREATE_RAG + Mémoire (wrapper compatible app.py)
# ------------------------------------------------------------
def create_rag(retriever):
    """
    Retourne une instance RagWrapper compatible avec ton app.py (méthode .invoke(question)).
    Fonctionnalités :
      - Multi-query simple (templates)
      - Injection de la mémoire locale dans le prompt
      - Sauvegarde de la paire question/réponse après chaque invocation
    """

    api_key = get_openai_key_from_aws()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt : on inclut un bloc {memory} qui sera vide si pas de mémoire
    system_prompt = (
        "Tu es un assistant spécialisé dans la recherche d'information.\n"
        "Tu dois répondre en te basant STRICTEMENT sur le contexte fourni.\n"
        "Si l'information n'est pas dans le contexte, dis clairement que ce n'est pas présent.\n"
        "Donne une réponse structurée et précise. Cite les passages ou indique 'Source : page X' si disponible.\n\n"
        "Mémoire (extraits de la conversation précédente) :\n{memory}\n\n"
        "Contexte (extraits trouvés dans les documents) :\n{context}\n\n"
        "QUESTION : {question}\n\n"
        "RÉPONDS :"
    )

    # helper formatage docs
    def format_docs(docs):
        # join, limiter la longueur pour ne pas dépasser le contexte
        # on limite chaque chunk pour éviter prompts trop longs
        MAX_CHUNK_LEN = 2500
        parts = []
        for d in docs:
            txt = getattr(d, "page_content", "")
            if len(txt) > MAX_CHUNK_LEN:
                txt = txt[:MAX_CHUNK_LEN] + " [...]"
            parts.append(txt)
        return "\n\n---\n\n".join(parts)

    # simple multi-query templates (rapide, pas de coût)
    def generate_query_variants(question: str):
        q = question.strip()
        return [
            q,
            f"Définition : {q}",
            f"Explique simplement : {q}",
            f"Résumé des points clés à propos de : {q}",
            f"Quels sont les détails importants sur : {q}"
        ]

    class RagWrapper:
        def __init__(self, retriever, llm, system_prompt):
            self.retriever = retriever
            self.llm = llm
            self.system_prompt = system_prompt

        def invoke(self, input_arg):
            # Normaliser l'entrée
            if isinstance(input_arg, dict):
                question = input_arg.get("input") or input_arg.get("question") or ""
            else:
                question = str(input_arg)

            if not question:
                return "Erreur : question vide."

            # ------------------------
            # 1) Multi-query retrieval (rapide)
            # ------------------------
            variants = generate_query_variants(question)
            all_docs = []
            for v in variants:
                docs = retrieve_with_fallback(self.retriever, v, k=7)
                if docs:
                    # extend normalized list
                    if isinstance(docs, (list, tuple)):
                        all_docs.extend(docs)
                    else:
                        all_docs.append(docs)

            # déduplication simple (par contenu)
            unique = []
            seen = set()
            for d in all_docs:
                txt = getattr(d, "page_content", "")
                if txt not in seen:
                    seen.add(txt)
                    unique.append(d)

            # si aucun passage -> message d'erreur user-friendly
            if not unique:
                return "Aucun passage pertinent trouvé dans les documents."

            # on garde un nombre maximal de chunks (pour controler taille prompt)
            MAX_CHUNKS = 8
            selected_docs = unique[:MAX_CHUNKS]

            # ------------------------
            # 2) Préparer mémoire (si elle existe)
            # ------------------------
            mem_text = memory_to_text(max_items=MEMORY_MAX_ITEMS)
            if not mem_text:
                mem_text = "Aucune mémoire disponible."

            # ------------------------
            # 3) Formater contexte et construire prompt final
            # ------------------------
            context_text = format_docs(selected_docs)

            prompt_text = self.system_prompt.format(
                memory=mem_text,
                context=context_text,
                question=question
            )

            # ------------------------
            # 4) Appeler LLM (ChatOpenAI). On utilise la forme messages si possible.
            # ------------------------
            try:
                # certains ChatOpenAI attendent une liste de messages
                response = self.llm.invoke([{"role": "user", "content": prompt_text}])
            except Exception:
                try:
                    response = self.llm.invoke(prompt_text)
                except Exception as e:
                    return f"Erreur LLM : {e}"

            # normaliser la sortie
            if isinstance(response, str):
                answer_text = response
            elif hasattr(response, "content"):
                answer_text = response.content
            elif isinstance(response, dict):
                answer_text = response.get("result") or response.get("text") or str(response)
            else:
                answer_text = str(response)

            # ------------------------
            # 5) Sauvegarder la paire Q/A dans la mémoire locale
            # ------------------------
            try:
                append_memory(question=question, answer=answer_text)
            except Exception:
                # ne laisse pas une erreur de mémoire casser le flux
                pass

            # ------------------------
            # 6) Retourner la réponse
            # ------------------------
            return answer_text

    # retourner une instance prête à l'emploi (compatible app.py)
    return RagWrapper(retriever, llm, system_prompt)

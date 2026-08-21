import streamlit as st
import requests
import json
import logging
from pypdf import PdfReader
from ddgs import DDGS
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# Configurações de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OLLAMA_URL = "http://ollama:11434"
CHROMA_PERSIST_PATH = "/data/chroma_db"  # Salva no volume montado em /mnt/ext4
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Nyeera AI", page_icon="🍃", layout="wide")
st.title("🍃 Chat Nyeera")


# --- Inicialização do ChromaDB Persistente ---
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    ollama_ef = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="documentos_locais",
        embedding_function=ollama_ef
    )
    return collection

collection = get_chroma_collection()

# --- Funções Utilitárias ---
def get_installed_models():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if res.status_code == 200:
            return [m["name"] for m in res.json().get("models", [])]
        return []
    except Exception:
        return []

def split_text_into_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_store_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    if not text.strip():
        return 0

    chunks = split_text_into_chunks(text)
    ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": uploaded_file.name, "chunk_id": i} for i in range(len(chunks))]
    
    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def query_vector_db(query, n_results=3):
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        formatted_chunks = []
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "Documento")
            formatted_chunks.append(f"Fonte [{source}]:\n{doc}")
        return "\n\n".join(formatted_chunks)
    except Exception as e:
        logging.error(f"[ChromaDB] Erro na consulta vetorial: {e}")
        return ""

def search_web(query, max_results=3):
    if not query or not query.strip():
        return ""
    logging.info(f"[DuckDuckGo] Buscando: '{query}'")
    results = []
    try:
        with DDGS() as ddgs:
            for idx, r in enumerate(ddgs.text(query, max_results=max_results), start=1):
                results.append(f"Título: {r.get('title')}\nTrecho: {r.get('body')}\nFonte: {r.get('href')}")
        return "\n\n".join(results)
    except Exception as e:
        logging.error(f"[DuckDuckGo] Erro: {e}")
        return ""

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    models = get_installed_models()
    selected_model = st.selectbox("Modelo LLM:", models if models else ["llama3.2"])
    
    st.divider()
    st.header("📂 Base de Conhecimento Vetorial")
    st.caption(f"Documentos indexados: **{collection.count()} chunks**")
    
    uploaded_files = st.file_uploader("Indexar Novos PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Indexar no ChromaDB"):
        if uploaded_files:
            with st.spinner("Vetorizando e salvando no ChromaDB..."):
                total_chunks = 0
                for file in uploaded_files:
                    total_chunks += process_and_store_pdf(file)
                st.success(f"Concluído! {total_chunks} chunks indexados permanentemente.")
                st.rerun()
        else:
            st.warning("Selecione pelo menos um arquivo PDF.")
            
    if st.button("🗑️ Limpar Banco Vetorial"):
        chromadb.PersistentClient(path=CHROMA_PERSIST_PATH).delete_collection("documentos_locais")
        st.success("Banco de dados resetado.")
        st.rerun()

    st.divider()
    enable_web = st.toggle("Habilitar Busca Web (DuckDuckGo)", value=False)
    
    if st.button("Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Digite sua dúvida..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    context_chunks = []
    
    # 1. Recuperação Vetorial Semântica (RAG)
    if collection.count() > 0:
        with st.status("Consultando base vetorial...", expanded=False):
            doc_context = query_vector_db(user_query, n_results=3)
            if doc_context:
                context_chunks.append(f"--- DOCUMENTOS INTERNOS ---\n{doc_context}")
                st.write("Trechos relevantes recuperados do ChromaDB.")

    # 2. Busca Web
    if enable_web:
        with st.status("Consultando DuckDuckGo...", expanded=False):
            web_context = search_web(user_query)
            if web_context:
                context_chunks.append(f"--- DADOS DA WEB ---\n{web_context}")
                st.write("Informações da web recuperadas.")

    # 3. Montagem do Prompt do Sistema
    system_prompt = (
        "Você é um assistente técnico e objetivo. Responda em Português do Brasil.\n"
        "Utilize o contexto fornecido abaixo para fundamentar sua resposta. "
        "Se a resposta não estiver contida nas fontes, seja transparente.\n\n"
    )
    if context_chunks:
        system_prompt += "\n\n".join(context_chunks)

    payload_messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        payload_messages.append({"role": m["role"], "content": m["content"]})

    # 4. Stream da Resposta via Ollama
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "stream": True
            }
            with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=90) as res:
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            content = chunk.get("message", {}).get("content", "")
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Erro na API Ollama: {res.status_code}")
        except Exception as e:
            st.error(f"Erro de conexão com Ollama: {str(e)}")
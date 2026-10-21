import streamlit as st
import requests
import json
import logging
import os
from datetime import datetime
from pypdf import PdfReader
from ddgs import DDGS
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Configurações de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Descubra o número de núcleos disponíveis ou defina um valor fixo (ex: 4 ou 8)
CPU_THREADS = max(1, os.cpu_count() - 1)  # Deixa 1 núcleo livre para o sistema

OLLAMA_URL = "http://ollama:11434"
CHROMA_PERSIST_PATH = "/data/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Nyeera AI", page_icon="🍃", layout="wide")
st.title("🍃 Chat Nyeera")

def get_now_str():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# --- Inicialização do ChromaDB Persistente ---
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    
    # O ChromaDB aceita passar parâmetros extras de cliente ou configurar a URL
    #ollama_ef = OllamaEmbeddingFunction(
    #    url=f"{OLLAMA_URL}/api/embeddings",
    #    model_name=EMBEDDING_MODEL
    #)
    # Usa o modelo BAAI/bge-small-en-v1.5 rodando em ONNX CPU ultrarrápido
    #embedding_fn = FastEmbedEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    embedding_fn = DefaultEmbeddingFunction()

    # Aumenta o timeout padrão do cliente httpx interno do Ollama (em segundos)
    #if hasattr(ollama_ef, "_client") and hasattr(ollama_ef._client, "_client"):
    #    ollama_ef._client._client.timeout = 300.0  # 5 minutos
        
    if hasattr(embedding_fn, "_client") and hasattr(embedding_fn._client, "_client"):
        embedding_fn._client._client.timeout = 300.0  # 5 minutos

    collection = client.get_or_create_collection(
        name="documentos_locais",
        #embedding_function=ollama_ef
        embedding_function=embedding_fn
    )
    return collection

# Obtenha a coleção protegendo a leitura com tratamento de exceção
try:
    collection = get_chroma_collection()
    doc_count = collection.count()
except Exception:
    # Se a coleção foi apagada ou ficou órfã em memória, recria
    st.cache_resource.clear()
    collection = get_chroma_collection()
    doc_count = collection.count()

# --- Funções Utilitárias ---
def get_installed_models():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if res.status_code == 200:
            return [m["name"] for m in res.json().get("models", [])]
        return []
    except Exception:
        return []

# --- Divisão de Chunks e Ingestão em Lotes (Batching) ---
#def split_text_into_chunks(text, chunk_size=1000, overlap=100):
def split_text_into_chunks(text, chunk_size=600, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_and_store_pdf(uploaded_file, batch_size=10):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    if not text.strip():
        return 0

    chunks = split_text_into_chunks(text)
    total_chunks = len(chunks)
    
    # Processa em blocos menores (ex: 10 chunks por requisição)
    progress_bar = st.progress(0, text=f"Indexando {uploaded_file.name}...")
    
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = [f"{uploaded_file.name}_chunk_{j}" for j in range(i, i + len(batch_chunks))]
        batch_meta = [{"source": uploaded_file.name, "chunk_id": j} for j in range(i, i + len(batch_chunks))]
        
        collection.upsert(
            documents=batch_chunks,
            ids=batch_ids,
            metadatas=batch_meta
        )
        
        # Atualiza o progresso visual na interface
        progress_val = min((i + batch_size) / total_chunks, 1.0)
        progress_bar.progress(progress_val, text=f"Processados {min(i + batch_size, total_chunks)} de {total_chunks} chunks...")

    progress_bar.empty()
    return total_chunks

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
    st.caption(f"Documentos indexados: **{doc_count} chunks**")
    
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
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
            client.delete_collection("documentos_locais")
        except Exception:
            pass
        st.cache_resource.clear()  # <-- Limpa o cache para recriar a coleção limpa
        st.success("Banco de dados resetado.")
        st.rerun()

    st.divider()
    enable_web = st.toggle("Habilitar Busca Web (DuckDuckGo)", value=False)
    
    if st.button("Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Gerenciamento e Exibição do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico mantendo os metadados de tempo
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timestamp_info"):
            st.caption(msg["timestamp_info"])

# --- Entrada do Usuário ---
if user_query := st.chat_input("Digite sua dúvida..."):
    asked_at_dt = datetime.now()
    asked_at_str = asked_at_dt.strftime("%d/%m/%Y %H:%M:%S")

    # Registra a mensagem do usuário com o timestamp do envio
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "timestamp_info": f"🕒 **Enviado em:** {asked_at_str}"
    })
    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(f"🕒 **Enviado em:** {asked_at_str}")

    context_chunks = []
    
    # 1. Busca Vetorial (ChromaDB)
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

    # 3. Prompt do Sistema
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

    # 4. Stream da Resposta e Cronometragens
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        timing_placeholder = st.empty()
        
        full_response = ""
        first_token_received = False
        first_token_dt = None
        first_token_str = "Aguardando..."

        try:
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "stream": True,
                "options": {
                    "num_ctx": 2048,  # Limita o contexto a 2048 tokens para responder mais rápido na CPU
                    "num_thread": CPU_THREADS,   # Ajuste para a quantidade de threads/núcleos disponíveis da sua CPU deixando 1 livre para o OS
                    "temperature": 0.2     # Reduz dispersão em respostas técnicas
                }
            }

            with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10,300)) as res:
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            content = chunk.get("message", {}).get("content", "")
                            
                            # Registra o momento do PRIMEIRO TOKEN gerado
                            if not first_token_received and content:
                                first_token_received = True
                                first_token_dt = datetime.now()
                                first_token_str = first_token_dt.strftime("%d/%m/%Y %H:%M:%S")

                            full_response += content
                            response_placeholder.markdown(full_response + "▌")

                    # Finalização completa da geração
                    finished_at_dt = datetime.now()
                    finished_at_str = finished_at_dt.strftime("%d/%m/%Y %H:%M:%S")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Cálculo de durações
                    ttft_seconds = (first_token_dt - asked_at_dt).total_seconds() if first_token_dt else 0
                    total_seconds = (finished_at_dt - asked_at_dt).total_seconds()

                    # Bloco de metadados de tempo formatado
                    timing_info = (
                        f"⏱️ **Pergunta feita:** `{asked_at_str}` | "
                        f"**Início da resposta:** `{first_token_str}` (Latência: `{ttft_seconds:.2f}s`) | "
                        f"**Concluído em:** `{finished_at_str}` (Total: `{total_seconds:.2f}s`)"
                    )

                    timing_placeholder.caption(timing_info)

                    # Persiste a resposta e os tempos no histórico
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp_info": timing_info
                    })
                else:
                    st.error(f"Erro na API Ollama: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Erro de conexão com Ollama: {str(e)}")

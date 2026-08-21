# 🧠 Local AI Platform: RAG, Web Search & Data Pipeline

Uma plataforma integrada e 100% local com orquestração via **Docker Compose**, combinando execução de Modelos de Linguagem (LLMs), RAG semântico persistente com **ChromaDB**, busca em tempo real na web via **DuckDuckGo**, orquestração com **Apache Airflow** e mensageria com **Apache Kafka**.

---

## 🏗️ Arquitetura do Sistema

* **LLM Engine & Embeddings:** [Ollama](https://ollama.com/) (execução local otimizada para CPU/GPU)
* **Frontend / Interface:** [Streamlit](https://streamlit.io/) com streaming em tempo real
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) com persistência em disco local
* **Web Search Engine:** DuckDuckGo Search API (`ddgs`)
* **Orquestração de Dados:** Apache Airflow 2.8+ (Standalone)
* **Mensageria & Streaming:** Apache Kafka (KRaft mode - sem dependência de Zookeeper)

---

## 🚀 Como Executar

### 1. Pré-requisitos
* [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/) instalados.
* Pelo menos 8GB de RAM disponível no host.

---

### 2. Inicialização dos Containers

Clone o repositório e suba a stack completa:

```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio
docker compose up -d

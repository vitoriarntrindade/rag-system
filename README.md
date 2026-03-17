# Sistema RAG

Um sistema de **Retrieval-Augmented Generation (RAG)** modular, desacoplado e pronto para produção — construído com arquitetura **Ports & Adapters (Hexagonal)** para suportar múltiplos providers de LLM e embeddings sem modificar o núcleo da aplicação.

---

![unnamed](https://github.com/user-attachments/assets/74ff963a-b233-4c1e-a69d-e24bcf44a52f)

---

## 🧠 Visão Geral da Arquitetura

### O problema que foi resolvido

Na versão original, o sistema estava **acoplado diretamente ao OpenAI**:

- `generator.py` importava `ChatOpenAI` diretamente
- `vector_store.py` importava `OpenAIEmbeddings` diretamente
- `rag_pipeline.py` recebia `openai_api_key` e escrevia em `os.environ` globalmente
- Trocar de provider significava modificar múltiplos arquivos do core

### A solução: Ports & Adapters

O refactor aplicou o padrão **Hexagonal Architecture** (Ports & Adapters), separando claramente três camadas:

```
┌─────────────────────────────────────────────────────┐
│                      CORE                           │
│  generator · vector_store · retriever · pipeline    │
│  Depende APENAS de interfaces (ports)               │
└──────────────────┬──────────────────────────────────┘
                   │ usa
┌──────────────────▼──────────────────────────────────┐
│                    PORTS                            │
│  BaseLLMProvider · BaseEmbeddingProvider            │
│  Contratos ABC — o core nunca sabe quem implementa  │
└──────────────────┬──────────────────────────────────┘
                   │ implementado por
┌──────────────────▼──────────────────────────────────┐
│                  ADAPTERS                           │
│  OpenAILLMAdapter · OpenAIEmbeddingAdapter          │
│  (futuros: OllamaAdapter · AzureAdapter · ...)      │
└──────────────────┬──────────────────────────────────┘
                   │ criado por
┌──────────────────▼──────────────────────────────────┐
│                  FACTORY                            │
│  create_llm_provider · create_embedding_provider    │
│  Lê settings e instancia o adapter correto          │
└─────────────────────────────────────────────────────┘
```

### Benefícios práticos

| Antes | Depois |
|---|---|
| `from langchain_openai import ChatOpenAI` no core | Core importa apenas `BaseLLMProvider` |
| Trocar provider = editar generator + vector_store + pipeline | Trocar provider = mudar 1 variável no `.env` |
| Testes dependem de mock manual da classe OpenAI | Testes injetam `MagicMock(spec=BaseLLMProvider)` |
| `os.environ['OPENAI_API_KEY']` setado globalmente | Credencial passada explicitamente por injeção |
| Acoplamento forte — difícil testar isolado | Baixo acoplamento — cada módulo testável sozinho |

---

## 🚀 Funcionalidades

- **Suporte a múltiplos formatos**: PDF, TXT, DOCX, DOC, Markdown
- **Arquitetura Hexagonal (Ports & Adapters)**: core isolado de providers externos
- **Multi-provider**: troca de LLM ou embeddings via variável de ambiente
- **Injeção de dependência**: providers injetáveis via construtor (ideal para testes)
- **Factory centralizada**: único ponto de criação de providers a partir da config
- **Ingestão flexível**: arquivo único, diretório ou processamento em lote
- **Recuperação vetorial**: ChromaDB com busca por similaridade ou MMR
- **Interface CLI completa**: ingest, query e chat interativo
- **Pronto para produção**: logging estruturado, tratamento de erros, configuração via env

---

## 📋 Requisitos

- Python 3.9+
- Chave de API de um provider suportado (atualmente: OpenAI)

---

## 🔧 Instalação

```bash
git clone https://github.com/vitoriarntrindade/rag-system.git
cd rag-system

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## ⚙️ Configuração

### Arquivo `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
# ─── Provider selection ────────────────────────────────────────────
# Qual backend usar para LLM e embeddings.
# Valores suportados agora: "openai"
# Futuros: "azure", "ollama"
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

# ─── OpenAI ────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# Modelo de embeddings (geração dos vetores)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Modelo de chat (geração das respostas)
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# Criatividade das respostas (0.0 = determinístico, 2.0 = criativo)
OPENAI_TEMPERATURE=0.3

# ─── Processamento de texto ────────────────────────────────────────
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# ─── Recuperação ───────────────────────────────────────────────────
RETRIEVAL_TOP_K=5
RETRIEVAL_SEARCH_TYPE=similarity   # ou mmr

# ─── Logging ───────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_TO_FILE=true
```

> O `main.py` carrega o `.env` automaticamente com `load_dotenv()`. Não é necessário exportar a variável no shell.

---

## 🎯 Uso via CLI

```bash
# Ingestão de documento único
python main.py ingest --file data/documento.pdf

# Ingestão de diretório completo
python main.py ingest --directory data/

# Filtrar por tipo de arquivo
python main.py ingest --directory data/ --file-types pdf txt md

# Listar arquivos disponíveis sem processar
python main.py ingest --directory data/ --list-files

# Forçar recriação do índice
python main.py ingest --file data/documento.pdf --force

# Consulta pontual
python main.py query "Qual é o tema principal?"

# Consulta sem exibir fontes
python main.py query "Explique o conceito" --no-sources

# Chat interativo
python main.py chat
```

---

## 📦 Uso como Biblioteca

### Uso básico (provider padrão via `.env`)

```python
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()

pipeline = RAGPipeline(api_key=os.getenv("OPENAI_API_KEY"))

pipeline.ingest_documents(file_path="data/documento.pdf")
answer, sources = pipeline.query("O que é mudança climática?")
print(answer)
```

### Injeção de provider personalizado (Ports & Adapters)

Para usar um provider diferente sem alterar nenhum arquivo do core, basta implementar o port e injetar:

```python
from src.ports.llm_provider import BaseLLMProvider
from src.ports.embedding_provider import BaseEmbeddingProvider
from src.rag_pipeline import RAGPipeline

# Implementação customizada — pode ser Ollama, Azure, um mock, etc.
class MeuLLMProvider(BaseLLMProvider):
    def generate(self, system_prompt: str, user_message: str) -> str:
        # sua lógica aqui
        return "resposta gerada localmente"

class MeuEmbeddingProvider(BaseEmbeddingProvider):
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384

# Core não sabe que não é OpenAI
pipeline = RAGPipeline(
    api_key="",  # não necessário quando providers são injetados
    llm_provider=MeuLLMProvider(),
    embedding_provider=MeuEmbeddingProvider(),
)
```

### Integração com FastAPI

```python
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline

load_dotenv()
app = FastAPI(title="RAG API")
pipeline = RAGPipeline(api_key=os.getenv("OPENAI_API_KEY"))

class Pergunta(BaseModel):
    texto: str

class Resposta(BaseModel):
    resposta: str

@app.post("/perguntar", response_model=Resposta)
def perguntar(req: Pergunta):
    answer, _ = pipeline.query(req.texto)
    return Resposta(resposta=answer)
```

### Chatbot simples

```python
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()
pipeline = RAGPipeline(api_key=os.getenv("OPENAI_API_KEY"))
pipeline.load_existing_index()

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        break
    resposta, _ = pipeline.query(pergunta)
    print(f"Bot: {resposta}")
```

---

## 🏗️ Estrutura do Projeto

```
rag-system/
├── src/
│   ├── ports/                       # Contratos (interfaces ABC)
│   │   ├── llm_provider.py          #   BaseLLMProvider
│   │   └── embedding_provider.py    #   BaseEmbeddingProvider
│   │
│   ├── adapters/                    # Implementações concretas dos ports
│   │   ├── llm/
│   │   │   └── openai_llm.py        #   OpenAILLMAdapter
│   │   └── embeddings/
│   │       └── openai_embeddings.py #   OpenAIEmbeddingAdapter
│   │
│   ├── factories/                   # Criação de providers via config
│   │   └── provider_factory.py      #   create_llm_provider / create_embedding_provider
│   │
│   ├── document_loader.py           # Carregamento de PDF, TXT, DOCX, MD
│   ├── text_processor.py            # Chunking e divisão de texto
│   ├── vector_store.py              # ChromaDB (usa BaseEmbeddingProvider)
│   ├── retriever.py                 # Recuperação de documentos relevantes
│   ├── generator.py                 # Geração de resposta (usa BaseLLMProvider)
│   ├── rag_pipeline.py              # Orquestrador principal
│   └── utils/
│       └── logger.py
│
├── config/
│   └── settings.py                  # Pydantic Settings v2
│
├── tests/
│   ├── conftest.py
│   ├── test_adapters.py             # Testes dos adapters OpenAI
│   ├── test_provider_factory.py     # Testes da factory
│   ├── test_generator.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   ├── test_retriever.py
│   ├── test_document_loader.py
│   ├── test_text_processor.py
│   ├── test_settings.py
│   └── test_logger.py
│
├── data/                            # Documentos para ingestão
├── main.py                          # Entry point CLI
├── example_usage.py                 # Exemplos de uso como biblioteca
├── requirements.txt
└── .env                             # Variáveis de ambiente (não versionado)
```

---

## 🔌 Adicionando um Novo Provider

Graças ao padrão Ports & Adapters, adicionar suporte a Ollama, Azure ou qualquer outro provider requer **apenas 2 passos**:

**1. Criar o adapter** em `src/adapters/llm/` ou `src/adapters/embeddings/`:

```python
# src/adapters/llm/ollama_llm.py
from src.ports.llm_provider import BaseLLMProvider

class OllamaLLMAdapter(BaseLLMProvider):
    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        self._base_url = base_url

    def generate(self, system_prompt: str, user_message: str) -> str:
        # integração com Ollama API
        ...
```

**2. Registrar na factory** em `src/factories/provider_factory.py`:

```python
elif name == "ollama":
    from src.adapters.llm.ollama_llm import OllamaLLMAdapter
    return OllamaLLMAdapter(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
```

**Nenhum outro arquivo precisa ser alterado.** O core, os testes e o pipeline continuam funcionando sem modificação.

---

## �� Testes

A suíte cobre **219 testes** com 100% de aprovação, organizados por camada:

| Arquivo | O que testa |
|---|---|
| `test_adapters.py` | Contrato ABC, inicialização, delegação e exceções dos adapters OpenAI |
| `test_provider_factory.py` | Criação correta do provider, model/api_key forwarding, ValueError para providers inválidos |
| `test_generator.py` | Formatação de contexto, delegação ao LLM provider, propagação de exceção |
| `test_vector_store.py` | ChromaDB com embedding provider injetado, busca por similaridade |
| `test_rag_pipeline.py` | Orquestração end-to-end, injeção de dependência, validação de api_key |
| `test_retriever.py` | Recuperação de documentos, search_type, top_k |
| `test_document_loader.py` | Carregamento de PDF, TXT, DOCX, MD e listagem de arquivos |
| `test_text_processor.py` | Chunking e divisão de texto |
| `test_settings.py` | Validação de campos, carregamento de env vars, Pydantic v2 |
| `test_logger.py` | Configuração de logging |

```bash
# Rodar todos os testes
pytest tests/

# Com cobertura
pytest --cov=src tests/

# Apenas os testes de arquitetura (ports, adapters, factory)
pytest tests/test_adapters.py tests/test_provider_factory.py -v
```

---

## 📦 Formatos de Documento Suportados

| Formato | Extensão |
|---|---|
| PDF | `.pdf` |
| Texto puro | `.txt` |
| Word | `.docx`, `.doc` |
| Markdown | `.md` |

---

## 🔒 Boas Práticas de Segurança

- Credenciais passadas por **injeção explícita** — o core nunca lê `os.environ` diretamente
- Nunca versione o arquivo `.env` (já incluso no `.gitignore`)
- Em produção, use secret managers (AWS Secrets Manager, GCP Secret Manager, Vault)
- Faça rotação periódica de chaves de API
- Em ambientes de CI/CD, injete as variáveis via secrets da plataforma

---

## 🗺️ Roadmap

- [ ] Adapter para **Ollama** (LLM e embeddings locais, sem custo de API)
- [ ] Adapter para **Azure OpenAI**
- [ ] Adapter para **HuggingFace** (embeddings open-source)
- [ ] Suporte a **múltiplos vector stores** (Pinecone, Weaviate, pgvector)
- [ ] API REST com FastAPI
- [ ] Interface web (Streamlit ou Gradio)
- [ ] Suporte a conversas com histórico (memória de contexto)

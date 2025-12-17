# Sistema RAG

Um sistema de **Retrieval-Augmented Generation (RAG)** modular e pronto para produção, com ingestão flexível de documentos, arquitetura escalável e suporte a múltiplos formatos de documentos.

---

## 🚀 Funcionalidades

- **Suporte a múltiplos formatos de documentos**: PDF, TXT, DOCX, DOC, Markdown  
- **Arquitetura modular**: Separação clara de responsabilidades com módulos dedicados  
- **Ingestão flexível de documentos**: Arquivos únicos, diretórios ou processamento em lote  
- **Recuperação baseada em vetores**: Busca eficiente por similaridade usando ChromaDB  
- **Geração com LLM**: Modelos OpenAI GPT para respostas inteligentes  
- **Interface CLI**: Ferramentas de linha de comando fáceis de usar  
- **Pronto para produção**: Logging completo, tratamento de erros e gerenciamento de configuração  
- **Gerenciamento seguro de chave de API**: Injeção explícita de API Token  

---

## 📋 Requisitos

- Python 3.9+
- Chave de API da OpenAI

---

## 🔧 Instalação

### Opção 1: Desenvolvimento Local

```bash
    git clone https://github.com/vitoriarntrindade/rag-system.git
    cd rag-system
    
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    
    pip install -r requirements.txt
```

### Opção 2: Como Biblioteca (lib)

```bash
    pip install git+https://github.com/vitoriarntrindade/rag-system.git
```

---

## ⚙️ Configuração

### Configuração da Chave de API (Obrigatória)

O sistema requer uma chave de API da OpenAI. **Ela deve ser fornecida explicitamente** — nunca faça hardcode da chave no código-fonte.

#### Método 1: Variável de Ambiente (Recomendado)

```bash
    export OPENAI_API_KEY='your-api-key-here'
    echo "OPENAI_API_KEY=your-api-key-here" > .env
```

#### Método 2: Carregar do .env no Código

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from src.rag_pipeline import RAGPipeline
pipeline = RAGPipeline(openai_api_key=api_key)
```

#### Método 3: Injeção Direta (somente para testes)

```python
from src.rag_pipeline import RAGPipeline
pipeline = RAGPipeline(openai_api_key="sk-your-key-here")
```

### 🛠️ Explicação das Configurações

Abaixo estão as variáveis de ambiente disponíveis no projeto e o propósito de cada uma:
### 🛠️ Explicação das Configurações

Abaixo estão as variáveis de ambiente disponíveis no projeto e o propósito de cada uma:

```
# ============================
# 🤖 Configurações da OpenAI
# ============================

# 🧠 Modelo utilizado para gerar os embeddings (vetores) dos documentos.
# Esses vetores são usados na busca semântica por similaridade.
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 💬 Modelo de linguagem utilizado para gerar as respostas finais (LLM).
# Responsável por transformar os trechos recuperados em respostas naturais.
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# 🎨 Controla o nível de criatividade das respostas do modelo.
# Valores mais baixos tornam as respostas mais objetivas e determinísticas.
OPENAI_TEMPERATURE=0.3


# ============================
# ✂️ Processamento de Texto
# ============================

# 📏 Tamanho máximo de cada chunk (trecho) de texto gerado a partir dos documentos.
# Valores maiores preservam mais contexto, valores menores melhoram a precisão da busca.
CHUNK_SIZE=1000

# 🔁 Quantidade de caracteres que se sobrepõem entre chunks consecutivos.
# Ajuda a evitar perda de contexto entre trechos.
CHUNK_OVERLAP=200


# ============================
# 🔍 Recuperação (Retrieval)
# ============================

# 🗂️ Número de documentos/trechos mais relevantes que serão recuperados
# para responder a uma pergunta.
RETRIEVAL_TOP_K=5

# 🧭 Tipo de estratégia de busca utilizada no vector store.
# 'similarity' retorna os vetores mais próximos semanticamente da pergunta.
RETRIEVAL_SEARCH_TYPE=similarity


# ============================
# 📋 Logging
# ============================

# 🧾 Define o nível de detalhamento dos logs da aplicação.
# Exemplos: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# 💾 Indica se os logs devem ser gravados em arquivo além da saída no console.
LOG_TO_FILE=true

````

---

## 🎯 Uso

### Uso via CLI

```
bash
    export OPENAI_API_KEY='your-key-here'
```

#### Ingestão de Documentos

```bash
    python main.py ingest --file data/document.pdf
    python main.py ingest --directory data/
    python main.py ingest --directory data/ --file-types pdf txt
    python main.py ingest --directory data/ --no-recursive
    python main.py ingest --file data/document.pdf --force
```

#### Consultas

```bash
    python main.py query "Qual é o tema principal?"
    python main.py query "Explique o conceito" --no-sources
```

#### Chat Interativo

```bash
    python main.py chat
```

---

## 📦 Uso do RAG System como Biblioteca

O **RAG System** foi projetado para funcionar como um **componente reutilizável**, podendo ser facilmente integrado em outros projetos Python, como:

- Chatbots
- APIs backend (FastAPI / Flask)
- Ferramentas internas
- Sistemas de busca semântica
- Aplicações corporativas

### Exemplo: Integração em um Projeto Python

```python
from pathlib import Path
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(openai_api_key="YOUR_API_KEY")

pipeline.ingest_documents(
    directory=Path("data/"),
    file_types=[".pdf", ".txt"],
    recursive=True
)

answer, sources = pipeline.query("O que é mudança climática?")
print(answer)
```

---

### Exemplo: Uso em um Chatbot

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(openai_api_key="YOUR_API_KEY")

while True:
    question = input("Usuário: ")
    if question.lower() in {"sair", "exit", "quit"}:
        break

    answer, _ = pipeline.query(question)
    print(f"Bot: {answer}")
```

---

### Exemplo: Integração com FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG API")

pipeline = RAGPipeline(openai_api_key="YOUR_API_KEY")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_rag(request: QuestionRequest):
    answer, _ = pipeline.query(request.question)
    return AnswerResponse(answer=answer)
```

---

## 🏗️ Arquitetura

```
rag/
├── src/
│   ├── document_loader.py
│   ├── text_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── utils/
│       └── logger.py
├── config/
│   └── settings.py
├── data/
├── db/
├── logs/
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 📦 Formatos Suportados

| Formato | Extensão |
|--------|----------|
| PDF | .pdf |
| Texto | .txt |
| Word | .docx, .doc |
| Markdown | .md |

---

## 🔒 Boas Práticas de Segurança

- Use variáveis de ambiente
- Nunca versione arquivos `.env`
- Use secret managers em produção
- Faça rotação periódica de chaves

---

## 🧪 Testes

A suíte de testes cobre:

- `test_document_loader.py`: Carregamento de documentos, listagem de arquivos e operações com diretórios  
- `test_text_processor.py`: Funcionalidade de chunking e divisão de texto  
- `test_vector_store.py`: Armazenamento vetorial e operações de busca por similaridade  
- `test_retriever.py`: Funcionalidade de recuperação de documentos  
- `test_generator.py`: Geração de respostas via LLM e formatação de prompts  
- `test_rag_pipeline.py`: Orquestração end-to-end do pipeline RAG  
- `test_settings.py`: Gerenciamento e validação de configurações  
- `test_logger.py`: Configuração e setup de logging  

```bash
    pytest tests/
    pytest --cov=src tests/
```
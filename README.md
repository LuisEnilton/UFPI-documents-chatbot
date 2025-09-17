# UFPI-documents-chatbot
ChatBot desenvolvido para facilitar o acesso a informações contidas em documentos oficiais da Universidade Federal do Piauí.

Desenvolvido para a matéria Tópicos em Inteligência Artificial.

## Tecnologias Utilizadas

- Python3
- Langchain
- Poetry
- Tesseract-OCR e Poppler
## Como rodar

### 1 - Instalar o Poetry
https://python-poetry.org/docs/

### 2 - Instalar o Tesseract e o Poppler

#### Linux

`sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils`

#### Windows (Revisar processo foi gerado por IA):
Este é um processo manual.
a. Tesseract: Baixe e instale a partir deste [link](https://github.com/tesseract-ocr/tessdoc/blob/main/Downloads.md). Durante a instalação, marque a opção para adicionar ao PATH do sistema e selecione o idioma "Portuguese".
b. Poppler: Baixe os binários [aqui](https://github.com/oschwartz10612/poppler-windows/releases/), descompacte em um local como C:\poppler-24.02.0\, e adicione a subpasta bin (C:\poppler-24.02.0\bin) ao PATH do sistema.


### 3 - Clonar o projeto

`git clone git@github.com:LuisEnilton/UFPI-documents-chatbot.git`


### 4 - Ative o ambiente virtual

`poetry shell`

### 5 - Instalar as dependências

`poetry install`

### 6 - Configure o .env

Crie um arquivo '.env', copiando do arquivo '.env.sample' e substituindo por sua chave do Gemini. Você também pode configurar diferentes modelos de embedding.

#### Opções de Embeddings

O sistema agora suporta vários provedores de embeddings. Configure no arquivo `.env`:

```
# Provedor de embeddings
EMBEDDING_PROVIDER=local  # Opções: "google", "openai", "huggingface", "local"

# Nome do modelo (opcional)
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

Modelos recomendados:
- **Google**: "models/embedding-001" (requer GOOGLE_API_KEY)
- **OpenAI**: "text-embedding-3-small" (mais econômico) ou "text-embedding-3-large" (requer OPENAI_API_KEY)
- **Local**: "paraphrase-multilingual-MiniLM-L12-v2" (bom para português, não requer API)
- **Hugging Face**: "intfloat/multilingual-e5-small" (modelo multilíngue compacto, não requer API)

> **Nota**: Os provedores "local" e "huggingface" não exigem chave de API, sendo gratuitos, mas baixam o modelo na primeira execução.
>
> **Configuração avançada**: Para usar GPU com modelos locais, configure no arquivo `.env`:
> ```
> EMBEDDING_KWARGS={"encode_kwargs": {"batch_size": 16, "normalize_embeddings": true}, "device": "cpu"}

### 7 - Execute a aplicação

`python app.py` ou `python3 app.py`


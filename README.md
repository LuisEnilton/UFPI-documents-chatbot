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

### 4 - Instalar as dependências

`poetry install`

### 5 - Ative o ambiente virtual

`poetry shell`

### 6 - Configure o .env

Crie um arquivo '.env' , copiando do arquivo '.env.sample' e substituindo por sua chave do gemini

### 7 - Execute a aplicação

`python app.py` ou `python3 app.py`


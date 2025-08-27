import os
import glob
from dotenv import load_dotenv

# Novas importações para o OCR
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

# Importações do LangChain (algumas foram removidas, como o Loader)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def processar_pdf_com_ocr(caminho_do_arquivo: str) -> Document:
    """
    Converte um PDF escaneado em um objeto Document do LangChain usando OCR.
    
    Args:
        caminho_do_arquivo: O caminho para o arquivo PDF.

    Returns:
        Um objeto Document com o texto extraído e metadados.
    """
    print(f"  Processando com OCR: {os.path.basename(caminho_do_arquivo)}...")
    
    # 1. Converte as páginas do PDF em uma lista de imagens
    imagens = convert_from_path(caminho_do_arquivo)
    
    texto_completo = ""
    # 2. Itera sobre cada imagem/página e extrai o texto
    for i, imagem in enumerate(imagens):
        try:
            # lang='por' especifica o idioma português para o Tesseract
            texto_da_pagina = pytesseract.image_to_string(imagem, lang='por')
            texto_completo += texto_da_pagina + "\n\n"
        except Exception as e:
            print(f"    Erro na página {i+1} do arquivo {caminho_do_arquivo}: {e}")
            
    # 3. Retorna um único objeto Document para o PDF inteiro
    return Document(
        page_content=texto_completo,
        metadata={"source": caminho_do_arquivo}
    )

def main():
    # 1. Carregar a chave de API
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("A chave GOOGLE_API_KEY não foi encontrada no arquivo .env")

    print("Configurando a aplicação de RAG com OCR...")

    # 2. Encontrar e processar TODOS os PDFs da pasta 'dados' usando OCR
    caminho_da_pasta = "dados/"
    arquivos_pdf = glob.glob(os.path.join(caminho_da_pasta, "*.pdf"))

    if not arquivos_pdf:
        print(f"Nenhum arquivo PDF encontrado na pasta '{caminho_da_pasta}'.")
        return

    print(f"Encontrados {len(arquivos_pdf)} arquivo(s) PDF. Iniciando processamento OCR...")
    
    # Agora usamos nossa nova função para carregar os documentos
    todos_os_documentos = [processar_pdf_com_ocr(arquivo) for arquivo in arquivos_pdf]

    if not todos_os_documentos:
        print("Não foi possível extrair conteúdo de nenhum dos arquivos PDF.")
        return


    print(f"Todos os documentos {todos_os_documentos}" )

    # 3. Dividir os documentos em pedaços (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(todos_os_documentos)
    
    if not chunks:
        print("\nERRO: Nenhum chunk de texto foi criado. Verifique a qualidade das imagens nos PDFs.")
        return
        
    print(f"\nDocumentos processados e divididos em {len(chunks)} chunks.")

    # 4. Criar Embeddings e a Vector Store (FAISS)
    print("Criando embeddings e a base de vetores...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    # O resto do código permanece o mesmo
    # 5. Criar o Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Aumentei para 5 chunks

    # 6. Definir o LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)

    # 7. Criar o Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Responda à pergunta do usuário com base apenas no contexto fornecido. Seja detalhado e cite a fonte se possível.
    Se a resposta não estiver no contexto, diga "Não encontrei informações sobre isso nos documentos."

    Contexto:
    {context}

    Pergunta: {input}
    """)

    # 8. Criar a Cadeia (Chain) de RAG
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n✅ Aplicação pronta! Faça sua pergunta ou digite 'sair' para terminar.")
    
    # 9. Loop da Interface de Terminal
    while True:
        query = input("\nSua pergunta: ")
        if query.lower() == 'sair':
            break
        if not query.strip():
            continue

        print("Buscando resposta...")
        response = retrieval_chain.invoke({"input": query})
        print("\nResposta:")
        print(response["answer"])

if __name__ == "__main__":
    main()
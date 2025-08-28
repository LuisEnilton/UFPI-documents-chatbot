import os
import glob
from dotenv import load_dotenv

# Importação principal da unstructured
from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def processar_pdf_com_layout(caminho_do_arquivo: str) -> Document:
    """
    Usa a biblioteca 'unstructured' para extrair conteúdo de um PDF,
    preservando a estrutura de tabelas como HTML.
    """
    print(f"  Analisando layout de: {os.path.basename(caminho_do_arquivo)}...")
    
    try:
        # Usa a estratégia "hi_res" para PDFs escaneados e pede para inferir a estrutura da tabela
        elementos = partition_pdf(
            filename=caminho_do_arquivo,
            strategy="hi_res",                # Estratégia de alta resolução para OCR
            infer_table_structure=True,       # Pede para analisar e estruturar tabelas
            languages=['por']                 # Define o idioma para o OCR
        )
        
        conteudo_final = ""
        for el in elementos:
            # Se o elemento for uma tabela, pegamos sua representação em HTML
            if "unstructured.documents.elements.Table" in str(type(el)):
                conteudo_final += "\n" + el.metadata.text_as_html + "\n"
            # Para outros elementos (títulos, parágrafos), pegamos o texto simples
            else:
                conteudo_final += "\n" + el.text + "\n"
                
        return Document(
            page_content=conteudo_final,
            metadata={"source": caminho_do_arquivo}
        )

    except Exception as e:
        print(f"    Erro ao processar {os.path.basename(caminho_do_arquivo)} com unstructured: {e}")
        return None

def main():
    # 1. Carregar a chave de API
    load_dotenv()
    # ... (código para carregar a chave de API permanece o mesmo)
    api_key = os.getenv("GOOGLE_API_KEY")

    print("Configurando a aplicação de RAG com Análise de Layout (unstructured)...")

    # 2. Encontrar e processar todos os PDFs
    caminho_da_pasta = "dados/"
    arquivos_pdf = glob.glob(os.path.join(caminho_da_pasta, "*.pdf"))

    if not arquivos_pdf:
        print(f"Nenhum arquivo PDF encontrado na pasta '{caminho_da_pasta}'.")
        return

    print(f"Encontrados {len(arquivos_pdf)} arquivo(s) PDF. Iniciando análise de layout...")
    
    # Usamos nossa nova e poderosa função de processamento
    docs_processados = [processar_pdf_com_layout(arquivo) for arquivo in arquivos_pdf]
    
    # Filtra qualquer resultado None em caso de erro no processamento
    todos_os_documentos = [doc for doc in docs_processados if doc is not None]

    if not todos_os_documentos:
        print("Não foi possível extrair conteúdo de nenhum dos arquivos PDF.")
        return

    # 3. Dividir os documentos em pedaços (chunks)
    # Importante: Usaremos um separador diferente para HTML
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language="html",
        chunk_size=2000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(todos_os_documentos)
    
    print(f"\nDocumentos processados e divididos em {len(chunks)} chunks.")

    print(todos_os_documentos)


    # 4. Criar Embeddings e a Vector Store (FAISS)
    print("Criando embeddings e a base de vetores...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    # 5. Criar o Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 6. Definir o LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)

    # 7. Criar o Prompt Template (ajustado para mencionar tabelas)
    prompt = ChatPromptTemplate.from_template("""
    Você é um assistente especialista em análise de documentos. Responda à pergunta do usuário com base apenas no contexto fornecido.
    O contexto pode conter textos normais e tabelas em formato HTML. Analise ambos para formular sua resposta.

    Contexto:
    {context}

    Pergunta: {input}
    """)

    # 8. Criar a Cadeia (Chain) de RAG
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n✅ Aplicação pronta! Faça sua pergunta ou digite 'sair' para terminar.")
    
    # 9. Loop da Interface de Terminal
    # ... (O loop de perguntas e respostas permanece exatamente o mesmo)
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
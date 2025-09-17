import os
from dotenv import load_dotenv

# Importação da nossa aplicação modularizada
from src.application.rag_application import RAGApplication


def main():
    # 1. Carregar as variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Configuração do modelo de embedding
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    
    # Parâmetros avançados para embeddings e retriever
    import json
    
    # Tenta carregar parâmetros avançados de embedding do .env (formato JSON)
    embedding_kwargs = None
    if os.getenv("EMBEDDING_KWARGS"):
        try:
            embedding_kwargs = json.loads(os.getenv("EMBEDDING_KWARGS"))
            print(f"Carregados parâmetros avançados para embedding: {embedding_kwargs}")
        except json.JSONDecodeError:
            print("AVISO: EMBEDDING_KWARGS não é um JSON válido, ignorando.")
    
    # Tenta carregar configuração do retriever do .env (formato JSON)
    retriever_config = None
    if os.getenv("RETRIEVER_CONFIG"):
        try:
            retriever_config = json.loads(os.getenv("RETRIEVER_CONFIG"))
            print(f"Carregada configuração do retriever: {retriever_config}")
        except json.JSONDecodeError:
            print("AVISO: RETRIEVER_CONFIG não é um JSON válido, ignorando.")
    
    # Decide qual API key usar
    selected_api_key = api_key
    if embedding_provider == "openai" and openai_api_key:
        selected_api_key = openai_api_key
    
    if not selected_api_key and embedding_provider in ["google", "openai"]:
        print(f"Erro: Chave de API para {embedding_provider} não encontrada no arquivo .env")
        print("Por favor, configure as chaves de API no arquivo .env:")
        print("- GOOGLE_API_KEY para modelos do Google")
        print("- OPENAI_API_KEY para modelos da OpenAI")
        print("Alternativamente, use embedding_provider=local ou embedding_provider=huggingface para modelos locais")
        return

    try:
        # 2. Criar e inicializar a aplicação RAG
        print(f"Usando o modelo de embedding: {embedding_provider}" + 
              (f" ({embedding_model_name})" if embedding_model_name else ""))
        
        app = RAGApplication(
            api_key=selected_api_key, 
            documents_folder="dados/",
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_kwargs=embedding_kwargs,
            retriever_config=retriever_config
        )
        
        app.initialize()
        
        # 3. Executar a aplicação em modo interativo
        app.run_interactive()
        
    except Exception as e:
        print(f"Erro ao executar a aplicação: {e}")


if __name__ == "__main__":
    main()
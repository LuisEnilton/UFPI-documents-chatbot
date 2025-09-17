"""
Módulo principal da aplicação RAG.
"""
import os

from langchain.chains import create_retrieval_chain

from src.readers.document_reader import DocumentReader
from src.models.embedding_model import EmbeddingModel
from src.models.language_model import LanguageModel
from src.storage.vector_store import VectorStore


class RAGApplication:
    """
    Classe principal que orquestra todos os componentes da aplicação RAG.
    """

    def __init__(
            self, 
            api_key: str, 
            documents_folder: str = "dados/", 
            embedding_provider: str = "local", 
            embedding_model_name: str = None,
            embedding_kwargs: dict = None,
            retriever_config: dict = None
        ):
        """
        Inicializa a aplicação RAG.
        
        Args:
            api_key: Chave de API para os serviços do Google.
            documents_folder: Pasta onde estão os documentos a serem processados.
            embedding_provider: Provedor do modelo de embeddings ("google", "openai", "huggingface" ou "local").
            embedding_model_name: Nome do modelo de embeddings a ser usado.
            embedding_kwargs: Argumentos adicionais para o modelo de embeddings.
            retriever_config: Configuração do retriever:
                - search_type: "similarity", "mmr", ou "similarity_score_threshold"
                - k: Número de documentos a recuperar
                - score_threshold: Pontuação mínima para documentos
                - fetch_k: Número inicial de documentos para MMR
                - lambda_mult: Balanço entre relevância e diversidade para MMR (0.0-1.0)
                - filter: Filtros de metadados para a busca
        """
        self.api_key = api_key
        self.documents_folder = documents_folder
        self.retriever_config = retriever_config or {"k": 5}
        
        # Inicializa os componentes
        self.document_reader = DocumentReader(documents_folder)
        
        # Modelo de embeddings com base no provedor especificado
        self.embedding_model = EmbeddingModel(
            api_key=api_key,
            provider=embedding_provider,
            model_name=embedding_model_name,
            model_kwargs=embedding_kwargs
        )
        
        self.language_model = LanguageModel(api_key)
        self.vector_store = VectorStore(self.embedding_model)
        
        # Cadeia RAG
        self.retrieval_chain = None

    def initialize(self):
        """
        Inicializa todos os componentes da aplicação.
        """
        print("Configurando a aplicação de RAG com Análise de Layout...")
        
        # 1. Processa os documentos
        documents = self.document_reader.process_all_documents()
        
        if not documents:
            raise ValueError("Não foi possível processar nenhum documento.")
        
        print(documents)
        
        # 2. Cria o armazenamento de vetores
        print("Criando embeddings e a base de vetores...")
        self.vector_store.create_vector_store(documents)
        
        # 3. Cria o retriever com as configurações especificadas
        retriever = self.vector_store.get_retriever(
            search_type=self.retriever_config.get("search_type", "similarity"),
            k=self.retriever_config.get("k", 5),
            score_threshold=self.retriever_config.get("score_threshold"),
            fetch_k=self.retriever_config.get("fetch_k"),
            lambda_mult=self.retriever_config.get("lambda_mult", 0.5),
            filter=self.retriever_config.get("filter")
        )
        
        # 4. Cria a cadeia de documentos
        document_chain = self.language_model.create_document_chain()
        
        # 5. Cria a cadeia RAG completa
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        print("\n✅ Aplicação pronta!")
        
    def process_query(self, query: str) -> str:
        """
        Processa uma consulta e retorna a resposta.
        
        Args:
            query: Consulta a ser processada.
            
        Returns:
            Resposta à consulta.
        """
        if not self.retrieval_chain:
            raise ValueError("A aplicação ainda não foi inicializada. Chame initialize primeiro.")
            
        print("Buscando resposta...")
        response = self.retrieval_chain.invoke({"input": query})
        
        return response["answer"]
        
    def run_interactive(self):
        """
        Executa a aplicação em modo interativo.
        """
        print("Faça sua pergunta ou digite 'sair' para terminar.")
        
        while True:
            query = input("\nSua pergunta: ")
            if query.lower() == 'sair':
                break
            if not query.strip():
                continue

            try:
                answer = self.process_query(query)
                print("\nResposta:")
                print(answer)
            except Exception as e:
                print(f"Erro ao processar a consulta: {e}")
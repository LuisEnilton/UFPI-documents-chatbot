"""
Módulo para gerenciar armazenamento de vetores.
"""
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorStore:
    """
    Classe responsável por gerenciar o armazenamento de vetores dos documentos.
    """

    def __init__(self, embedding_model):
        """
        Inicializa o armazenamento de vetores.
        
        Args:
            embedding_model: Modelo de embedding a ser usado para gerar os vetores.
        """
        self.embedding_model = embedding_model
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="html",
            chunk_size=4000, 
            chunk_overlap=200
        )

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Divide os documentos em pedaços menores (chunks).
        
        Args:
            documents: Lista de documentos a serem divididos.
            
        Returns:
            Lista de documentos divididos em chunks.
        """
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]):
        """
        Cria o armazenamento de vetores a partir dos documentos.
        
        Args:
            documents: Lista de documentos a serem armazenados.
        """
        chunks = self.create_chunks(documents)
        print(f"Documentos processados e divididos em {len(chunks)} chunks.")
        
        self.vector_store = FAISS.from_documents(
            chunks, 
            embedding=self.embedding_model.get_embeddings_model()
        )
        
    def get_retriever(self, 
                   search_type: str = "similarity", 
                   k: int = 5,
                   score_threshold: float = None,
                   fetch_k: int = None,
                   lambda_mult: float = 0.5,
                   filter: dict = None):
        """
        Cria um retriever para buscar documentos similares.
        
        Args:
            search_type: Tipo de busca a ser usada. 
                - "similarity": Busca padrão por similaridade de cosseno
                - "mmr": Maximum Marginal Relevance - equilibra relevância e diversidade
                - "similarity_score_threshold": Retorna apenas documentos acima de um limite de pontuação
            k: Número de documentos similares a serem recuperados.
            score_threshold: Limite mínimo de pontuação para documentos (usado apenas com "similarity_score_threshold").
            fetch_k: Número de documentos a buscar inicialmente antes de aplicar MMR (usado apenas com "mmr").
            lambda_mult: Balanceamento entre relevância e diversidade para MMR, entre 0.0-1.0.
                - 0.0: Diversidade máxima
                - 1.0: Relevância máxima
            filter: Dicionário para filtrar documentos por metadados.
            
        Returns:
            Retriever para buscar documentos similares.
        """
        if not self.vector_store:
            raise ValueError("O vector store ainda não foi criado. Chame create_vector_store primeiro.")
        
        # Configura os parâmetros de busca
        search_kwargs = {"k": k}
        
        if search_type == "mmr" and fetch_k is not None:
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult
            
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
            
        if filter is not None:
            search_kwargs["filter"] = filter
            
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
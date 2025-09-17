"""
Módulo para gerenciar modelos de embeddings.
"""
import importlib
import warnings
from typing import Optional, Dict, Any, Union, Literal

# Importações principais
from langchain_core.embeddings import Embeddings

# Importação garantida para Google (já está no projeto)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Importações condicionais para outros provedores
def _import_optional_module(module_name: str):
    """Importa um módulo opcional, retornando None se não estiver instalado."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


class EmbeddingModel:
    """
    Classe responsável por gerenciar modelos de embeddings com suporte a diferentes provedores.
    """

    def __init__(self, 
                 api_key: Optional[str] = None, 
                 provider: Literal["google", "openai", "huggingface", "local"] = "local",
                 model_name: Optional[str] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Inicializa o modelo de embeddings.
        
        Args:
            api_key: Chave de API para acessar o serviço de embeddings (para Google e OpenAI).
            provider: Provedor do modelo de embeddings ("google", "openai", "huggingface" ou "local").
            model_name: Nome do modelo de embeddings a ser usado.
                - Para Google: "models/embedding-001"
                - Para OpenAI: "text-embedding-ada-002" ou "text-embedding-3-small" ou "text-embedding-3-large"
                - Para HuggingFace/Local: "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", etc.
            model_kwargs: Argumentos adicionais para o modelo.
                - Google: {"task_type": "retrieval_query" ou "retrieval_document", "title": "título opcional", "cache": True}
                - OpenAI: {"chunk_size": 1000, "timeout": 60, "show_progress_bar": True, "retry_on_rate_limit": True}
                - HuggingFace/Local: {"encode_kwargs": {"batch_size": 32, "show_progress_bar": True, "normalize_embeddings": True}, 
                                     "model_kwargs": {"device": "cuda" ou "cpu"}}
        """
        self.api_key = api_key
        self.provider = provider
        self.embeddings = None
        
        # Configurações padrão por provedor
        if provider == "google" and model_name is None:
            self.model_name = "models/embedding-001"
        elif provider == "openai" and model_name is None:
            self.model_name = "text-embedding-3-small"  # Modelo menor e mais econômico da OpenAI
        elif provider == "huggingface" and model_name is None:
            self.model_name = "intfloat/multilingual-e5-small"  # Bom modelo multilíngue compacto
        elif provider == "local" and model_name is None:
            self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # Bom para português
        else:
            self.model_name = model_name
        
        self.model_kwargs = model_kwargs or {}
        self.initialize_model()

    def initialize_model(self):
        """
        Inicializa o modelo de embeddings com base no provedor selecionado.
        """
        if self.provider == "google":
            if not self.api_key:
                raise ValueError("API key é necessária para o provedor Google")
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name, 
                google_api_key=self.api_key,
                **self.model_kwargs
            )
        
        elif self.provider == "openai":
            if not self.api_key:
                raise ValueError("API key é necessária para o provedor OpenAI")
            
            # Importação condicional para OpenAI
            openai_module = _import_optional_module("langchain_openai")
            if openai_module is None:
                raise ImportError(
                    "O pacote 'langchain-openai' não está instalado. "
                    "Instale com: pip install langchain-openai"
                )
            
            self.embeddings = openai_module.OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                **self.model_kwargs
            )
        
        elif self.provider in ["huggingface", "local"]:
            # Importação condicional para HuggingFace usando o pacote recomendado
            hf_embeddings = None
            
            # Primeiro tenta o novo pacote recomendado langchain_huggingface
            hf_module = _import_optional_module("langchain_huggingface")
            if hf_module is not None and hasattr(hf_module, "HuggingFaceEmbeddings"):
                hf_embeddings = hf_module.HuggingFaceEmbeddings
                
            # Se não encontrar, tenta na langchain_community (alternativa)
            if hf_embeddings is None:
                community_module = _import_optional_module("langchain_community.embeddings")
                if community_module is not None and hasattr(community_module, "HuggingFaceEmbeddings"):
                    warnings.warn(
                        "Usando HuggingFaceEmbeddings da langchain_community, que está depreciado. "
                        "Considere instalar o pacote langchain-huggingface.",
                        DeprecationWarning
                    )
                    hf_embeddings = community_module.HuggingFaceEmbeddings
            
            # Por último, tenta o legado langchain.embeddings (mais antigo)
            if hf_embeddings is None:
                legacy_module = _import_optional_module("langchain.embeddings")
                if legacy_module is not None and hasattr(legacy_module, "HuggingFaceEmbeddings"):
                    warnings.warn(
                        "Usando HuggingFaceEmbeddings do pacote legado langchain, que está depreciado. "
                        "Considere instalar o pacote langchain-huggingface.",
                        DeprecationWarning
                    )
                    hf_embeddings = legacy_module.HuggingFaceEmbeddings
            
            if hf_embeddings is None:
                raise ImportError(
                    "HuggingFaceEmbeddings não encontrado. "
                    "Instale com: pip install sentence-transformers langchain-huggingface"
                )
            
            # Para modelos locais, não é necessário API key
            # Corrigindo o tratamento do parâmetro device, que deve estar dentro de model_kwargs
            model_kwargs = self.model_kwargs.copy() if self.model_kwargs else {}
            
            # Se device foi passado diretamente ou dentro de encode_kwargs, movemos para model_kwargs
            if "device" in model_kwargs:
                device = model_kwargs.pop("device")
                model_kwargs["model_kwargs"] = model_kwargs.get("model_kwargs", {})
                model_kwargs["model_kwargs"]["device"] = device
            
            # Se encode_kwargs existe e contém device, movemos para model_kwargs
            if "encode_kwargs" in model_kwargs and "device" in model_kwargs["encode_kwargs"]:
                device = model_kwargs["encode_kwargs"].pop("device")
                model_kwargs["model_kwargs"] = model_kwargs.get("model_kwargs", {})
                model_kwargs["model_kwargs"]["device"] = device
                
            self.embeddings = hf_embeddings(
                model_name=self.model_name,
                **model_kwargs
            )
        
        else:
            raise ValueError(f"Provedor não suportado: {self.provider}")

    def get_embeddings_model(self) -> Embeddings:
        """
        Retorna o modelo de embeddings inicializado.
        
        Returns:
            Modelo de embeddings inicializado.
        """
        if not self.embeddings:
            self.initialize_model()
            
        return self.embeddings
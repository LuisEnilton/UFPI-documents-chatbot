"""
Módulo para gerenciar modelos de linguagem.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class LanguageModel:
    """
    Classe responsável por gerenciar o modelo de linguagem.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", temperature: float = 0.1):
        """
        Inicializa o modelo de linguagem.
        
        Args:
            api_key: Chave de API para acessar o serviço do modelo de linguagem.
            model_name: Nome do modelo de linguagem a ser usado.
            temperature: Temperatura para controlar a aleatoriedade das respostas (0.0 a 1.0).
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.initialize_model()

    def initialize_model(self):
        """
        Inicializa o modelo de linguagem.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name, 
            google_api_key=self.api_key, 
            temperature=self.temperature
        )

    def get_language_model(self):
        """
        Retorna o modelo de linguagem inicializado.
        
        Returns:
            Modelo de linguagem inicializado.
        """
        if not self.llm:
            self.initialize_model()
            
        return self.llm
        
    def create_document_chain(self, prompt_template=None):
        """
        Cria uma cadeia para processar documentos e responder perguntas.
        
        Args:
            prompt_template: Template de prompt personalizado (opcional).
            
        Returns:
            Cadeia para processar documentos.
        """
        if prompt_template is None:
            prompt_template = """
            Você é um assistente especialista em análise de documentos. Responda à pergunta do usuário com base apenas no contexto fornecido.
            O contexto pode conter textos normais e tabelas em formato HTML. Analise ambos para formular sua resposta.

            Contexto:
            {context}

            Pergunta: {input}
            """
            
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        return create_stuff_documents_chain(self.get_language_model(), prompt)
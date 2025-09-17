"""
Módulo para leitura e processamento de documentos PDF com OCR.
"""
import os
import glob
from typing import List, Optional

# Importação principal da unstructured
from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document


class DocumentReader:
    """
    Classe responsável pela leitura e processamento de documentos PDF,
    incluindo OCR e extração de conteúdo estruturado.
    """

    def __init__(self, documents_folder: str = "dados/"):
        """
        Inicializa o leitor de documentos.
        
        Args:
            documents_folder: Caminho para a pasta onde estão os documentos.
        """
        self.documents_folder = documents_folder

    def get_pdf_files(self) -> List[str]:
        """
        Encontra todos os arquivos PDF na pasta de documentos.
        
        Returns:
            Lista de caminhos para os arquivos PDF encontrados.
        """
        pdf_files = glob.glob(os.path.join(self.documents_folder, "*.pdf"))
        return pdf_files

    def process_pdf(self, file_path: str) -> Optional[Document]:
        """
        Processa um arquivo PDF com OCR e extração de conteúdo estruturado.
        
        Args:
            file_path: Caminho para o arquivo PDF a ser processado.
            
        Returns:
            Document: Documento processado ou None em caso de erro.
        """
        print(f"  Analisando layout de: {os.path.basename(file_path)}...")
        
        try:
            # Usa a estratégia "hi_res" para PDFs escaneados e pede para inferir a estrutura da tabela
            elementos = partition_pdf(
                filename=file_path,
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
                metadata={"source": file_path}
            )

        except Exception as e:
            print(f"    Erro ao processar {os.path.basename(file_path)} com unstructured: {e}")
            return None

    def process_all_documents(self) -> List[Document]:
        """
        Processa todos os arquivos PDF encontrados na pasta de documentos.
        
        Returns:
            Lista de documentos processados.
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            print(f"Nenhum arquivo PDF encontrado na pasta '{self.documents_folder}'.")
            return []

        print(f"Encontrados {len(pdf_files)} arquivo(s) PDF. Iniciando análise de layout...")
        
        # Processamos cada arquivo PDF
        processed_docs = [self.process_pdf(file) for file in pdf_files]
        
        # Filtra qualquer resultado None em caso de erro no processamento
        valid_documents = [doc for doc in processed_docs if doc is not None]

        if not valid_documents:
            print("Não foi possível extrair conteúdo de nenhum dos arquivos PDF.")
            
        return valid_documents
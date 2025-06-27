import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker

from transformers import AutoTokenizer

from typing import List

from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_LLM_MODELS,
)

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from ov_langchain_helper import OpenVINOLLM



#QT PDF view
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtPdfWidgets import QPdfView
from PyQt6.QtPdf import QPdfDocument
from PyQt6.QtCore import QPointF,QTimer
######################################################



int4_model_dir = Path("llama-3.2-3b-instruct") / "INT4_compressed_weights"

embedding_device = "GPU"
rerank_device = "GPU"
llm_device = "GPU"
llm_model_configuration = SUPPORTED_LLM_MODELS["English"]["llama-3.2-3b-instruct"]
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS["English"]["bge-large-en-v1.5"]

embedding_model_name = "bge-large-en-v1.5"
batch_size =4
embedding_model_kwargs = {"device": embedding_device, "compile": False}
encode_kwargs = {
    "mean_pooling": embedding_model_configuration["mean_pooling"],
    "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
    "batch_size": batch_size,
}

embedding = OpenVINOBgeEmbeddings(
    model_name_or_path=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs,
)

embedding.ov_model.compile()
#----------------------------------------
print("Embedding model loaded")
#----------------------------------------


rerank_model_name = "bge-reranker-large"
rerank_model_kwargs = {"device": rerank_device}
rerank_top_n = 2


reranker = OpenVINOReranker(
    model_name_or_path=rerank_model_name,
    model_kwargs=rerank_model_kwargs,
    top_n=rerank_top_n,
)

print("reranker model loaded")
#------------------------------------------------------


tokenizer = AutoTokenizer.from_pretrained(int4_model_dir)
llm = OpenVINOLLM.from_model_path(
    model_path=int4_model_dir,
    device=llm_device,
    tokenizer=tokenizer,
)

llm = OpenVINOLLM.from_model_path(
    model_path=int4_model_dir,
    device=llm_device,
    # tokenizer=tokenizer,
)

print("LLM model loaded")
llm.config.max_new_tokens = 2
llm.invoke("2 + 2 =")

#---------------------------------------

TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
}


LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

text_example_path = "Wizards_of_the_Coast_Essentials_Kit.pdf" #text_example_en.pdf"

rag_prompt_template = llm_model_configuration["rag_prompt_template"]


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text = new_text
    return partial_text


text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)

def create_vectordb(
    docs, spliter_name, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold):
    """
    Initialize a vector database

    Params:
      doc: orignal documents provided by user
      spliter_name: spliter method
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Search rerank top n
      run_rerank: whether run reranker
      search_method: top k search method
      score_threshold: score threshold when selecting 'similarity_score_threshold' method

    """
    global db
    global retriever
    global combine_docs_chain
    global rag_chain
    global db_pages

    if vector_rerank_top_n > vector_search_top_k:
        print("Search top k must >= Rerank top n")

    documents = []
    for doc in docs:
        if type(doc) is not str:
            doc = doc.name
        documents.extend(load_single_document(doc))
        for i, doc in enumerate(documents):
            doc.metadata["page_number"] = i + 1         

    text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding)
    db_pages = FAISS.from_documents(documents, embedding)
    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if run_rerank:
        reranker.top_n = vector_rerank_top_n
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate.from_template(rag_prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


    return "Vector database is Ready"



# initialize the vector store with example document
print("Creating Vector DB..")
create_vectordb(
    [text_example_path],
    "RecursiveCharacter",
    chunk_size=400,
    chunk_overlap=50,
    vector_search_top_k=10,
    vector_rerank_top_n=2,
    run_rerank=True,
    search_method="similarity_score_threshold",
    score_threshold=0.5,
)
print("Vector DB is ready!")

def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt):
    """
    callback function for running chatbot on submit button click

    Params:
    message: new message from user
    history: conversation history
    temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
    top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
    top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
    repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
    active_chat: chat state, if true then chat is running, if false then we should start it here.
    Returns:
    message: reset message and make it ""
    history: updated history with message and answer from chatbot
    active_chat: if we are here, the chat is running or will be started, so return True
    """

    llm.config.temperature = temperature
    llm.config.top_p = 1
    llm.config.top_k = 2
    llm.config.do_sample = temperature > 0.0
    llm.config.max_new_tokens = 2000
    llm.config.repetition_penalty = repetition_penalty
    #print("HISTORY",history)

    partial_text = ""
    streaming_response = rag_chain.stream({"input": history[-1][0]})

    for new_text in streaming_response:
        #print("new_text",new_text)
        if list(new_text.keys())[0] == "answer":
            partial_text = text_processor(partial_text, list(new_text.values())[0])
            
            history[-1][1] = partial_text
            yield history

class PDFViewer(QMainWindow):
    def __init__(self, pdf_path, start_page=0):
        super().__init__()
        self.setWindowTitle("PDF Viewer")

        # Create a QPdfDocument
        self.document = QPdfDocument(self)
        self.document.load(pdf_path)

        # Create a QPdfView to display the document
        self.pdf_view = QPdfView(self)
        self.pdf_view.setDocument(self.document)
        print("page count",self.document.pageCount())

        # Set the initial page
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        if 0 <= start_page < self.document.pageCount():
            QTimer.singleShot(100, lambda: self.pdf_view.pageNavigator().jump(start_page, QPointF(0, 0)))
        #self.pdf_view.setPage(start_page)

        # Layout setup
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.pdf_view)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
app = QApplication(sys.argv)

while True:
    print("\n")
    user_input = input('Question: What are you searching for?\n')
    history = [[user_input, '']]
    page_num = 0
    sentence = ""
    
    retrieved_docs = db_pages.similarity_search_with_score(user_input, k=1)
    filtered_docs = [
    (doc, score) for doc, score in retrieved_docs if score >= 0.3]
    if not filtered_docs:
        print("No relevant documents found above the similarity score threshold.")
    else:
        for doc,score in filtered_docs:
            print("=" * 80)
            page_num = doc.metadata.get('page_number', 'Unknown')
            print(f" Document Source: {doc.metadata.get('source', 'Unknown')}")
            print(f" Page Number: {page_num}")
            #print(f" Score:", score)
    for response in bot(history, 0.1, 1.0, 50, 1.1, True):
        sentence +=  response[-1][1]
        print(sentence.strip(), end="\r", flush=True)

    
    pdf_path = text_example_path  # Replace with your PDF file path
    start_page = page_num  # Open at page 5 (zero-based index)
    
    viewer = PDFViewer(pdf_path, start_page-1)
    viewer.resize(1000, 800)
    viewer.show()
    #app.exec()
    
    #sys.exit(app.exec())
    
    
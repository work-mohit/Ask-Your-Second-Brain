from datetime import datetime
import hashlib
import os
import uuid
import streamlit as st
from glob import glob
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import tempfile
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TEMP_DIR = Path(".tmp/")
TEMP_DIR.mkdir(exist_ok=True)

# You can change these as per you choice
HF_TEXT_GENERATION_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TGM_NUMBER_OF_TOKENS = 4000

TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 150

FAISS_ROOT = Path("faiss_index")
CHAT_HISTORY_TURNS = 4 


def get_session_faiss_path():
    return FAISS_ROOT/st.session_state.session_id


def file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest



def get_pdf_as_documents(uploaded_files):
    docs = []
    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
        temp_dir = Path(temp_dir)

        for uploaded_file in uploaded_files:
            temp_path = temp_dir / uploaded_file.name

            # Write uploaded PDF to temp file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(str(temp_path))
            file_docs = loader.load()
            # saving meta data for later to cite the sources 
            for d in file_docs:
                d.metadata["source_file"] = uploaded_file.name   # this will be used later 

            docs.extend(file_docs)  

    return docs


def text_splitting(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = TEXT_CHUNK_SIZE,
        chunk_overlap= TEXT_CHUNK_OVERLAP
    )
    splitted_text = splitter.split_documents(docs)
    return splitted_text


@st.cache_resource
def get_embedding_model():
    """
    returns the embedding model from Hugging face repo.
    
    :param model_name: repo_id from the Hugging Face. Default `sentence-transformers/all-MiniLM-L6-v2`
    """
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


@st.cache_resource
def get_text_generation_model():
    """
    It return the LLM model Object from hugging face.
    
    :param model_name: Provide the Hugging Face model name/repo id. Default is "meta-llama/Llama-3.1-8B-Instruct"
    :param no_of_tokens: Number of tokens you want to generate. Defaul is 4000.
    """
    llm = HuggingFaceEndpoint(
        repo_id=HF_TEXT_GENERATION_MODEL,
        task="text-generation",
        max_new_tokens=HF_TGM_NUMBER_OF_TOKENS,
        temperature=0.3,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    return ChatHuggingFace(llm=llm)


def build_vector_store(chunks):
    """
    BUG FIX: if two sessions uploaded the same file same chunk which means same cache keys
    Streamlit returned the cached object for the second session WITHOUT re-running the function body --
    which means save_local() never ran for that session's own folder, even
    though the UI showed "PDFs indexed successfully". Caching is removed here;
    the FAISS object now lives in st.session_state, scoped per session
    explicitly instead of relying on an argument-hash cache.
    """
    persist_dir = get_session_faiss_path()
    persist_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vector_store.save_local(persist_dir)
    return vector_store


def process_uploaded_pdfs(pdf_docs):
    """
    return : vector_store and boolean
    vector_store : vector store object which have the indexed data
    boolean: returns if the indexing has been performed or not

    Understanding for future: It checks the hashes of new file in the vector store, 
    Any files which hash doesn't match with the current residing in the vector store, it filters out those
    if nothing new, return the same old vector store with false
    else it break those files in docs and then chunks and later update the vector store with those updated chunks
    In this process it also updates the metadata to keep track of the new files that are coming into the context
    """
    already_indexed = st.session_state.get("indexed_file_hashes", set())
    
    new_files, new_hashes = [], []
    for f in pdf_docs:
        hsh = file_hash(f)
        if hsh not in already_indexed:
            new_files.append(f)
            new_hashes.append(hsh)
    
    if not new_files:
        return st.session_state.get("vector_store"), False

    docs = get_pdf_as_documents(new_files)
    chunks = text_splitting(docs)

    vector_store = st.session_state.get("vector_store")

    if vector_store is None:
        vector_store = build_vector_store(chunks)
    else:
        vector_store.add_documents(chunks)
        persist_dir = get_session_faiss_path()
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(persist_dir)

    st.session_state.vector_store = vector_store
    st.session_state.indexed_file_hashes = already_indexed | set(new_hashes)

    # we have to update meta data also 

    prev_meta = st.session_state.get("index_meta",{"num_files": 0, "num_chunks": 0})

    st.session_state.index_meta = {
        "num_files" : prev_meta["num_files"] + len(new_files),
        "num_chunks" : prev_meta["num_chunks"] + len(chunks),
        "embedding_model": HF_EMBEDDING_MODEL,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return vector_store, True


 # Retrieval + generation 


def format_docs_with_sources(docs):
    """
    it takes the list of Document objects the retriever pulled back (the top-k similar chunks), 
    and turns them into one formatted string like:
    [Source 1: notes.pdf, page 3]
    The quarterly revenue increased by...

    .get(key, default) is a dict method: return metadata["source_file"] if that key exists, 
    otherwise return the fallback string "unknown" instead of raising a KeyError.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "?"
        parts.append(f"[Source {i}: {source}, page {page_label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_chat_history(history):
    if not history:
        return "None"
    lines = []
    for turn in history[-CHAT_HISTORY_TURNS:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def handle_user_question(question):
    rag_chain = build_rag_chain()
    answer = rag_chain.invoke(question)

    st.write("### 🤖 Reply")
    st.write(answer)



def build_rag_chain():
    embeddings = get_embedding_model()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    """
    model = get_text_generation_model() 

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def main():
    st.header("Ask Your Second Brain!")
    user_question = st.text_input("Ask a Question from the PDF Files uploaded..")

    with st.sidebar:
        st.title("Uploads your files here..")

        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ",type=["pdf"], accept_multiple_files=True)
        os.makedirs(DOCS_PATH, exist_ok=True)

        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
                return
            with st.spinner("Processing..."): # user friendly message.
                docs = get_pdf_as_documents(pdf_docs) # get the pdf broken into document object
                text_chunks = text_splitting(docs) # get the text chunks
                create_vector_store(text_chunks) # create vector store
                st.success("PDFs indexed successfully")


    if user_question:
        handle_user_question(user_question)

    st.write("---")

    st.caption("AI App created by @ Mohit Joshi") 




if __name__ == "__main__":
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    main()






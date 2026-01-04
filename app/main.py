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
TEXT_CHUNK_SIZE = 50000
TEXT_CHUNK_OVERLAP = 1000
DOCS_PATH = "docs"



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
            docs.extend(loader.load())

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
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        # task="feature-extraction",
        # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


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
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    return ChatHuggingFace(llm=llm)

def get_session_faiss_path():
    return f"faiss_index_{st.session_state.session_id}"

@st.cache_resource
def create_vector_store(chunks):
    persist_dir_path = os.path.join(os.getcwd(), "faiss_index", get_session_faiss_path())
    os.makedirs(persist_dir_path, exist_ok=True)
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vector_store.save_local(persist_dir_path)
    return vector_store



def handle_user_question(question):
    rag_chain = build_rag_chain()
    answer = rag_chain.invoke(question)

    st.write("### 🤖 Reply")
    st.write(answer)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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






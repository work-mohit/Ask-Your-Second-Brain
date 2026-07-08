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

from footer import render_footer

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

CONDENSE_PROMPT = PromptTemplate(
    template="""Given the conversation history and a follow-up question, rewrite the
follow-up question as a standalone question that contains all the context
needed to search a document store on its own (resolve pronouns like "it" /
"that" / "the second one" using the history). If the follow-up question is
already standalone, or there is no history, return it unchanged. Output ONLY
the rewritten question, nothing else.
 
Chat History:
{chat_history}
 
Follow-up question: {question}
Standalone question:""",
    input_variables=["chat_history", "question"],
)
 
ANSWER_PROMPT = PromptTemplate(
        template="""Answer the question as detailed as possible using ONLY the provided
    context. Make sure to provide all relevant details. If the answer is not in
    the provided context, just say "answer is not available in the context" --
    don't make up information. When you use a fact, mention which numbered Source
    it came from.
    
    Chat History (for continuity only, e.g. resolving pronouns):
    {chat_history}
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:""",
        input_variables=["chat_history", "context", "question"],
    )

def condense_question(question, chat_history, model):
    """
    ENHANCEMENT (see changes.md #4): this is what actually makes multi-turn
    chat work over a vector store. A raw follow-up like "what about its
    limitations?" retrieves garbage on its own -- the vector search has no
    idea what "its" refers to. We first rewrite the question into a
    standalone form using the recent chat history, THEN retrieve with that.
    """
    if not chat_history:
        return question
    chain = CONDENSE_PROMPT | model | StrOutputParser()
    return chain.invoke(
        {"question": question, "chat_history": format_chat_history(chat_history)}
    ).strip()


def build_rag_chain(vector_store):

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    model = get_text_generation_model()

    def run(question:str, chat_history:str):
        standalone_question = condense_question(question, chat_history)
        docs = retriever.invoke(standalone_question)
        context = format_docs_with_sources(docs) 

        answer_chain = ANSWER_PROMPT | model | StrOutputParser()

        answer = answer_chain.invoke({
            "context": context,
            "chat_history": format_chat_history(chat_history),
            "question": standalone_question
        })

        return {"answer": answer, "sources": docs}

    return run


def handle_user_question(question):
    vector_store = st.session_state.get("vector_store")

    if vector_store is None:
        st.warning("Please upload and process at least one PDF first.")
        return

    rag_chain = build_rag_chain(vector_store)

    result = rag_chain(question,st.session_state.chat_history)

    st.session_state.chat_history.append({"question": question, "answer": result["answer"]})
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

# --- UI part 

def render_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used"):
                for i, doc in enumerate(msg["sources"], start=1):
                    source = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page")
                    page_label = page + 1 if isinstance(page, int) else "?"
                    st.caption(f"Source {i}: {source} (page {page_label})")
                    preview = doc.page_content[:300]
                    st.text(preview + ("..." if len(doc.page_content) > 300 else ""))



def main():
    st.header("Ask Your Second Brain!")

    with st.sidebar:
        # st.title("Upload your files here..")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here..",
            type=["pdf"], accept_multiple_files=True,
        )
        # below part of the code might need changes since I enabled the option to add more pdf to the context later in the chat  
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
            else:
                with st.spinner("Processing..."):
                    _, freshly_indexed = process_uploaded_pdfs(pdf_docs)
                if freshly_indexed:
                    st.success("PDFs indexed successfully")
                else:
                    st.info("These exact files are already indexed for this session -- skipped re-embedding.")
        if st.session_state.get("index_meta"):
            meta = st.session_state.index_meta
            st.divider()
            st.caption("📊 Index info")
            st.caption(f"Files: {meta['num_files']} · Chunks: {meta['num_chunks']}")
            st.caption(f"Embedding model: {meta['embedding_model']}")
            st.caption(f"Indexed at: {meta['indexed_at']}")
 
        st.divider()
        if st.button("🗑️ Clear chat & index"):
            for key in ("messages", "chat_history", "vector_store", "indexed_file_hashes", "index_meta"):
                st.session_state.pop(key, None)
            st.rerun()
 
    for msg in st.session_state.messages:
        render_message(msg)

    has_index = st.session_state.get("vector_store") is not None
    chat_placeholder = (
        "Ask a question about your uploaded PDFs..."
        if has_index
        else "Upload and process a PDF in the sidebar first..."
    )
    
    if user_question := st.chat_input(chat_placeholder, disabled=not has_index):
        with st.spinner("Thinking..."):
            handle_user_question(user_question)
        st.rerun()
 
    st.write("---")
    # st.caption("AI App created by Mohit Joshi")
    render_footer()


if __name__ == "__main__":
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    main()






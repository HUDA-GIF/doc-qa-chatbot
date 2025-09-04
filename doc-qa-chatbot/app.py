import streamlit as st
import os
import tempfile
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

class DocumentQA:
    def __init__(self):
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.llm_model = "google/flan-t5-base"
        self.embeddings = None
        self.llm = None
        self.vector_store = None

    def initialize_models(self):
        """Initialize embedding and language models"""
        try:
            # Initialize embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize language model
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1,
                repetition_penalty=1.1,
                device=-1  # Use CPU
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            return True
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return False
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        
        return documents
    
    def process_documents(self, documents: List[Document]):
        """Process documents: split, embed, and create vector store"""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        return len(chunks)
    
    def get_answer(self, question: str) -> str:
        """Get answer to question using retrieval-augmented generation"""
        if not self.vector_store:
            return "Please upload and process documents first."
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        try:
            result = qa_chain({"query": question})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Add source information
            if source_docs:
                sources = "\n\nSources:\n" + "\n".join([
                    f"- Page {doc.metadata.get('page', 'N/A')}: {doc.metadata.get('source', 'Unknown')}"
                    for doc in source_docs[:2]  # Show top 2 sources
                ])
                answer += sources
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    st.title("📚 Document Q&A Chatbot")
    st.markdown("Upload PDF or text documents and ask questions about their content.")
    
    # Initialize document QA system
    if "doc_qa" not in st.session_state:
        st.session_state.doc_qa = DocumentQA()
        with st.spinner("Loading AI models..."):
            if st.session_state.doc_qa.initialize_models():
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please refresh the page.")
                return
    
    # Sidebar for file upload and document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Store uploaded files in session state
            st.session_state.uploaded_files = uploaded_files
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files temporarily
                        temp_files = []
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_files.append(tmp_file.name)
                        
                        # Load and process documents
                        documents = st.session_state.doc_qa.load_documents(temp_files)
                        chunk_count = st.session_state.doc_qa.process_documents(documents)
                        
                        # Clean up temporary files
                        for tmp_file in temp_files:
                            os.unlink(tmp_file)
                        
                        st.session_state.documents_processed = True
                        st.success(f"Processed {len(documents)} documents into {chunk_count} chunks!")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.write(f"📄 {file.name}")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.documents_processed:
                st.warning("Please upload and process documents first.")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.doc_qa.get_answer(prompt)
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("ℹ Instructions")
        st.markdown("""
        1. *Upload Documents*: Use the sidebar to upload PDF or TXT files
        2. *Process Documents*: Click 'Process Documents' to make them searchable
        3. *Ask Questions*: Type your questions in the chat interface
        4. *Get Answers*: The AI will search through your documents and provide answers
        
        *Supported Formats*:
        - PDF documents (.pdf)
        - Text files (.txt)
        
        *Note*: Processing may take a few moments depending on document size.
        """)
        
        # System status
        st.subheader("System Status")
        if st.session_state.documents_processed:
            st.success("✅ Documents processed and ready for questions")
        else:
            st.warning("⏳ Waiting for document upload and processing")
        
        # Model information
        st.subheader("Model Information")
        st.markdown(f"""
        - *Embedding Model*: {st.session_state.doc_qa.embedding_model}
        - *Language Model*: {st.session_state.doc_qa.llm_model}
        - *Device*: CPU (optimized for free deployment)
        """)
if __name__ == "__main__":
    main()
    
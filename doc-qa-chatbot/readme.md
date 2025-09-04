📚 Document Q&A Chatbot

A free, open-source document question-answering chatbot built with LangChain, Hugging Face Transformers, and Streamlit. This application allows users to upload PDF or text documents and ask questions about their content using entirely local AI models - no API keys or paid services required!

✨ Features

· 📄 Multi-format Support: Upload and process PDF and TXT documents
· 🔍 Semantic Search: Local embedding model for accurate document retrieval
· 💬 Interactive Chat: Clean chat interface for asking questions about your documents
· 🆓 Completely Free: No API costs - all models run locally
· 🚀 Easy Deployment: Optimized for Hugging Face Spaces and Streamlit Cloud
· 🔒 Privacy-Focused: Your documents never leave your environment

🛠 Technologies Used

· LangChain: Document loading, chunking, and retrieval
· Hugging Face Transformers: Local embedding and language models
· Streamlit: Web interface and user experience
· FAISS: Vector storage and similarity search
· PyTorch: Model inference backend

📦 Models

· Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (Runs locally)
· Language Model: google/flan-t5-base (Runs locally)

🚀 Quick Start

Prerequisites

· Python 3.8 or higher
· pip package manager
· 2-3GB of free disk space for models (first-time download)

Installation

1. Clone or download this project to your local machine
2. Install the required dependencies:

bash
pip install -r requirements.txt


1. Run the application:

bash
streamlit run app.py


1. Open your web browser and navigate to http://localhost:8501

📖 Usage

1. Upload Documents: Use the sidebar to upload PDF or TXT files
2. Process Documents: Click the "Process Documents" button to make them searchable
3. Ask Questions: Type your questions in the chat interface
4. Get Answers: The AI will search through your documents and provide answers with sources

🌐 Deployment

Option 1: Hugging Face Spaces (Recommended)

1. Create a free account on Hugging Face if you don't have one
2. Go to Hugging Face Spaces and click "Create new Space"
3. Fill in the details:
   · Name your space
   · Select "Streamlit" as the SDK
   · Set visibility to "Public" or "Private"
4. Upload all files from this project or connect your GitHub repository
5. Your app will automatically build and deploy

Option 2: Streamlit Cloud

1. Push your code to a GitHub repository
2. Sign up for a free account on Streamlit Cloud
3. Click "New app" and connect your GitHub repository
4. Select the branch and main file path (app.py)
5. Click "Deploy" and your app will be live

📁 Project Structure


document-qa-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file


⚙ How It Works

1. Document Loading: PDF and text files are loaded using LangChain document loaders
2. Text Chunking: Documents are split into manageable chunks with overlapping content
3. Embedding Generation: Text chunks are converted to vectors using local embedding models
4. Vector Storage: Embeddings are stored in a FAISS vector database for efficient retrieval
5. Question Processing: User questions are embedded and matched with relevant document chunks
6. Answer Generation: The language model synthesizes answers from retrieved content

🔧 Customization

You can easily customize the app by:

· Changing models: Modify the embedding_model and llm_model variables in app.py
· Adjusting chunk size: Modify the chunk_size and chunk_overlap parameters
· Adding file formats: Extend the load_documents method to support more file types

📊 Performance Tips

· For faster processing, split large documents into smaller files
· The app is optimized for CPU usage, making it suitable for free deployment platforms
· First run will take longer as models are downloaded (subsequent runs will be faster)

❓ Frequently Asked Questions

Q: Do I need an API key?
A: No! This application uses entirely local models and doesn't require any API keys.

Q: How much does it cost to run?
A: Nothing! The app is designed to run on free-tier deployment platforms.

Q: What's the maximum document size?
A: It depends on your deployment platform's memory, but typically 10-50MB total is safe.

Q: Are my documents stored or sent anywhere?
A: No! All processing happens locally - your documents never leave your environment.

🐛 Troubleshooting

Issue: Models fail to download
Solution: Check your internet connection and try again. The first run requires downloading ~1-2GB of models.

Issue: App crashes with large documents
Solution: Split large documents into smaller files or use a deployment platform with more memory.

Issue: Answers are not accurate
Solution: Try processing the documents again or rephrasing your question.

🤝 Contributing

Contributions are welcome! Feel free to:

· Report bugs and issues
· Suggest new features
· Submit pull requests
· Improve documentation

📄 License

This project is open source and available under the MIT License.

🙏 Acknowledgments

· LangChain for the amazing framework
· Hugging Face for providing open-source models
· Streamlit for the easy-to-use web framework

📞 Support

If you encounter any problems or have questions:

1. Check the console for error messages
2. Ensure all dependencies are properly installed
3. Verify that you have sufficient storage space for the models

For additional help, please open an issue in the project repository.

---

Happy Document Exploring! 🎉
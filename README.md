AI Legal Assistant



This project is a Flask-based web application that serves as an AI Legal Assistant. It provides a chat interface to answer legal questions and analyze uploaded text documents, using the Groq API for language processing and a FAISS vector database for context retrieval.



🚀 Features

1. Web Chat Interface: A clean, responsive chat UI built with HTML, Tailwind CSS, and JavaScript.

2. LLM Integration: Connects to the Groq API (using the Llama 3 model) to generate fast and intelligent responses.

3. Document Analysis: Users can upload .txt files for summarization and analysis.

4. RAG (Retrieval-Augmented Generation): Uses a local FAISS vector store (faiss_law_db) to find relevant legal context and inject it into the prompt, providing more accurate, grounded answers.

5. Markdown Support: Renders the AI's responses as Markdown, allowing for formatted lists, bold text, and tables.



🛠️ Prerequisites

1. Python 3.8+

2. A Groq API Key

3. A pre-built FAISS vector database. This project assumes you have already created a faiss_law_db folder using langchain with your own legal documents.



📦 Installation

1. Clone the repository:

       git clone <your-repository-url>
       cd <your-project-folder>

2. Create a virtual environment:

       python -m venv venv
       source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install dependencies:

       pip install flask groq langchain-huggingface langchain-community faiss-cpu

Note: faiss-cpu is for CPU-based machines. If you have a compatible GPU, you can install faiss-gpu.



⚙️ Setup & Configuration

1. (CRITICAL) Secure Your API Key

Your code currently has a hardcoded API key. Do not run this in production or make it public.

Fix:
* Create a file named .env in the root of your project.
* Add your API key to it:
.env GROQ_API_KEY=gsk_YOUR_REAL_API_KEY_HERE 
* Install python-dotenv: pip install python-dotenv
* Modify app.py to load the key securely:

      ```python
      # At the top of app.py
      import os
      from dotenv import load_dotenv
      load_dotenv()  # Load variables from .env file

      # ...

      # Initialize Groq API client
      try:
        # Use the environment variable
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        client = groq.Groq(api_key=api_key)
      except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        client = None
      ```

2. Add Your Vector Store

This application will not work without a faiss_law_db folder. You must generate this folder yourself using your own legal documents and langchain. The application looks for it in the same directory as app.py.

    # From app.py
    if os.path.exists("faiss_law_db"):
        db = FAISS.load_local("faiss_law_db", embeddings, allow_dangerous_deserialization=True)
    else:
        db = None
        print("⚠️ No FAISS DB found. Please build one with your law texts first.")

3. Create a .gitignore file

Create a .gitignore file in your project's root directory to protect your keys and environment files:

    venv/
    __pycache__/
    *.env



🏃‍♂️ Run the Application

Once your .env file is set up and your faiss_law_db folder is in place, you can run the app:

    python app.py


Access the application in your browser at: http://127.0.0.1:5000

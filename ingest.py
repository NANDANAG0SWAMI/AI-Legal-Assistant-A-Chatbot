import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup the Embedding Model
# This uses a free, local model to convert text into numbers (vectors)
print("📥 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load your Documents
# Put your legal PDFs or Text files in a folder named 'docs'
documents = []
docs_folder = r"C:\Users\nanda\OneDrive\Desktop\Chatbot\docs"

# Get the absolute path so you know EXACTLY where the folder is
abs_path = os.path.abspath(docs_folder)

if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)
    print(f"📁 Created '{docs_folder}' folder at this location:")
    print(f"👉 {abs_path}")  # <--- LOOK AT THIS PATH IN YOUR TERMINAL
    print("Please put your legal PDFs or .txt files into that folder and run this script again.")
    exit()

print(f"📂 Scanning '{docs_folder}' folder at: {abs_path}")
files_found = False

for file in os.listdir(docs_folder):
    file_path = os.path.join(docs_folder, file)
    
    if file.endswith(".pdf"):
        print(f"  📖 Loading PDF: {file}")
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            files_found = True
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")
            
    elif file.endswith(".txt"):
        print(f"  📄 Loading Text: {file}")
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            files_found = True
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")

if not files_found:
    print(f"❌ No valid documents found in {abs_path}!")
    print("Put some PDFs in there and try again.")
    exit()

# 3. Split Text into Chunks
# AI cannot read a whole book at once. We split it into smaller pieces.
print("✂️  Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"🧩 Split {len(documents)} pages into {len(chunks)} chunks.")

# 4. Create and Save the Vector DB
print("🧠 Building the Brain (Vector DB)... this might take a minute...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_law_db")

print("✅ Success! 'faiss_law_db' folder created.")
print("🚀 Now you can run app.py!")
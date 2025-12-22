import os
from flask import Flask, request, jsonify, render_template
import groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)

# --- CONFIGURATION ---

# 1. Initialize Embeddings
# This processes the text queries. It must match what you used in ingest.py
# We use the CPU-friendly version of the model for standard cloud hosting
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the Brain (FAISS DB)
# IMPORTANT: You must upload the 'faiss_law_db' folder to GitHub for this to work
DB_PATH = "faiss_law_db"
db = None

if os.path.exists(DB_PATH):
    try:
        # allow_dangerous_deserialization is required for loading local pickle files
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ Brain Loaded: FAISS DB is ready.")
    except Exception as e:
        print(f"❌ Error loading Brain: {e}")
else:
    print("⚠️ No FAISS DB found. Please ensure 'faiss_law_db' folder is in your GitHub repo.")

# 3. Initialize Groq Client
# SECURITY UPDATE: We fetch the key from Environment Variables (set this in Render Dashboard)
# Do NOT hardcode the key here for deployment
MY_API_KEY = os.environ.get("GROQ_API_KEY")

client = None
if MY_API_KEY:
    try:
        client = groq.Groq(api_key=MY_API_KEY)
        print("✅ Groq Client Initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize Groq: {e}")
else:
    print("❌ Error: GROQ_API_KEY not found in environment variables.")


# --- ROUTES ---

@app.route("/")
def index():
    # Flask looks for 'index.html' inside the 'templates' folder
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Check if Groq client is ready
    if not client:
        print("❌ Error: Groq client is None during chat request.")
        return jsonify({"error": "Server Error: Groq Client not active. Check API Key in Render settings."}), 500

    user_message = request.form.get("message", "")
    file = request.files.get("file")

    if not user_message and not file:
        return jsonify({"error": "No input provided"}), 400

    try:
        # 1. Prepare the System Prompt
        system_prompt = {
            "role": "system",
            "content": """
            You are an AI Legal Assistant designed to help users understand complex legal concepts.
            - Provide answers in clear, plain English.
            - Use Markdown (bold, lists) to structure your response.
            - Always cite the relevant section of law if provided in the context.
            - If the information is not in the context, state that you do not know. Do not invent laws.
            """
        }

        messages = [system_prompt]
        law_context = ""

        # 2. Handle Text Input (Retrieval from DB)
        if user_message and db:
            print(f"🔍 Searching DB for: {user_message}")
            results = db.similarity_search(user_message, k=3)
            if results:
                # Combine content of top 3 chunks
                law_context = "\n\n".join([r.page_content for r in results])
                print("✅ Found relevant context from DB.")

        # 3. Handle File Input (Reading uploaded text)
        file_content = ""
        if file:
            filename = file.filename
            if filename.lower().endswith('.txt'):
                file_content = file.read().decode('utf-8')
            # Note: For production, you might want to add PDF parsing here

        # 4. Construct the Final Prompt
        final_user_content = f"Question: {user_message}\n"
        
        if law_context:
            final_user_content += f"\n[Relevant Database Knowledge]:\n{law_context}\n"
        
        if file_content:
            final_user_content += f"\n[User Uploaded Document]:\n{file_content}\n"

        messages.append({"role": "user", "content": final_user_content})

        # 5. Get Answer from Groq
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        
        bot_response = chat_completion.choices[0].message.content
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"❌ Error in chat route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # --- DEPLOYMENT AMENDMENT ---
    # 1. Get the PORT from Render's environment variables (default to 5000 only for local testing)
    port = int(os.environ.get("PORT", 5000))
    
    # 2. Set host to '0.0.0.0' to make it accessible from the outside world
    app.run(host="0.0.0.0", port=port)
import os
from flask import Flask, request, jsonify, render_template_string
import groq
import base64
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS DB if exists
if os.path.exists("faiss_law_db"):
    db = FAISS.load_local("faiss_law_db", embeddings, allow_dangerous_deserialization=True)
else:
    db = None
    print("⚠️ No FAISS DB found. Please build one with your law texts first.")

# HTML content (with your full HTML + CSS + JS)
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Legal Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
        .loading p::after {
            content: '.';
            animation: dots 1s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            40% { color: #888; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            60% { text-shadow: .25em 0 0 #888, .5em 0 0 rgba(0,0,0,0); }
            80%, 100% { text-shadow: .25em 0 0 #888, .5em 0 0 #888; }
        }
        #file-preview { max-height: 100px; }
    </style>
</head>
<body class="bg-gray-100 font-sans flex items-center justify-center min-h-screen">
    <div class="chat-container w-full max-w-2xl bg-white rounded-lg shadow-xl flex flex-col h-[80vh]">
        <header class="bg-blue-600 text-white p-4 rounded-t-lg text-center">
            <h1 class="text-2xl font-bold">AI Legal Assistant</h1>
        </header>

        <main id="chat-window" class="flex-1 p-6 overflow-y-auto">
             <div class="message bot-message mb-4 flex">
                 <div class="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3 font-bold text-xs">BOT</div>
                 <div class="bg-blue-100 text-gray-800 p-3 rounded-lg max-w-xs md:max-w-md">
                     <p>
                        Hello! I am an AI Legal Assistant. You can ask me about legal concepts or upload a legal document (.txt) for analysis.
                        <br><br>
                        <strong class="text-red-700">Disclaimer:</strong> I am not a lawyer. My responses are for informational purposes only.
                    </p>
                 </div>
             </div>
        </main>

        <footer class="p-4 border-t border-gray-200 bg-white rounded-b-lg">
             <div id="preview-container" class="hidden mb-2 p-2 border rounded-lg">
                 <div class="flex items-center justify-between">
                     <div class="flex items-center">
                         <img id="file-preview" class="hidden mr-2 rounded" />
                         <span id="file-name" class="text-sm text-gray-600"></span>
                     </div>
                     <button id="remove-file-btn" class="text-red-500 hover:text-red-700">&times;</button>
                 </div>
             </div>
            <form id="chat-form" class="flex items-center">
                <input type="file" id="file-input" class="hidden" accept=".txt">
                <button type="button" id="attach-file-btn" class="mr-2 text-gray-500 hover:text-blue-600 p-2 rounded-full">
                     <svg xmlns="http://www.w.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
                </button>
                <input type="text" id="message-input" class="flex-1 border rounded-full py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Ask a legal question..." autocomplete="off">
                <button type="submit" id="send-button" class="ml-4 bg-blue-600 text-white rounded-full p-3 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors disabled:opacity-50" disabled>
                     <svg xmlns="http://www.w.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
                </button>
            </form>
        </footer>
    </div>

    <script>
        const chatForm = document.getElementById('chat-form');
        const messageInput = document.getElementById('message-input');
        const chatWindow = document.getElementById('chat-window');
        const sendButton = document.getElementById('send-button');
        const fileInput = document.getElementById('file-input');
        const attachFileBtn = document.getElementById('attach-file-btn');
        const previewContainer = document.getElementById('preview-container');
        const fileNameEl = document.getElementById('file-name');
        const filePreviewEl = document.getElementById('file-preview');
        const removeFileBtn = document.getElementById('remove-file-btn');

        function toggleSendButton() {
            const hasText = messageInput.value.trim() !== '';
            const hasFile = fileInput.files.length > 0;
            sendButton.disabled = !hasText && !hasFile;
        }

        messageInput.addEventListener('input', toggleSendButton);
        fileInput.addEventListener('change', toggleSendButton);
        
        attachFileBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileNameEl.textContent = file.name;
                previewContainer.classList.remove('hidden');
            } else {
                previewContainer.classList.add('hidden');
            }
        });

        removeFileBtn.addEventListener('click', () => {
            fileInput.value = ''; 
            previewContainer.classList.add('hidden');
            toggleSendButton();
        });

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = messageInput.value.trim();
            const file = fileInput.files[0];

            if (!message && !file) return;

            if (message) {
                 appendMessage('user', message);
            }
            if (file) {
                appendMessage('user', `<em>Attached: ${file.name}</em>`);
            }

            messageInput.value = '';
            fileInput.value = '';
            previewContainer.classList.add('hidden');
            sendButton.disabled = true;

            const loadingBubble = appendMessage('bot', '...', true);
            
            const formData = new FormData();
            formData.append('message', message);
            if (file) {
                formData.append('file', file);
            }

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                loadingBubble.querySelector('p').innerHTML = marked.parse(data.response);
                loadingBubble.classList.remove('loading');

            } catch (error) {
                console.error('Error:', error);
                loadingBubble.querySelector('p').textContent = `Sorry, something went wrong: ${error.message}`;
                loadingBubble.classList.add('bg-red-100');
                loadingBubble.classList.remove('loading');
            }
        });

        function appendMessage(sender, text, isLoading = false) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', `${sender}-message`, 'mb-4', 'flex');

            let avatar = '';
            let messageBubble = '';

            if (sender === 'user') {
                avatar = `<div class="flex-shrink-0 h-8 w-8 rounded-full bg-gray-500 text-white flex items-center justify-center ml-3 font-bold text-xs order-2">YOU</div>`;
                messageBubble = `<div class="bg-gray-200 text-gray-800 p-3 rounded-lg max-w-xs md:max-w-md order-1"><p>${text}</p></div>`;
                messageDiv.classList.add('justify-end');
            } else {
                avatar = `<div class="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3 font-bold text-xs">BOT</div>`;
                let bubbleClasses = 'bg-blue-100 text-gray-800 p-3 rounded-lg max-w-xs md:max-w-md';
                if (isLoading) {
                    bubbleClasses += ' loading';
                }
                messageBubble = `<div class="${bubbleClasses}"><p>${text}</p></div>`;
            }

            messageDiv.innerHTML = avatar + messageBubble;
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return messageDiv.querySelector('.max-w-xs, .max-w-md');
        }
    </script>
</body>
</html>
"""

# Initialize Groq API client
try:
    client = groq.Groq(api_key="notrealapikeysorry")  # Replace this with your valid API key
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    client = None

# Correct index route to render HTML properly
@app.route("/")
def index():
    return HTML_CONTENT

@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"error": "Groq client not initialized. Check API key."}), 500

    user_message = request.form.get("message", "")
    file = request.files.get("file")

    if not user_message and not file:
        return jsonify({"error": "No message or file provided"}), 400

    try:
        system_prompt = {
            "role": "system",
            "content": """
            You are an AI Legal Assistant. Your purpose is to explain complex legal concepts and analyze legal documents in clear, plain English. You are not a lawyer and cannot provide legal advice.

            **Core Directives:** 
             1. **Persona:** Maintain a professional, objective, and informative tone. Avoid speculation or expressing personal opinions. 
             2. **Document Analysis:** When a document is provided, first identify its type (e.g., Contract, Lease Agreement, NDA). Then, summarize its key clauses, identify the parties involved, and explain their primary rights and obligations. 
             3. **Legal Questions:** When asked a legal question, explain the relevant laws or principles clearly and concisely, citing general legal concepts where applicable. 
             4. **Formatting:** This must never be ignored
                - Use **Markdown** for all responses.  
                - Always make the main **topic title bold**.  
                - Use **headings** for major sections and **subheadings** for structure.  
                - Present rights, obligations, or key terms using **bullet points or numbered lists**.  
                - Where useful, create **tables** to organize comparisons, obligations, or timelines.  
                - Ensure responses are **concise, well-structured, and easy to understand**.  
                - Prioritize **clarity and readability** over excessive detail.          
             """
        }

        model = "llama-3.3-70b-versatile"
        messages = [system_prompt]

        if not file:
            law_context = ""
            if db:
                results = db.similarity_search(user_message, k=3)
                if results:
                    law_context = "\n\n".join([r.page_content for r in results])

            enhanced_prompt = f"""
            Question: {user_message}

            Relevant Indian Law:
            {law_context if law_context else "⚠️ No relevant law found in database."}
            """
            messages.append({"role": "user", "content": enhanced_prompt})

        else:
            filename = file.filename
            file_bytes = file.read()

            if filename.lower().endswith('.txt'):
                file_content = file_bytes.decode('utf-8')

                law_context = ""
                if db:
                    results = db.similarity_search(file_content[:1000], k=3)
                    if results:
                        law_context = "\n\n".join([r.page_content for r in results])

                full_prompt = (
                    f"{user_message}\n\n"
                    f"Analyze the document '{filename}':\n"
                    f"--- Content ---\n{file_content}\n\n"
                    f"Relevant Indian Law:\n{law_context if law_context else '⚠️ No relevant law found in database.'}"
                )
                messages.append({"role": "user", "content": full_prompt})

            else:
                return jsonify({"response": "Unsupported file type. Please upload a .txt file."})

        

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )
        
        bot_response = chat_completion.choices[0].message.content
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

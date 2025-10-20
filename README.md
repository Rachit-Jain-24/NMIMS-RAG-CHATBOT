# 🤖💬 NMIMS Hyderabad FAQ Chatbot  

> **An intelligent FAQ chatbot built with Streamlit and LangChain that answers questions about NMIMS Hyderabad using a PDF-based Retrieval-Augmented Generation (RAG) pipeline.**  
> The chatbot uses `NMIMS_FAQ.pdf` as its knowledge base and provides **context-aware, conversational responses**.  

---

## ✨ **Key Features**

### 📘 **PDF-Based Knowledge**
- Automatically reads and processes the **NMIMS FAQ PDF document**.  
- Optionally supports uploading and analyzing **custom PDFs**.

### 🔍 **RAG (Retrieval-Augmented Generation) Pipeline**
- Combines **LangChain**, **ChromaDB**, and **HuggingFace Embeddings** to retrieve relevant context.  
- Uses a **conversational memory** buffer for contextual responses.  

### 🧠 **Conversational AI**
- Powered by **OpenRouter GPT-3.5-Turbo** (via API).  
- Supports natural, multi-turn chat.  

### ⚡ **Streamlit Interface**
- Clean, intuitive, and interactive chat UI.  
- Cached RAG chain for faster subsequent queries.  

---

## 🧰 **Tech Stack**

| Category | Libraries / Tools |
|-----------|-------------------|
| **Frontend Framework** | Streamlit |
| **Document Loader** | PyPDFLoader (LangChain Community) |
| **Text Processing** | LangChain Text Splitter |
| **Vector Store** | Chroma |
| **Embeddings** | HuggingFace (MiniLM-L6-v2) |
| **LLM Backend** | ChatOpenAI via OpenRouter |
| **Memory** | ConversationBufferMemory |
| **Language** | Python 3.9+ |

---

## ⚙️ **Project Setup and Installation**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/NMIMS-FAQ-Chatbot.git
cd NMIMS-FAQ-Chatbot

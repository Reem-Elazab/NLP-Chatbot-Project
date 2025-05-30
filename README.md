# Jaz Almaza Bay Hotel Chatbot

A **Streamlit-powered conversational AI** for hotel information retrieval, leveraging **Retrieval-Augmented Generation (RAG)** with LangChain, vector search, and Groq-hosted LLMs.

---

## 🌟 Features

- **Conversational Chatbot**: Friendly web UI, remembers chat history per user session.
- **Retrieval-Augmented Generation (RAG)**: Answers questions using actual hotel documentation (PDF), not just LLM “memory.”
- **Context-Aware QA**: Handles follow-up questions by reformulating them for accurate context retrieval.
- **Secure LLM Integration**: Uses Groq-hosted large language models with API key control.
- **Embeddings & Vector Search**: Fast, relevant document lookup using HuggingFace MiniLM and ChromaDB.
- **Session Management**: Each conversation is kept separate via session IDs.

---

## 🧰 Requirements

- Python 3.9+
- streamlit
- langchain
- chromadb
- python-dotenv
- torch
- Groq API Key
- HuggingFace Token
- Your hotel PDF file (default: `Jaz_Almaza_Beach_Resort_Brochure.pdf`)

Install everything with:

```bash
pip install streamlit langchain langchain-community langchain-core langchain-huggingface langchain-groq chromadb python-dotenv torch
```

---

## ⚙️ Setup: Environment Variables

Create a `.env` file in your project directory with the following content:

```
HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
```

- `HF_TOKEN`: [Get from HuggingFace](https://huggingface.co/settings/tokens)
- `GROQ_API_KEY`: [Get from Groq Console](https://console.groq.com/)

❗ **Never commit your `.env` file!**

---

## 🧠 How It Works

1. **PDF Loading**: The brochure is split into overlapping chunks for better search.
2. **Vector Embedding**: Each chunk is converted into embeddings via MiniLM.
3. **Chroma Vectorstore**: Chunks are stored for fast similarity search.
4. **User Input Processing**:
   - Reformulates follow-up questions using history.
   - Retrieves relevant chunks from the PDF.
   - Feeds them + your question into a Groq-hosted LLM.
   - Generates an answer **based on real content**.
5. **Session Memory**: Each session has unique history, tracked via session IDs.

---

## 🚀 Usage

1. **Place your PDF** in the project folder. Rename or update the file path in the code if needed.
2. **Run the App**:
   ```bash
   streamlit run main.py
   ```
3. **Configure in the UI**:
   - Enter a custom Session ID in the sidebar (optional).
4. **Start Chatting**:
   - Ask anything about the hotel: services, hours, amenities, etc.
   - Bot responds with real info from the brochure.

---

## 🔍 Key Code Components

| Section                    | Description                                                |
|---------------------------|------------------------------------------------------------|
| `Streamlit`               | Provides the web interface                                 |
| `PyPDFLoader`             | Loads and parses the hotel PDF                             |
| `RecursiveCharacterTextSplitter` | Splits PDF into chunks for effective retrieval          |
| `HuggingFaceEmbeddings`   | Converts text chunks into vector embeddings                |
| `Chroma`                  | Stores/retrieves chunks using vector similarity            |
| `Groq LLM`                | Generates responses using document-based context           |
| `RAG Chain`               | Combines retrieval + generation                            |
| `RunnableWithMessageHistory` | Handles chat memory and conversation flow              |

---

## 🛠️ Customization

- **Change PDF**: Update `file_path` to use a different brochure.
- **Try Other LLMs**: Change the model name in Groq LLM init.
- **Tune Chunking**: Adjust `chunk_size` and `chunk_overlap` as needed.

---

## 💬 Example Conversation

```
User: What time is breakfast served?
Bot: Breakfast at Jaz Almaza Bay is served daily from 7:00 AM to 10:30 AM.

User: Do you have a spa?
Bot: Yes, the hotel offers a full-service spa with various treatments. Please refer to the brochure for details.

User: Can I check in early?
Bot: Early check-in is subject to availability. Please contact the front desk in advance.
```

---

## 🔐 Security & Privacy

- LLM access is protected via environment-based API keys.
- Session-specific memory ensures chat privacy.
- PDF contents are processed **only locally** and only for the current session.

---

## 🔧 Extending

- 📄 Support multiple PDFs / hotels
- 📤 Add upload option in the UI
- 🌍 Add multilingual support
- ⭐ Enable feedback/rating for answers

---

## 🧾 Credits

Built with:
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/)
- [HuggingFace](https://huggingface.co/)

---

**Enjoy chatting with your Jaz Almaza Bay Hotel assistant! 🏨🤖**

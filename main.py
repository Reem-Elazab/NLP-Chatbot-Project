# Streamlit for UI
import streamlit as st

# Load PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Vector store and retriever
from langchain_community.vectorstores import Chroma

# Chat message memory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Prompt templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chains for retrieval-augmented generation (RAG)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# For persistent history in the runnable pipeline
from langchain_core.runnables import RunnableWithMessageHistory

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Groq LLM wrapper
from langchain_groq import ChatGroq

# Load environment variables from .env
import os
from dotenv import load_dotenv

# Load API keys and tokens
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_APY_KEY'] = os.getenv('GROQ_API_KEY')
GROQ_API = os.getenv('GROQ_API_KEY')

# Initialize embeddings model 
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# App title
st.title("Jaz Hotel Chatbot")

# Sidebar settings
with st.sidebar:
    st.title("Settings")
    session_id = st.text_input("Enter your session Id", value="default_session")

# Ensure API key is correctly loaded
if GROQ_API:
    # Load LLM from Groq
    model_name = "deepseek-r1-distill-llama-70b"
    llm = ChatGroq(api_key=GROQ_API, model=model_name)

    # Initialize in-memory storage for sessions
    if "store" not in st.session_state:
        st.session_state.store = {}

    # Load PDF
    loader = PyPDFLoader(file_path='/Users/hlaakhattab/Downloads/Chat-hotel-RAG/Jaz_Almaza_Beach_Resort_Brochure.pdf')
    data = loader.load()

    # Split text into chunks
    split_data = RecursiveCharacterTextSplitter(chunk_size=200,     # Number of characters per chunk
                                                chunk_overlap=100)  # Number of overlapping characters between chunks 
    splits = split_data.split_documents(data)

    # Create vector store from chunks (generate vector representations and store in Chroma DB)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Return relevant chunks from Chroma based on the query
    retriever = vectorstore.as_retriever()

    # Prompt to reformulate follow-up questions into standalone ones using chat history context
    contextualize_q_system_propmpt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate(
        [('system', contextualize_q_system_propmpt),
         MessagesPlaceholder('chat_history'),
         ('human', "{input}")]
    )

    # Create a retriever that reformulates the query based on chat history,
    # then fetches relevant chunks using the retriever
    history_aware_retriever = create_history_aware_retriever(llm,       
                                                             retriever, 
                                                             contextualize_q_prompt)

    # System prompt for the QA chain: answers hotel-related questions using retrieved context.
    # Instructs the model not to guess if the answer is unknown.
    # "{context}" will be replaced with the retrieved text chunks.
    system_prompt = (
        "You are a Chatbot for a Hotel. Answer all questions related to the hotel only. The hotel name is: Jaz Almaza Bay. "
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    # Generate the final answer using the LLM and context-aware prompt
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full RAG chain: reformulates input if needed, retrieves relevant chunks, and generates the final answer
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Get or initialize the session-specific chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store: 
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Wrap the full RAG chain to support chat history management per session
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,              #Full pipeline                          
        get_session_history,    #Function that return chat history              
        history_messages_key='chat_history', #Store chat history under 'chat_history'
        input_messages_key='input',          #Find user question under 'input'  
        output_messages_key='answer'         #Store the denerated answer under 'answer'
    )

    # Restore previous messages after page reload
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # Keeps chat visible after reload
            st.markdown(message["content"])

    # Capture user input and add it to the session message history
    if prompt := st.chat_input(placeholder='What is the hotel name?'):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if prompt:
            # Generate the assistant's response using the RAG chain and chat history
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': prompt},
                config={'configurable': {'session_id': session_id}}
            )

            # Remove any extra formatting like </think> from the model's output
            assistant_response = response['answer'].split("</think>")[-1]

            print(assistant_response)  # Debugging output

            # Add assistant's response to the session message history and display it
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
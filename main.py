import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_APY_KEY'] = os.getenv('GROQ_API_KEY')
GROQ_API = os.getenv('GROQ_API_KEY')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title("Chatbot with history chat")
with st.sidebar:
    st.title("Settings")

    
    session_id = st.text_input("Enter your session Id",value="default_session")

if GROQ_API:
    model_name = "deepseek-r1-distill-llama-70b"
    llm = ChatGroq(api_key=GROQ_API,model=model_name)
    if "store" not in st.session_state:
        st.session_state.store = {}
    loader = PyPDFLoader(file_path='/Users/hlaakhattab/Downloads/Chat-hotel-RAG/Jaz_Almaza_Beach_Resort_Brochure.pdf')
    data = loader.load()
    split_data = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=100)
    splits = split_data.split_documents(data)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever()    
    contextualize_q_system_propmpt = (
            
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    contextualize_q_prompt = ChatPromptTemplate(
        [('system',contextualize_q_system_propmpt),
        MessagesPlaceholder('chat_history'),
        ('human',"{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    system_prompt = (
            "You are a Chatbot for an Hotel answer all questions related to Hotel only the hotel name is: Jaz Almaza Bay "
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            "{context}"
        ) 
    
    qa_prompt = ChatPromptTemplate.from_messages([
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])
    
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        history_messages_key='chat_history',
        input_messages_key='input',
        output_messages_key='answer'
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):# this is for after reloading the chat interface doesnt go awy from the interface 
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(placeholder='What is the hotel name?'):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if prompt:
            # Get session history and generate response
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': prompt},  # Fixed typo: 'prompt' instead of 'prompt'
                config={'configurable': {'session_id': session_id}}
            )
            
            # Get the assistant's response
            assistant_response = response['answer'].split("</think>")[-1]
            
            print(assistant_response)
            
            # Add assistant response to chat history and display
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
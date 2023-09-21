import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the NiyamBhodak", page_icon="⛏️", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the NiyamBhodak 💬⛏️")
st.info("A Chatbot to help with various Acts, Rules, and Regulations applicable to Mining industries", icon="🤖")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "NiyamBhodak", "content": "Ask me a question about Mining Industry!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Give me 1-2 minutes to tokenize things ."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Indian mining sector and your job is to answer  questions related to various Acts, Rules, and Regulations applicable to Mining industries. Assume that all questions are related to the .Indian Mining industries. Keep your answers True and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Indian mining sector and your job is to answer  questions related to various Acts, Rules, and Regulations applicable to Mining industries. Assume that all questions are related to the .Indian Mining industries. Keep your answers True and based on facts – do not hallucinate features.")
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "NiyamBhodak":
    with st.chat_message("NiyamBhodak"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "NiyamBhodak", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

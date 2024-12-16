import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

# Set up the app title
st.title("SoSE&T GGV Student HelpDesk")
st.markdown("Most of the information is specific to the Department of Information Technology")

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a section:",
    ("Home", "Chat with Chatbot", "About")
)

# Sidebar sections
if option == "Home":
    st.subheader("Welcome to the Student HelpDesk")
    st.write("""
        This platform provides a chatbot to assist students with their queries. 
        You can also create a knowledge base for the chatbot and explore additional features.
    """)
    st.image("Chatbot.png", caption="Student HelpDesk Demo", use_column_width=True)

elif option == "Chat with Chatbot":
    # Chatbot Interface
    st.subheader("Chat with the Chatbot")

    # Button to create knowledge base
    btn = st.button("Create Knowledgebase")
    if btn:
        st.write("Knowledge base created successfully!")
        create_vector_db()

    # Session state for storing messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Prompt for user question
    question = st.chat_input("Question: ")

    # Handle user input
    if question:
        # Display user message
        st.chat_message('user').markdown(question)

        # Add user message to session state
        st.session_state.messages.append({'role': 'user', 'content': question})

        # Process question using the chain
        chain = get_qa_chain()
        response = chain.invoke(question)

        # Display assistant's response
        st.chat_message('assistant').markdown(response["result"])

        # Add assistant's response to session state
        st.session_state.messages.append({'role': 'assistant', 'content': response["result"]})

elif option == "About":
    st.subheader("About the HelpDesk")
    st.write("""
        The SoSE&T GGV Student HelpDesk is an AI-powered chatbot developed to provide instant responses to student queries. 
        The chatbot leverages LangChain's advanced QA chains and a custom knowledge base to deliver accurate and helpful answers.
        
        *Key Features:*
        - Streamlined access to department-specific information.
        - Interactive chatbot interface for real-time communication.
        - Ability to create and update knowledge bases.
    """)
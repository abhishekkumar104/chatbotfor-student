import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

# setup the app title
st.title("SoSE&T GGV Student HelpDesk")
btn = st.button("Create Knowldegebase")
if btn:
    pass

# setup a session state message variable to hold all  the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
question = st.chat_input("Question: ")

# If the user hits enter then 
if question:
    # Display the prompt
    st.chat_message('user').markdown(question)

    # store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content':question})

    # send the prompt to the llm
    chain = get_qa_chain()
    response = chain.invoke(question)

    # show the llm response
    st.chat_message('asssistant').markdown(response["result"])

    # store the LLM response in state
    st.session_state.messages.append({'role':'assistant', 'content':response["result"]})
from openai import OpenAI
import streamlit as st
import sys
from model import *

# Streamlit page settings
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("💬 Singapore Budget 2024 Chatbot")
st.caption("🚀 A RAG-powered Chatbot on the Singapore Budget 2024")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


# load Singapore Budget 2024 data
pdf2text, pdf2textLink = loadData()

# Create vector store
vectorstore = createVectorStore(pdf2text)

# Initialize LLM Client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Default Messages for various purposes.
welcomeMessage = 'Welcome! I am a chatbot with expertise in the Singapore Budget 2024. Please only ask me questions about the Singapore Budget! :smiley:'
defaultRejectionMessage = 'Sorry, would you have other questions about the Singapore Budget 2024 that I can help with?'
outOfTopicMessage = 'Sorry, I am designed to only answer questions about the Singapore Budget 2024. Would you like to ask me another question about the Singapore Budget? smiley:'

# Initialize Streamlit chat history
st = initializeStreamlit (st, outOfTopicMessage, welcomeMessage)

# Wait for user input
if prompt := st.chat_input():

    # add user input to history and print
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write_stream(stream_data(prompt))

    # Check if user input is safe
    user_securityAwareMsg = guardrailAgent(prompt, client, mode='user')

    # Proceed to generate output message
    if user_securityAwareMsg=='Safe':

        # Gnerate a history-aware prompt to best answer the latest user prompt that may have lacked context to retrieve the top K chunks accurately.
        historyAwareMsg = historyAwareAgent(st, client)
        # if rephrasing LLM does not make a response, we will use the user prompt directly without rephrasing
        if historyAwareMsg == '':
            historyAwareMsg = prompt

        # extract top K chunks, includes both Layer 1 RAG and Layer 2 Retrieval Filtering Agent.
        filteredTopK = extractTopChunksWithFiltering(historyAwareMsg, vectorstore, client)

        # build the combined reference list from chunks e.g., grouping chunks to documents
        contextList, contextListDict = buildChunksToReferenceList(filteredTopK, pdf2textLink)

        # Given user prompt and combined reference list of chunks, generate the output to best answer the latest user prompt.
        llm_msg = mainConversationAgent(st, prompt, contextList, client)

        # If no output, send a default rejection message
        if llm_msg=='':
            finalOutput_msg = defaultRejectionMessage
        else:
            # Perform further cleaning of the citations and references e.g. only include references that were cited for clarity.
            finalOutput_msg = processCitations(llm_msg, contextListDict)

    else: # Unsafe or other responses, we will submit a default rejection message
        finalOutput_msg = defaultRejectionMessage

    # Check if final output is safe
    llm_securityAwareMsg = guardrailAgent(finalOutput_msg, client, mode='assistant' )

    # Unsafe or other output will trigger default rejection message to be safe.
    if llm_securityAwareMsg!='Safe':
        finalOutput_msg = defaultRejectionMessage

    # Add to chat history and print to output.
    st.session_state["messages"].append({"role": "assistant", "content": finalOutput_msg})
    with st.chat_message("assistant"):
        st.write_stream(stream_data(finalOutput_msg))


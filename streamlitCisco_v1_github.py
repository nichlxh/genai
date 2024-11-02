import streamlit as st
import numpy as np
import pandas as pd



langSmithKey = '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'


st.set_page_config(page_title="Chatbot", layout="wide")

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


from openai import OpenAI
import streamlit as st

modelType = 'langchain_OpenAI'
# modelType = 'OpenAI'
# modelType = 'huggingface'

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
# with st.sidebar:
#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#     st.button('Run Cisco Model Benchmarking')
huggingFaceKey = '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'
openai_api_key = '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'

if modelType == 'huggingface':


    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=huggingFaceKey
    )



elif modelType == 'OpenAI':


    client = OpenAI(api_key=openai_api_key)
elif modelType == 'langchain_OpenAI':

    import getpass
    import os

    os.environ["OPENAI_API_KEY"] = openai_api_key

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")

    import bs4
    from langchain import hub
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()



st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")


# st.session_state["messages"] is a list of dict
# "role": "assistant",
# "content": "How can I help you?

# which is used to store the conversation history.
# e.g. st.session_state["messages"] [0] is the 1st msg etc.


# initliaze
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# write out to the chat window where necessary, such as how are you.
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# loops on streamlit and triggers when input is received.
# walrus operator := that allows inline assignment of variable + condition checking.
# if there is input to chat window (st.chat_input()), assign it to variable prompt,
# which would be a local variable like prompt = ...
if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()




    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if modelType == 'OpenAI':
        response = client.chat.completions.create(model="gpt-4o", messages=st.session_state["messages"])
        msg = response.choices[0].message.content
    elif modelType == 'huggingface':
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=st.session_state["messages"],
            max_tokens=5000,
            stream=True
        )

        msg = []
        for chunk in stream:
            msg.append(chunk.choices[0].delta.content)

        msg = ''.join(msg)




    elif modelType == 'langchain_OpenAI':
        prompt_fixed = hub.pull("rlm/rag-prompt")



        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_fixed
                | llm
                | StrOutputParser()
        )

        rag_chain.invoke("What is Task Decomposition?")

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # prints the chat window
    # st.session_state["messages"]
    # st.session_state["messages"][0:1]
# Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

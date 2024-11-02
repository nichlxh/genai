import streamlit as st
import numpy as np
import pandas as pd

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


# modelType = 'OpenAI'
modelType = 'huggingface'

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
# with st.sidebar:
#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#     st.button('Run Cisco Model Benchmarking')
if modelType == 'huggingface':
    huggingFaceKey = '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'

    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=huggingFaceKey
    )



elif modelType == 'OpenAI':

    openai_api_key = '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'
    client = OpenAI(api_key=openai_api_key)

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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state["messages"])
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

  
    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # prints the chat window
    # st.session_state["messages"]
    # st.session_state["messages"][0:1]
# Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import streamlit as st
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore

langSmithKey = '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'
huggingFaceKey = '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'
openai_api_key = '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'



st.set_page_config(page_title="Chatbot", layout="wide")

st.title("💬 Chatbot")
st.caption("🚀 A RAG-powered Chatbot")


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



os.environ["OPENAI_API_KEY"] = openai_api_key





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



vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)
# def reset_conversation(st):
#   st.session_state.messages = []

# "Mistral-7B-Instruct-v0.3"
option = st.selectbox(
    "Current LLM (switchable):",
    ("GPT-4o","GPT-4", "o1-Preview", "o1-Mini","Meta-Llama-3-8B-Instruct","Qwen2.5-72B-Instruct"),index=0
)

modelSource=''
if option in ["GPT-4o","GPT-4", "o1-Preview", "o1-Mini"]:
    modelSource='openAI'
else:
    modelSource='huggingFace'

# if option=='GPT-4o':
#     modelType = 'RAG_OpenAI'
# modelType = 'langchain_OpenAI'
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


if modelSource=='huggingFace':


    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=huggingFaceKey
    )


elif modelSource=='openAI':
    client = OpenAI(api_key=openai_api_key)


# st.session_state["messages"] is a list of dict
# "role": "assistant",
# "content": "How can I help you?

# which is used to store the conversation history.
# e.g. st.session_state["messages"] [0] is the 1st msg etc.


# initliaze
# if "messages" not in st.session_state:

# if len(st.session_state.messages) == 0:
#     st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! How may I help you?"}]
#     st.session_state["messages"] = [{}]


# ** IMPORTANT, in general, the flow should always be Optional System/User/Assistant/User/Assistant....
# ** below is needed, as streamlit refreshes everytime when LLM replies etc, hence, it has to reprint everything in st.chat_message
# write out to the chat window where necessary, such as how are you.
if "messages"  in st.session_state:
    st.chat_message("assistant").write('Welcome! How may I help you?')  # only for printing, not stored in memory.
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
else:
    st.chat_message("assistant").write('Welcome! How may I help you?')  # only for printing, not stored in memory.


# loops on streamlit and triggers when input is received.
# walrus operator := that allows inline assignment of variable + condition checking.
# if there is input to chat window (st.chat_input()), assign it to variable prompt,
# which would be a local variable like prompt = ...

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # ** as there is guaranteed an initial prompt here by the user, we can initilize here instead.
    if "messages" not in st.session_state:
        st.session_state["messages"] =  [{"role": "user", "content": prompt}]
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)

    # compute top K chunk to CURRENT prompt (exclude history)
    topK = vectorstore.similarity_search_with_score(prompt)

    filteredTopK = []
    for eachChunk in topK:
        if eachChunk[1]>0.7:
            filteredTopK.append(eachChunk)

    # now ask GPT to respond given Prompt + reference list
    contextListDict = {}
    contextList=''
    for index, eachChunk in enumerate(filteredTopK):
        content = eachChunk[0].page_content
        # clean the content abit
        content = content.replace('\n',' ')
        contextList += '[' + str(index+1) + '] ' + content + '\n\n'
        contextListDict[index+1] = content

    # to ask GPT
    contextPrompt = 'Given the below (a) user prompt and (b) context list sections, answer the prompt by citing the contents in context list, where the citation should be in the format of e.g. [1], [2], [1,2]'
    contextPrompt += 'Try to leverage all of the contexts for the answer where possible.'
    contextPrompt += 'However, if you feel any or all of the contexts in the context list are not helpful or relevant enough to assist you in the building of the answer, then you can ignore them.'
    contextPrompt += 'Note that you do not need to explain yourself if you exclude any context that is not helpful. You should also directly reply to the prompt.'
    contextPrompt += 'While you need to include citations, you do not need to include the reference list or context list sections.'
    contextPrompt += '\n\nPrompt: \n'  + prompt + '\n\n'
    contextPrompt += 'Context List: \n' + contextList + '\n'

    # add role and format
    # here we retrieve the chat history, excluding the current user prompt, replacing it with the context prompt for input to LLM
    # however, in terms of the actual chat history, we instead retain the "actual" chat history with the user prompt (no context list).
    excludeLatestPromptHistory= st.session_state["messages"][:-1]
    finalContextPrompt = excludeLatestPromptHistory + [{"role": "user", "content":  contextPrompt}]
    # st.write (st.session_state["messages"])
    # st.write(st.session_state["messages"][:-1])
    if modelSource=='openAI':
        response = client.chat.completions.create(model=option.lower(), messages=finalContextPrompt)
        msg = response.choices[0].message.content
    elif modelSource=='huggingFace':

        # if optionption in ["Meta-Llama-3-70B-Instruct","gemma-2-27b-it", "Phi-3-mini-128k-instruct","Qwen2.5-72B-Instruct"]:

        model2prefix = {'Meta-Llama-3-8B-Instruct': 'meta-llama/',
                        'Phi-3-mini-4k-instruct': 'microsoft/',
                        'Qwen2.5-72B-Instruct': 'Qwen/',
                        'gemma-2-2b-it': 'google/',
                        'Mistral-7B-Instruct-v0.3': 'mistralai/'}

        prefix = model2prefix[option]
        modelFullName = prefix + option

        stream = client.chat.completions.create(
            model=modelFullName,
            messages=finalContextPrompt,
            max_tokens=2000 ,
            stream=True
        )

        msg = []
        for chunk in stream:
            msg.append(chunk.choices[0].delta.content)

        msg = ''.join(msg)


    # if LLM no make response, we put a place holder reply otherwise it will show as empty
    # further chats can still be made.
    if msg=='':
        msg = 'Sorry, I do not have a response. Please try another prompt.'



    # add back the context list since the response dont have context list and we want to keep the integrity of the context list (instead of being generated)
    # we remove contexts that were not in the response to omit "useless contexts"
    # Regex pattern to match numbers within square brackets
    pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    import re
    # Find all matches
    matches = re.findall(pattern, msg)

    # Process matches to extract individual numbers
    numbers = [int(num) for group in matches for num in re.split(r',\s*', group)]
    numbers = sorted(list(set(numbers)))


    # contextlist to dic
    # for eachContext in contextList:

    outputContextList = ''
    for  eachContextID in numbers:
        if eachContextID in contextListDict:
            outputContextList += '[' + str(eachContextID) + '] ' + contextListDict[eachContextID] + '\n\n'''
        # if '[' +eachContextID in outputContextList:


    if outputContextList!='': #if no retreieved then skip
        msg += '\n\n**Retrieved Context List:** \n\n' + outputContextList + '\n'

    # do this at the end to prevent error in not alternating user and assistant.

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # prints the chat window
    # st.session_state["messages"]
    # st.session_state["messages"][0:1]
# Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

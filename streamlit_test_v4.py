from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import streamlit as st
import getpass
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore





dataDir = 'data'


# from pypdf import PdfReader
import pymupdf # imports the pymupdf library

pdf2text = {}

for eachPDF in os.listdir(dataDir):
    # reader = PdfReader(dataDir + '/' + eachPDF)
    doc = pymupdf.open(dataDir + '/' + eachPDF)  # open a document

    allPageText = ''
    for page in doc:  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        allPageText += text + '\n'
    pdf2text[eachPDF] = allPageText

pdf2textLink = {'ANNEX B-1 - ASSURANCE PACKAGE ENHANCEMENTS.pdf': 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexb1.pdf' ,
'ANNEX B-2 - ENTERPRISE SUPPORT PACKAGE.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexb2.pdf',
'ANNEX C-1 - REFUNDABLE INVESTMENT CREDIT.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexc1.pdf',
'ANNEX C-2 - ENERGY EFFICIENCY GRANT.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexc2.pdf',
'ANNEX D-1 - SKILLSFUTURE LEVEL-UP PROGRAMME.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexd1.pdf',
'ANNEX E-1 - UPLIFTING LOWER-WAGE WORKERS.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexe1.pdf',
'ANNEX E-2 - COMLINK+ PROGRESS PACKAGES.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexe2.pdf',
'ANNEX F-1 - STRENGTHENING AND RATIONALISING OUR RETIREMENT SYSTEM.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf1.pdf',
'ANNEX F-2 - MAJULAH PACKAGE.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf',
'ANNEX F-3 - ONE-TIME MEDISAVE BONUS.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf',
'ANNEX F-4 - ENHANCEMENTS TO SUBSIDY SCHEMES FOR HEALTHCARE AND ASSOCIATED SOCIAL SUPPORT.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf4.pdf',
'ANNEX G-1 - NATIONAL SERVICE LIFESG CREDITS.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexg1.pdf',
'ANNEX G-2 - OVERSEAS HUMANITARIAN ASSISTANCE TAX DEDUCTION SCHEME.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexg2.pdf',
'ANNEX H-1 - TAX CHANGES.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexh1.pdf',
'ANNEX H-2 - FISCAL POSITION FOR FY2024.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexh2.pdf',
'ANNEX I-1 - EXAMPLES FOR BUDGET 2024 STATEMENT.pdf' : 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexi1.pdf',
'Budget_Support_For_Households.pdf' : 'https://www.mof.gov.sg/singaporebudget/resources/support-for-households',
'fy2024_budget_debate_round_up_speech.pdf' : 'https://www.mof.gov.sg/singaporebudget/budget-2024/budget-debate-round-up-speech',
'fy2024_budget_statement.pdf': 'https://www.mof.gov.sg/singaporebudget/budget-2024/budget-statement'}

    # number_of_pages = len(reader.pages)
    # allPageText = ''
    # for eachPageNum in range(number_of_pages):
    #     page = reader.pages[eachPageNum]
    #
    #     text = page.extract_text()
    #     allPageText += text + '\n'
    # pdf2text[eachPDF] = allPageText




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

import time


def stream_data(msg):
    for char in msg:
        yield char
        time.sleep(0.01)









# website = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# # website = 'https://www.mof.gov.sg/singaporebudget/resources/support-for-households'
# # Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(
#     # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     web_paths=(website,),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# for k,v in pdf2text.items():


# create docs from list
allDocContent = list(pdf2text.values())

allDocSource = list(pdf2text.keys())
metaDataSource =[]
for eachDoc in allDocSource:
    metaDataSource.append({'source' : eachDoc,
                           'link': pdf2textLink[eachDoc]})

texts = text_splitter.create_documents(allDocContent, metaDataSource)
splits = text_splitter.split_documents(texts)

# print(splits[8].page_content)


vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings() )
# def reset_conversation(st):
#   st.session_state.messages = []

# "Mistral-7B-Instruct-v0.3"
# option = st.selectbox(
#     "Current LLM (switchable):",
#     ("GPT-4o","GPT-4", "o1-Preview", "o1-Mini","Qwen2.5-72B-Instruct", "Meta-Llama-3-8B-Instruct (Unstable)"),index=0
# )
# if 'Meta' in option:
#     option = option.replace(' (Unstable)','')
#
# modelSource=''
# if option in ["GPT-4o","GPT-4", "o1-Preview", "o1-Mini"]:
#     modelSource='openAI'
# else:
#     modelSource='huggingFace'

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


# if modelSource=='huggingFace':
#
#
#     client = OpenAI(
#         base_url="https://api-inference.huggingface.co/v1/",
#         api_key=huggingFaceKey
#     )


# elif modelSource=='openAI':
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# st.session_state["messages"] is a list of dict
# "role": "assistant",
# "content": "How can I help you?

# which is used to store the conversation history.
# e.g. st.session_state["messages"] [0] is the 1st msg etc.

welcomeMessage = 'Welcome! How may I help you?'
# initliaze
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": welcomeMessage}]
    with st.chat_message("assistant"):
        st.write_stream(stream_data(welcomeMessage))
else:
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
# if len(st.session_state.messages) == 0:
#     st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! How may I help you?"}]
#     st.session_state["messages"] = [{}]


# ** IMPORTANT, in general, the flow should always be Optional System/User/Assistant/User/Assistant....
# ** below is needed, as streamlit refreshes everytime when LLM replies etc, hence, it has to reprint everything in st.chat_message
# write out to the chat window where necessary, such as how are you.


# if "messages"  in st.session_state:
    # with st.chat_message("assistant"):
        # st.write_stream(stream_data(welcomeMessage))
    # st.chat_message("assistant").write(welcomeMessage)  # only for printing, not stored in memory.

        # with st.chat_message(msg["role"]):
        #     st.write_stream(stream_data(msg["content"]))
# else:
    # st.chat_message("assistant").write('Welcome! How may I help you?')  # only for printing, not stored in memory.
    # with st.chat_message("assistant"):
    #     st.write_stream(stream_data(welcomeMessage))

# loops on streamlit and triggers when input is received.
# walrus operator := that allows inline assignment of variable + condition checking.
# if there is input to chat window (st.chat_input()), assign it to variable prompt,
# which would be a local variable like prompt = ...

if prompt := st.chat_input(placeholder="Am I eligible for the Majulah Package?"):
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # ** as there is guaranteed an initial prompt here by the user, we can initilize here instead.
    if "messages" not in st.session_state:
        st.session_state["messages"] =  [{"role": "user", "content": prompt}]

        st.session_state["messages"]
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # st.chat_message("user").write(prompt)
    with st.chat_message("user"):
        st.write_stream(stream_data(prompt))

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
        sourceFile = eachChunk[0].metadata['source']
        link = eachChunk[0].metadata['link']
        # clean the content abit
        content = content.replace('\n',' ')

        # we did not include source file here as it is not necessary.
        contextList += '[' + str(index+1) + '] ' + content + '\n\n'
        # for later reference
        contextListDict[index+1] = (sourceFile,content, link)

    # need to bucket by source fo the  rag content, so that multiple chunk citation, will go to a single doc!

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
    # if modelSource=='openAI':
    response = client.chat.completions.create(model='gpt-4o', messages=finalContextPrompt)
    msg = response.choices[0].message.content

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
            sourceFileName = contextListDict[eachContextID][0]
            # remove pdf format
            sourceFileName = sourceFileName.replace('.pdf','')
            # content = contextListDict[eachContextID][1]
            # url = pdf2textLink[sourceFileName]
            link = contextListDict[eachContextID][2]
            # outputContextList += '[' + str(eachContextID) + '] [' + sourceFileName + '](%s)' % link +  '\n\n'''
            outputContextList += '[' + str(eachContextID) + '] ' + sourceFileName + ': ' + link + '\n\n'''
            # st.write("check out this [link](%s)" % url)
        # if '[' +eachContextID in outputContextList:

    # link = 'https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/fy2024_budget_debate_round_up_speech.pdf'
    if outputContextList!='': #if no retreieved then skip
        msg += '\n\n**References:** \n\n' + '' + outputContextList + ''+ '\n'
        # msg +=  '\n\n[fy2024_budget_debate_round_up_speech.pdf](%s)' %link +'\n\n'+'\n\n**References:** \n\n' + '' + outputContextList + '' + '\n'
    # to add doc names
    # st.write("check out this [link](%s)" % url)
    # do this at the end to prevent error in not alternating user and assistant.

    # final cleaning for presentation
    msg = msg.replace('$', '\$')


    st.session_state["messages"].append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)
    with st.chat_message("assistant"):
        st.write_stream(stream_data(msg))



    # prints the chat window
    # st.session_state["messages"]
    # st.session_state["messages"][0:1]
# Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

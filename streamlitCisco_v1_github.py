import streamlit as st
import numpy as np
import pandas as pd



langSmithKey = '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'





from openai import OpenAI
import streamlit as st

st.set_page_config(page_title="Chatbot", layout="wide")

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")


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


modelType = 'RAG_OpenAI'
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
    # from langchain import hub
    # from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    # from langchain_core.output_parsers import StrOutputParser
    # from langchain_core.runnables import RunnablePassthrough
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

    from langchain_core.vectorstores import InMemoryVectorStore

    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )

    # a[2][1], a[2][0].page_content


    # vectorstore = Chroma.from_documents( documents=splits, embedding=OpenAIEmbeddings())

    # vectorstore = Chroma.afrom_documents("langchain_store", documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")
    # vectorstore2 = Chroma("langchain_store", OpenAIEmbeddings(),persist_directory="./chroma_db" )
    # vectorstore = Chroma(
    #     collection_name="foo",
    #     embedding_function=OpenAIEmbeddings(),
    #     # other params...
    # )
    # from langchain_core.documents import Document
    #
    # # document_1 = Document(page_content="foo", metadata={"baz": "bar"})
    # # document_2 = Document(page_content="thud", metadata={"bar": "baz"})
    # document_1 = Document(page_content=str(docs))
    #
    # documents = [document_1]
    # ids = ["1"]
    # vectorstore.add_documents(documents=documents, ids=ids)




    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
elif modelType=='RAG_OpenAI':
    client = OpenAI(api_key=openai_api_key)

    import getpass
    import os

    os.environ["OPENAI_API_KEY"] = openai_api_key

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")

    import bs4
    # from langchain import hub
    # from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    # from langchain_core.output_parsers import StrOutputParser
    # from langchain_core.runnables import RunnablePassthrough
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

    from langchain_core.vectorstores import InMemoryVectorStore

    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )




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
        # prompt_fixed = hub.pull("rlm/rag-prompt")
        #
        #
        #
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)
        #
        #
        # rag_chain = (
        #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #         | prompt_fixed
        #         | llm
        #         | StrOutputParser()
        # )
        #
        # rag_chain.invoke("What is Task Decomposition?")
        # system_prompt = (
        #     "You are an assistant for question-answering tasks. "
        #     "Use the following pieces of retrieved context to answer "
        #     "the question, explaining and referencing explicitly on how each context was used to build the final answer clearly. Also, show all contexts that were retrieved as a reference list section"
        #     "\n\n"
        #     "Retrieved Contexts:\n"
        #     "{context}"
        # )
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question, explaining and referencing explicitly, based on the index in the reference list, on how each context was used to build the final answer clearly. Also, show all contexts that were retrieved as a reference list section"
            "\n\n"
            "Retrieved Contexts:\n"
            "{context}"
        )
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        promptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, promptTemplate)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        msg = rag_chain.invoke({"input": prompt})
        msg = msg["answer"]
        # print(msg)
    elif modelType == 'RAG_OpenAI':
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
        contextPrompt += 'Prompt: \n'  + prompt + '\n\n'
        contextPrompt += 'Context List: \n' + contextList + '\n'

        # add role and format
        # here we retrieve the chat history, excluding the current user prompt, replacing it with the context prompt for input to LLM
        # however, in terms of the actual chat history, we instead retain the "actual" chat history with the user prompt (no context list).
        excludeLatestPromptHistory= st.session_state["messages"][:-1]
        finalContextPrompt = excludeLatestPromptHistory + [{"role": "assistant", "content":  contextPrompt}]

        response = client.chat.completions.create(model="gpt-4o", messages=finalContextPrompt)
        msg = response.choices[0].message.content

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

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # prints the chat window
    # st.session_state["messages"]
    # st.session_state["messages"][0:1]
# Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

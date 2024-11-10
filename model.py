from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import os
from langchain_core.vectorstores import InMemoryVectorStore
import copy
import pymupdf
from utils import *


def loadData():
    '''
    Function to load the SG budget data and return dictionaries of PDF2text and PDF2textLink.
    '''
    dataDir = 'data'

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

    pdf2text = {}

    for eachPDF in os.listdir(dataDir):
        # open each document
        doc = pymupdf.open(dataDir + '/' + eachPDF)

        allPageText = ''
        for page in doc:  # iterate the document pages
            text = page.get_text()
            allPageText += text + '\n'
        pdf2text[eachPDF] = allPageText

    return pdf2text, pdf2textLink



def createVectorStore (pdf2text):
    '''
    Function to do Extract Transform Load (ETL) to Vector DB.
    :param pdf2text: dict of pdf to text
    :return: vectorstore
    '''

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")

    allDocContent = list(pdf2text.values())

    allDocSource = list(pdf2text.keys())
    metaDataSource =[]
    for eachDoc in allDocSource:
        metaDataSource.append({'source' : eachDoc})

    # Create docs with metadata
    texts = text_splitter.create_documents(allDocContent, metaDataSource)
    splits = text_splitter.split_documents(texts)

    # Load to Vector DB
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings() )

    return vectorstore

def guardrailAgent(message ,client,  mode):
    '''
    Function to initialize Guardrail Agent for both input and output guards.
    :param message: input message from either user or LLM output
    :param client: LLM client
    :param mode: either user or LLM, for the role to be correctly reflected in the chat history
    :return: securityAwareMsg of whether Safe or Unsafe.
    '''
    securityAwareSystemPrompt = (
    'You are a guardrail LLM agent, responsible for checking if the input message or prompt has a high confidence score on any of the following: '
    'Personally identifiable information (PII), toxic language, Not Safe for Work (NSFW) text, profanity, vulgarities, religion, drug, sensitive topics, unusual prompt, security hacking prompt, racial discrimination, dialect discrimination. '
    'It is acceptable if the prompt or message includes the person race as supporting context for the question, as long as there is no comparison, discrimination, and sensitivity involved. '
    'As long as you believe that there is a chance that the input message or prompt could relate to any of the mentioned, you should strictly output only the word \"Unsafe\", else strictly output only the word \"Safe\".'
    )

    # Input Guard
    if mode == 'user':
        securityAwareMessage = [{"role": "system", "content": securityAwareSystemPrompt},
                                {"role": "user", "content": message},]
    # Output Guard
    elif mode == 'assistant':
        securityAwareMessage = [{"role": "system", "content": securityAwareSystemPrompt},
                                {"role": "assistant", "content": message},]

    securityAwareResponse = client.chat.completions.create(model='gpt-4o', messages=securityAwareMessage, seed=888)
    securityAwareMsg = securityAwareResponse.choices[0].message.content

    return securityAwareMsg
def historyAwareAgent(st, client):
    '''
    Function to initialize history-aware rephrasing agent.
    :param st: streamlit object to access chat history
    :param client: LLM client
    :return: historyAwareMsg rephrased user prompt
    '''
    historyAwareSystemPrompt = (
        "You are a rephrasing LLM agent. Given a chat history and the latest user prompt or message, "
        "which might reference context in the chat history, your role is to "
        "rephrase the latest user prompt or message by incoporating the historical context (if any), where the rephrased user message can now be understood "
        "without the chat history. Do NOT answer the question. "
        "If no rephrasing is needed, then just output the latest user prompt or message."
    )

    # replace system message to retain chat history but with a different agent initialization message.
    historyAwareChatHistory = copy.deepcopy(st.session_state["messages"])
    historyAwareChatHistory[0]['content'] = historyAwareSystemPrompt

    historyAwareResponse = client.chat.completions.create(model='gpt-4o', messages=historyAwareChatHistory, seed=888)
    historyAwareMsg = historyAwareResponse.choices[0].message.content

    return historyAwareMsg


def retrievalFilterAgent(message, client):
    '''
    Function to initialize retrievalFilterAgent to filter chunks retrieved from Cosine Similarity via Vector DB.
    :param message: input message with the retrieved chunks
    :param client: LLM client
    :return: retrievalFilterAgentMsg, the relevant citation numbers from the list of chunks
    '''
    retreivalFilterSystemPrompt = (
        'You are a context filtering LLM agent, specifically, given (a) the user prompt and (b) a list of context paragraphs, you are responsible in evaluating which context paragraphs are relevant and supportive in answering the user prompt. '
        'For a given context list, denoted with e.g., [1], [2], [3], if you find that only [2] and [3] are relevant, then you should output \"[2,3]\". '
        'Note that you must strictly output in the format of e.g., [1,2,3] as the final output. '
        'If there are no context shared which you find to be relevant or supportive to answering the user prompt, then you should output \"None\"')

    # only user prompt is needed to be considered here.
    retrievalFilterMessage = [{"role": "system", "content": retreivalFilterSystemPrompt},
                              {"role": "user", "content": message}, ]
    retrievalFilterResponse = client.chat.completions.create(model='gpt-4o', messages=retrievalFilterMessage,
                                                             seed=888)
    retrievalFilterAgentMsg = retrievalFilterResponse.choices[0].message.content

    return retrievalFilterAgentMsg


def mainConversationAgent(st, prompt, contextList, client):
    '''
    Function to initialize mainConversationAgent to conversate with user, where LLM will answer user prompts with the given context list.
    :param st: streamlit object
    :param prompt: user prompt
    :param contextList: list of retrieved chunks
    :param client: LLM client
    :return: Main conversation agent output message
    '''

    contextPrompt = 'Given the chat history, and the below (a) user prompt and (b) context list sections, answer the prompt by citing the contents in context list, where the citation should be in the format of e.g. [1], [2], [1, 2]. '
    contextPrompt += 'Try to leverage all of the contexts for the answer where possible. '
    contextPrompt += 'However, if you feel any or all of the contexts in the context list are not helpful or relevant enough to assist you in the building of the answer, then you can ignore them.'
    contextPrompt += 'Note that you do not need to explain yourself if you exclude any context that is not helpful. You should also directly reply to the prompt. '
    contextPrompt += 'While you need to include citations, you do not need to include the reference list or context list sections.'
    contextPrompt += '\n\nUser Prompt: \n \"' + prompt + '\" \n\n'
    contextPrompt += 'Context List: \n\n' + contextList + '\n'

    # Exclude the latest user prompt as we are instead using our user prompt with context list (i.e., chunks retrieved) to answer the user prompt.
    excludeLatestPromptHistory = st.session_state["messages"][:-1]
    finalContextPrompt = excludeLatestPromptHistory + [{"role": "user", "content": contextPrompt}]

    response = client.chat.completions.create(model='gpt-4o', messages=finalContextPrompt, seed=888)
    msg = response.choices[0].message.content

    return msg


def processCitations(msg, contextListDict):
    '''
    Function to clean the LLM output by:
    1) Add in reference list (instead of being generated by LLM) to the final output
    2) Only keep references that have been cited for clarity.
    3) Reorder the references from 1 onward for better experience.
    :param msg: input message to clean citations
    :param contextListDict: dict of fileID to (sourceFile, content, link)
    :return: output message with fixed citation and references.
    '''

    # find citations in the message
    matches, numbers = extractCitations(msg)

    # Build the reference list
    outputContextList = ''
    for eachContextID in numbers:
        if eachContextID in contextListDict:
            sourceFileName = contextListDict[eachContextID][0]

            # remove pdf format for printing
            sourceFileName = sourceFileName.replace('.pdf', '')

            link = contextListDict[eachContextID][2]
            outputContextList += '[' + str(eachContextID) + '] ' + sourceFileName + ': ' + link + '\n\n'''

    if outputContextList != '':
        msg += '\n\n**References:** \n\n' + '' + outputContextList + '' + '\n'

    # reorder the citations from 1 onward
    citationMappingDict = {}
    for index, eachRefID in enumerate(numbers):
        citationMappingDict[str(eachRefID)] = str(index + 1)

    # find all matches to replace
    matches, _ = extractCitations(msg)

    # replace citations to reorder
    matches = sorted(list(set(matches)))
    for eachMatch in matches:
        toReplace = '[' + eachMatch + ']'
        replacedText = ''
        for eachChar in toReplace:
            if eachChar in citationMappingDict:
                replacedText += citationMappingDict[eachChar]
            else:
                replacedText += eachChar
        msg = msg.replace(toReplace, replacedText)

    return msg


def extractTopChunksWithFiltering(historyAwareMsg, vectorstore, client):
    '''
    Function to extract top chunks based on historyAwareMsg from the vector store.
    :param historyAwareMsg: rephrased user prompt
    :param vectorstore: vector database
    :param client: LLM client
    :return: list of top chunks
    '''

    topK = vectorstore.similarity_search_with_score(historyAwareMsg)

    simThreshold = 0.4
    filteredTopK = []
    for eachChunk in topK:
        if eachChunk[1] >= simThreshold:
            filteredTopK.append(eachChunk)

    # ====== LLM Chunk Filtering =======

    # build the context list for the agent to filter
    retrievalFilterAgentContextList = ''
    for index, eachChunk in enumerate(filteredTopK):
        chunkContent = eachChunk[0].page_content
        chunkContent = chunkContent.replace('\n', ' ')
        retrievalFilterAgentContextList += '[' + str(index + 1) + '] ' + chunkContent + '\n\n'

    retrievalFilterAgentInputMsg = historyAwareMsg + '\n\n' + 'Context List: \n\n' + retrievalFilterAgentContextList
    retrievalFilterAgentMsg = retrievalFilterAgent(retrievalFilterAgentInputMsg, client)

    # filter agent find none of the chunks are relevant.
    if retrievalFilterAgentMsg == 'None':
        filteredTopK = []


    # filter agent either found all or some of the chunks to be relevant.
    else:

        matches, numbers = extractCitations(retrievalFilterAgentMsg)

        # if valid LLM output AND if indeed there is a reduction of chunks
        if (len(numbers) > 0) and (len(numbers) < len(filteredTopK)):

            # Extract chunks from which the agent decided to keep.
            tempFilteredTopK = []
            for eachRelevantChunkID in numbers:
                tempFilteredTopK.append(filteredTopK[eachRelevantChunkID - 1])

            # replace with Agent filtered chunks
            filteredTopK = tempFilteredTopK

    return filteredTopK


def buildChunksToReferenceList(filteredTopK, pdf2textLink):
    '''
     Function to build chunks to reference list, such as by grouping the different chunks to the respective documents.
    :param filteredTopK: list of top chunks
    :param pdf2textLink: dict of pdf to web link
    :return: contextList of grouped references containing chunks,
             contextListDict dict of fileID to (sourceFile, content, link)
    '''
    # dict to group chunks to document.
    refFile2content = {}
    for index, eachChunk in enumerate(filteredTopK):
        content = eachChunk[0].page_content
        sourceFile = eachChunk[0].metadata['source']
        content = content.replace('\n', ' ')

        if sourceFile not in refFile2content:
            refFile2content[sourceFile] = ''

        refFile2content[sourceFile] += '- \"' + content + '\"\n'

    # dict to store fileID to document name, its content, and its web link.
    contextListDict = {}
    contextList = ''
    for fileID, (sourceFile, content) in enumerate(refFile2content.items()):
        contextList += '[' + str(fileID + 1) + '] \"' + sourceFile + '\": \n\n' + content + '\n\n'

        link = pdf2textLink[sourceFile]
        contextListDict[fileID + 1] = (sourceFile, content, link)

    return contextList, contextListDict

def initializeStreamlit (st,outOfTopicMessage,welcomeMessage):
    '''
    Function to initialize Streamlit chat history and default Main Conversation agent system prompt.
    :param st: streamlit object
    :param outOfTopicMessage: default out of topic message
    :param welcomeMessage:  default welcome message
    :return: initialized streamlit object
    '''
    if "messages" not in st.session_state:
        systemPrompt = ('You are a friendly and enthusiastic conversational question-answering agent to help country citizens to learn more about the Singapore Budget 2024. '
                        'The Singapore Budget 2024 includes many initiatives, payouts, and benefits to the citizens of Singapore. '
                        'You are only allowed to answer questions that are about the Singapore Budget 2024. '
                        'Questions to better understand the budget, such as its initiatives, payouts, and benefits, must be answered. '
                        'if the user is asking questions with references to the chat history, that is about the Singapore Budget 2024, then you should respond since the reference is about the Singapore Budget 2024. '
                        'If you feel that there are questions by the user that are not for the purpose of better understanding or related to the Singapore Budget 2024, then you should say '
                        '\"' + outOfTopicMessage + '\" ')
        st.session_state["messages"] = [{"role": "system", "content": systemPrompt}]
        st.session_state["messages"].append({"role": "assistant", "content": welcomeMessage})
        with st.chat_message("assistant"):
            st.write_stream(stream_data(welcomeMessage))
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] != "system":
                # we do not stream here for user experience.
                st.chat_message(msg["role"]).write(cleanOutput(msg["content"]))
    return st

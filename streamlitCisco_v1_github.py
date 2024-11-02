import streamlit as st
import numpy as np
import pandas as pd

import os
import sys

import plotly.express as px
import copy

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# os.chdir('/home/nicholaslim/projects/questionAnswering')
# sys.path.append('/home/nicholaslim/projects/questionAnswering')

import math  # round up when >0.5 else round down


def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals


st.set_page_config(page_title="Which LLM to Use?", layout="wide", page_icon='ciscoLogo.svg')
st.markdown(""" 
<style>
.big-font {
    font-size:20px !important;
    display:inline; 

}  

</style> 
""", unsafe_allow_html=True)
# .st-emotion-cache-100bu1d{
#     min-height: 0rem !important;
#
# }
# .st-emotion-cache{
#     gap: 0rem !important;
#
# }
# .st-emotion-cache-zhoa2m{
#     gap: 0rem !important;
#
# }
# div[data-testid="stVerticalBlockBorder"][class^="st-emotion-cache-184rwn5"] {
#     gap: 0rem;
# }

# [class^="st-emotion-cache-vrbkgx"] {
#     gap: 0rem;
# }
# div[data-testid="stVerticalBlock"][class^="st-emotion-cache-"] {
#     gap: 0rem;
# }
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

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem; 
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


def changeModelNameToCompanyName(df):
    modelList = df['Model'].tolist()
    modelList = [i[i.index('[') + 1: i.index('[') - 1] for i in modelList]
    df['Model'] = modelList
    return df


st.title("Which LLM to Use?")
logCol1, logCol2, logCol3, logCol4, logCol5, _, _ = st.columns(7)
with logCol1:
    # st.image('mainLogo.png', width=500)
    with open("mainLogo.svg") as logo_file:
        logo = logo_file.read()
    st.image(logo, width=550)
# with logCol2:
#     st.image('ciscoLogo.png', width=330)
# with logCol3:

# overall color
# color_discrete_map = {
#     '[OpenAI] GPT-4o': '#0909FF',
#     '[OpenAI] GPT-4-Turbo': '#1974D2',
#     '[OpenAI] GPT-4': '#488AC7',
#     '[OpenAI] GPT-4o-Mini': '#659EC7',
#     '[OpenAI] GPT-3.5-Turbo': '#87AFC7',
#     '[Alibaba] Calme-2.4-rys-78b': '#F5E216',
#     '[Alibaba] Qwen2-72B-Instruct': '#FFCE44',
#     '[Alibaba] RYS-XLarge': '#F6BE00',
#     '[Google] Gemma-2-27b-it': '#000FA9A',
#     '[Meta] Llama-3.1-70B-Instruct': '#FF6347'
# }

# replaced 4o mini and 3.5 turbo.
color_discrete_map = {
    '[OpenAI] o1-preview': '#0909FF',
    '[OpenAI] o1-mini': '#1974D2',
    '[OpenAI] GPT-4o': '#488AC7',
    '[OpenAI] GPT-4-Turbo': '#659EC7',
    '[OpenAI] GPT-4': '#87AFC7',
    '[Alibaba] Calme-2.4-rys-78b': '#F5E216',
    '[Alibaba] Qwen2-72B-Instruct': '#FFCE44',
    '[Alibaba] RYS-XLarge': '#F6BE00',
    '[Google] Gemma-2-27b-it': '#008000',
    '[Meta] Llama-3.1-70B-Instruct': '#FF6347',
    '[Nvidia] Llama-3.1-Nemotron-70B': '#800080'
}

st.caption("🚀 Evaluating Large Language Models (LLMs) for Organizational Use")

# ordered is followed for website printing + also used for others to process all the models
# globalModelOrder = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-4o-mini', 'gpt-3.5-turbo', 'calme-2.4-rys-78b',
#                     'Qwen2-72B-Instruct', 'RYS-XLarge', 'gemma-2-27b-it', 'Meta-Llama-3.1-70B-Instruct']
# globalModelOrderForDisplay = ['[OpenAI] GPT-4o', '[OpenAI] GPT-4-Turbo', '[OpenAI] GPT-4', '[OpenAI] GPT-4o-Mini',
#                               '[OpenAI] GPT-3.5-Turbo', '[Alibaba] Calme-2.4-rys-78b', '[Alibaba] Qwen2-72B-Instruct',
#                               '[Alibaba] RYS-XLarge', '[Google] Gemma-2-27b-it', '[Meta] Llama-3.1-70B-Instruct']

globalModelOrder = ['o1-preview', 'o1-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'calme-2.4-rys-78b',
                    'Qwen2-72B-Instruct', 'RYS-XLarge', 'gemma-2-27b-it', 'Meta-Llama-3.1-70B-Instruct', 'Llama-3.1-Nemotron-70B']
globalModelOrderForDisplay = ['[OpenAI] o1-preview', '[OpenAI] o1-mini', '[OpenAI] GPT-4o', '[OpenAI] GPT-4-Turbo',
                              '[OpenAI] GPT-4', '[Alibaba] Calme-2.4-rys-78b', '[Alibaba] Qwen2-72B-Instruct',
                              '[Alibaba] RYS-XLarge', '[Google] Gemma-2-27b-it', '[Meta] Llama-3.1-70B-Instruct', '[Nvidia] Llama-3.1-Nemotron-70B']

# globalModelOrderForDisplay = ['[OpenAI] GPT-4o', '[OpenAI] GPT-4-Turbo', '[OpenAI] GPT-4', '[OpenAI] GPT-4o-Mini','[OpenAI] GPT-3.5-Turbo','[Alibaba] Calme-2.4-rys-78b','[Alibaba] Qwen2-72B-Instruct','[Alibaba] RYS-XLarge','[Google] Gemma-2-27b-it','[Meta] Llama-3.1-70B-Instruct']
# from openai import OpenAI

totalNumberOfModelsConsidered = len(globalModelOrderForDisplay)

import streamlit as st

# read in all model pred DF
combinedDF = pd.DataFrame()

modelDF = {}
for modelName, modelNameDisplay in zip(globalModelOrder, globalModelOrderForDisplay):
    # for modelName in ['gpt-4o','gpt-4o-mini','gpt-4-turbo','gpt-4','gpt-3.5-turbo']:
    if modelName == 'Llama-3.1-Nemotron-70B': # because too long, so we reduced the name, but still needed for file reading, hence added here.
        modelName = 'Llama-3.1-Nemotron-70B-Instruct-HF'

    modelDF[modelNameDisplay] = pd.read_excel('outputs/ciscoBooks_QA_combined_pred_' + modelName + '_Cleaned2.xlsx')
    # modelDF[modelName] = modelDF[modelName].replace('‚Äôs','\'s', regex=True)
    modelDF[modelNameDisplay]['ModelName'] = modelNameDisplay

    combinedDF = pd.concat([combinedDF, modelDF[modelNameDisplay]], ignore_index=True)
    # combinedDF = combinedDF.append(modelDF[modelName], ignore_index=True)
    # print(len(combinedDF))

combinedDF['QuestionType'] = combinedDF['predictedQuestionType']
combinedDF.drop(['predictedQuestionType'], axis=1, inplace=True)


def createTab(tabChoice, questionType, combinedDF, modelDF):
    # filter by question type section
    combinedDF_internal = combinedDF.copy()
    modelDF_internal = copy.deepcopy(modelDF)
    if questionType != 'Overall':  # filter except first tab
        combinedDF_internal = combinedDF_internal[combinedDF_internal['QuestionType'] == questionType]
        for modelName in globalModelOrderForDisplay:
            modelDF_internal[modelName]['QuestionType'] = modelDF_internal[modelName]['predictedQuestionType']
            modelDF_internal[modelName].drop(['predictedQuestionType'], axis=1, inplace=True)

            modelDF_internal[modelName] = modelDF_internal[modelName][
                modelDF_internal[modelName]['QuestionType'] == questionType]

    with (tabChoice):
        chart_data1 = pd.DataFrame(columns=['Accuracy', 'Model'])

        accScore = []
        index = 0
        for modelName in globalModelOrderForDisplay:
            acc = sum(modelDF_internal[modelName]['EM']) / len(modelDF_internal[modelName]['EM'])
            # acc = round(acc*100,1)
            accScore.append(acc)

            chart_data1.loc[index] = acc, modelName
            index += 1

        # import decimal
        # # import decimal
        # decimal.getcontext().rounding = decimal.ROUND_CEILING

        # def round_up(x, place=0):
        #     context = decimal.getcontext()
        #     # get the original setting so we can put it back when we're done
        #     original_rounding = context.rounding
        #     # change context to act like ceil()
        #     context.rounding = decimal.ROUND_CEILING
        #
        #     rounded = round(decimal.Decimal(str(x)), place)
        #     context.rounding = original_rounding
        #     return float(rounded)

        col1_title, col1_metric, col2_metric, col3_metric, col4_metric = st.columns(spec=[0.4, 0.15, 0.15, 0.15, 0.15])
        if questionType != 'Overall':
            with col1_title:
                st.markdown('<h3>' + questionType + '</h3>', unsafe_allow_html=True)
        else:
            with col1_title:
                st.markdown('<h3>' + questionType + ' Results</h3>', unsafe_allow_html=True)

        col1_metric.metric("Best Model Result", "  " + str(normal_round(max(accScore) * 100, 1)) + '%')
        col2_metric.metric("Total Domains",
                           "   " + str(len(modelDF_internal['[OpenAI] o1-preview']['Domain'].unique())))
        col3_metric.metric("Total Topics",
                           "   " + str(len(modelDF_internal['[OpenAI] o1-preview']['ChapterTitle'].unique())))
        col4_metric.metric("Total Questions", "  " + str(len(modelDF_internal['[OpenAI] o1-preview'])))

        def mainOverallPerformance(chart_data1):
            # chart_data1 = changeModelNameToCompanyName(chart_data1)
            # newnames = {'[OpenAI] GPT-4o': 'OpenAI', '[OpenAI] GPT-4-Turbo': 'OpenAI', '[OpenAI] GPT-4': 'OpenAI',
            #                               '[OpenAI] GPT-4o-Mini': 'OpenAI', '[OpenAI] GPT-3.5-Turbo': 'OpenAI',
            #                               '[Alibaba] Calme-2.4-rys-78b': 'Alibaba', '[Alibaba] Qwen2-72B-Instruct': 'Alibaba',
            #                               '[Alibaba] RYS-XLarge': 'Alibaba', '[Google] Gemma-2-27b-it': 'Google',
            #                               '[Meta] Llama-3.1-70B-Instruct': 'Meta'}
            # chart_data1['modelCompany'] = [i[i.index('[') + 1: i.index(']') ] for i in chart_data1['Model']]
            # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model',
            #              category_orders={"Model": globalModelOrderForDisplay}, barmode='relative')
            fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"modelCompany": globalModelOrderForDisplay}, barmode='relative',
                         color_discrete_map=color_discrete_map)
            if questionType == 'Overall':
                fig.update_layout(title_text='Overall Results - How Well Does the LLM Perform in My Company\'s Domain?',
                                  title_x=0.1)
            else:
                fig.update_layout(title_text='Overall Results', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')
            # fig.update_layout(
            #     xaxis=dict(
            #         tickmode='array',
            #         tickvals=chart_data1.index,
            #         ticktext=chart_data1.modelCompany
            #     )
            # )

            # newnames = {'[OpenAI] GPT-4o': 'OpenAI', 'col2': 'hi'}

            # fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
            #                                       legendgroup=newnames[t.name],
            #                                       hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
            #                                       )
            #                    )

            # Plot!
            st.plotly_chart(fig, use_container_width=False)
            # st.markdown("""
            #             <style>
            #             [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            #                 gap: 0rem;
            #             }
            #             </style>
            #             """, unsafe_allow_html=True)

        def associatePerformance():
            # Associate performance
            chart_data3 = pd.DataFrame(columns=['Accuracy', 'Model'])

            accScore = []
            index = 0
            for modelName in globalModelOrderForDisplay:
                df = modelDF_internal[modelName].copy()
                df = df[df['Book'] != 'ccnp']
                acc = sum(df['EM']) / len(df['EM'])
                accScore.append(acc)

                chart_data3.loc[index] = acc, modelName
                index += 1

            fig = px.bar(chart_data3, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"Model": globalModelOrderForDisplay}, color_discrete_map=color_discrete_map)
            fig.update_layout(title_text='Overall Results - CCNA (Associate)', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)

        def professionalPerformance():
            # Professional Performance
            chart_data4 = pd.DataFrame(columns=['Accuracy', 'Model'])

            accScore = []
            index = 0
            for modelName in globalModelOrderForDisplay:
                df = modelDF_internal[modelName].copy()
                df = df[df['Book'] == 'ccnp']
                acc = sum(df['EM']) / len(df['EM'])
                accScore.append(acc)

                chart_data4.loc[index] = acc, modelName
                index += 1

            fig = px.bar(chart_data4, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"Model": globalModelOrderForDisplay}, color_discrete_map=color_discrete_map)
            fig.update_layout(title_text='Overall Results - CCNP (Professional)', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)

        # with col1:
        if questionType == 'Overall':
            # col1 = st.columns(1)
            # with col1:
            mainOverallPerformance(chart_data1)

            col3, col4 = st.columns(2)
            # with col3:

            with col3:
                associatePerformance()
            with col4:
                professionalPerformance()



        else:
            col1, col2 = st.columns(2)
            with col1:

                # else:
                # combinedDF_copy = combinedDF_internal.copy()
                # if questionType == 'Recollection':
                # examplePrint = combinedDF_copy[[ 'FullQuestion', 'Answer']]
                #
                # st.markdown("""
                #     <style>
                #         .stTable tr {
                #             height: 50px; # use this to adjust the height
                #         }
                #     </style>
                # """, unsafe_allow_html=True)
                # st.table(examplePrint)
                # st.markdown('<span style="word-wrap:break-word;">' + examplePrint.item()  + '</span>', unsafe_allow_html=True)
                #
                col_question1, col_button = st.columns(spec=[0.75, 0.25])

                #
                with col_question1:

                    st.markdown('<p class="big-font" ><b>Example Questions:</b></span>', unsafe_allow_html=True)
                # answerButton = st.toggle("Show Answers", key=questionType + 'Button1')

                with col_button:
                    #     answerButton = st.toggle("Show Answers", key=questionType + 'Button1')
                    # st.markdown("""
                    #       <style>
                    #       [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
                    #           gap: 0rem;
                    #       }
                    #       </style>
                    #       """, unsafe_allow_html=True)
                    # with st.container(height=30, border=False):
                    # answerButton = st.button("Show Answers", type="primary", key = questionType + 'Button1')
                    # answerButton = st.checkbox("Show Answers", key=questionType + 'Button1')
                    answerButton = st.checkbox("Show Answers", key=questionType + 'Button1', value=True)
                    # always true by default
                    # answerButton = True

                # answerButton = st.toggle("Show Answers", key=questionType + 'Button1')
                with st.container(height=420, border=False):

                    # ================================extract example questions that are all wrong==================================
                    combinedDF_copy10 = combinedDF.copy()
                    # a = combinedDF_copy10.groupby(['Book', 'ChapterID', 'QID']).size().reset_index(name='Counts')
                    # a['Counts']

                    # a = combinedDF_copy10.groupby(['Book', 'ChapterID', 'QID', 'EM']).size().reset_index(name='Counts')

                    # compute total sample counts for that specific question
                    # total should per unique combi key should be 10 counts (since now 10 models, can be EM 0 one count, EM 1, nine counts.
                    a = combinedDF_copy10.groupby(
                        ['Book', 'ChapterID', 'QID', 'QuestionType', 'EM']).size().reset_index(
                        name='Counts')

                    #
                    # all models got it wrong
                    # because, for EM=0 (wrong) and that there are 10 models or the current number of models
                    # means all got it wrong.
                    b = a[(a['Counts'] == totalNumberOfModelsConsidered) & (a['EM'] == 0)]

                    c = b.groupby(['QuestionType']).size().reset_index(name='Counts')
                    # print(c)

                    # ensure each section have at least 1 example to show
                    for i in c['Counts'].tolist():
                        assert i >= 1, 'ensure each section have at least 1 example to show'
                    # [assert i>=1 for i in c['Counts'].tolist() ]
                    # assert len(c) == 4  # must have wrong questions for each section, currently 4 sections.

                    # out of current total models, e.g. 10, how many are wrong?
                    exampleQuestionDF = pd.DataFrame(
                        columns=['FullQuestion', 'modelOutputsCleaned', 'QuestionType', 'ModelWrongCount'])
                    exampleQuestionDF_index = 0

                    lessCorrectSamplesThreshold = 5  # consider samples with e.g. 7 wrong out of total model counts e.g. 10

                    temptList = []  # checking
                    for index, row in a.iterrows():
                        if (row['EM'] == 0 and row['Counts'] == totalNumberOfModelsConsidered) \
                                or (row['EM'] == 0 and row[
                            'Counts'] == lessCorrectSamplesThreshold):  # consider 3 wrong samples
                            # if row['EM'] == 0 and row['Counts'] == totalNumberOfModelsConsidered:
                            # print(row)
                            filtered = combinedDF_copy10[
                                (combinedDF_copy10['Book'] == row['Book']) & (
                                        combinedDF_copy10['ChapterID'] == row['ChapterID']) & (
                                        combinedDF_copy10['QID'] == row['QID'])]

                            # Skip certain samples where some model predicted extremely long answers compared to the rest
                            # not nice for display but good to keep on original record
                            allPredAnswers = filtered['predCleaned_answer2'].tolist()
                            averageAnswerLength = np.average([len(i) for i in allPredAnswers], axis=0)
                            toSkipThisSample = 0
                            for eachPredAnswer in allPredAnswers:
                                buffer = 3  # add to average length to determine strong outliers.
                                if len(eachPredAnswer) > (averageAnswerLength + buffer):
                                    toSkipThisSample = 1
                            if toSkipThisSample == 1:
                                continue

                            # as we take only 1 of the many questions by diff models, we double check that e.g. 9 models should have 1 same full question
                            assert len(set(filtered['FullQuestion'].tolist())) == 1
                            question = filtered['FullQuestion'].tolist()[0]

                            assert len(set(filtered['QuestionType'].tolist())) == 1
                            collapsedQuestionType = filtered['QuestionType'].tolist()[0]

                            assert len(set(filtered['Answer'].tolist())) == 1
                            answer = filtered['Answer'].tolist()[0].strip()
                            # modelPredText
                            # combine model answers into a single big text
                            if answerButton:
                                # PREDICTION SIDE side Green and Red
                                modelOutputsCleaned = []
                                for eachModel, eachPred in zip(filtered['ModelName'], filtered['predCleaned_answer2']):
                                    modelOutputsCleaned_row = '<span class="modelPredText">' + eachModel + ': </span>'
                                    if eachPred == answer:
                                        modelOutputsCleaned_row += '<span class="greenText">' + eachPred + '</span>'
                                    else:
                                        modelOutputsCleaned_row += '<span class="redText">' + eachPred + '</span>'

                                    modelOutputsCleaned.append(modelOutputsCleaned_row)
                                    #     modelOutputsCleaned = '<span class="modelPredText">' + filtered[
                                    # 'ModelName'] + ': </span>' + '<span class="redText">' + filtered[
                                    #                       'predCleaned_answer2'] + '</span>'
                                # color green question on QUESTION SIDE
                                for eachAnswerChar in answer:
                                    textToFind = eachAnswerChar + '. '

                                    lines = question.split('\n')
                                    for eachLine in lines:
                                        if textToFind in eachLine:
                                            greenText = '<span class="greenText">' + eachLine + '</span>'
                                            question = question.replace(eachLine, greenText)

                                question += '\n\n<b>Correct Answer:</b> ' + '<span class="greenText">' + answer + '</span>'
                                # if textToFind in question:
                                #     greenText = '<p class="greenText">' + textToFind + '</p>'
                                #     question = question.replace(textToFind, greenText)

                            else:
                                # no color
                                modelOutputsCleaned = '<span class="modelPredText">' + filtered[
                                    'ModelName'] + ': </span>' + filtered[
                                                          'predCleaned_answer2']
                                question += '\n\n <span style="opacity: 0.0;">RESERVE SPACE ONLY</span>'
                                # question += '\n\n<div style="display:none;"><b>Correct Answer:</b> ' + '<span class="greenText">' + answer + '</span></div>'
                                modelOutputsCleaned = modelOutputsCleaned.tolist()
                            modelOutputsCleaned = '<br>'.join(modelOutputsCleaned)
                            # modelOutputsCleaned = '<br>' + modelOutputsCleaned

                            # checking
                            # print(filtered)
                            # print(filtered['FullQuestion'][0:1].item())
                            # print(filtered['QuestionType'][0:1].item())
                            # checking
                            temptList.append(filtered['QuestionType'][0:1].item())

                            exampleQuestionDF.loc[
                                exampleQuestionDF_index] = question, modelOutputsCleaned, collapsedQuestionType, row[
                                'Counts']

                            exampleQuestionDF_index += 1

                    from collections import Counter
                    # checking
                    Counter(temptList)
                    # ================================extract example questions that are all wrong==================================

                    exampleQuestionDF_copy = exampleQuestionDF.copy()
                    exampleQuestionDF_copy = exampleQuestionDF_copy[
                        exampleQuestionDF_copy['QuestionType'] == questionType]
                    exampleQuestionDF_copy.drop(columns=['QuestionType'], inplace=True)

                    # sort by model wrong count to show less wrong samples at the top
                    exampleQuestionDF_copy.sort_values(by=['ModelWrongCount'], inplace=True)

                    # how many of the less wrong samples to keep? (e.g 7 wrongs), print easier to understand.
                    num_lessCorrect_SamplesToShow = 1
                    assert len(exampleQuestionDF_copy[exampleQuestionDF_copy[
                                                          'ModelWrongCount'] == lessCorrectSamplesThreshold]) >= num_lessCorrect_SamplesToShow \
                        , 'Not enough samples to show for less correct, set lower or change wrong threshold for more samples.'
                    # combine as we only need a few of the less correct samples
                    exampleQuestionDF_copy = pd.concat([exampleQuestionDF_copy[exampleQuestionDF_copy[
                                                                                   'ModelWrongCount'] == lessCorrectSamplesThreshold][
                                                        0:num_lessCorrect_SamplesToShow],
                                                        exampleQuestionDF_copy[exampleQuestionDF_copy[
                                                                                   'ModelWrongCount'] == totalNumberOfModelsConsidered]])

                    exampleQuestionDF_copy = exampleQuestionDF_copy.reset_index(drop=True)

                    # drop as can cause errors later and dont need already
                    exampleQuestionDF_copy.drop(columns=['ModelWrongCount'], inplace=True)

                    # if answerButton
                    # st.subheader('Question Example:')st
                    # st.divider()
                    # st.datafrahme(examplePrint.style.set_properties(subset=['FullQuestion'], **{'white-space': 'normal'}), use_container_width=True,hide_index=True)
                    # index = 1

                    st.markdown("""
                    <style>
                    .questionText {
                        font-size:15px !important;
                        text-align:left;
                    }
                    .modelPredText {
                        font-size:15px !important;
                        display:inline;
                        font-weight: bold;
                        text-align:left;
                    }
                    .redText {
                        font-size:15px !important;
                        color: red;
                        display:inline;
                        font-weight: bold;
                        text-align:left;
                    } 
                    .greenText {
                        font-size:15px !important;
                        color: green;
                        display:inline;
                        font-weight: bold;
                        text-align:left;
                        margin: 0px;
                        padding: 0px;
                    }  
                    .dfHeader {
                        font-size:15px !important;
                        font-weight: bold;
                        text-align:left;
                    } 
                    td {
                          vertical-align: top;
                          text-align: left;
                        }
                    </style> 
                    """, unsafe_allow_html=True)

                    for index, row in exampleQuestionDF_copy.iterrows():
                        exampleQuestionDF_copy.loc[index] = [
                            # '<div style="width:450px", align="justify"><h6 style="font-size:1vw, font-weight:normal">' + row['FullQuestion'] + '</h6></div>',
                            '<div align="justify"><span class="questionText">' + row['FullQuestion'] + '</span></div>',
                            '<div align="justify"><span class="questionText">' + row[
                                'modelOutputsCleaned'] + '</span></div>']

                    # exampleQuestionDF_copy.columns = columns=['<p style="text-align:left"><b>Question</b></p>',
                    #            '<p style="text-align:left"><b>Predictions</b></p>']
                    exampleQuestionDF_copy.columns = columns = [
                        '<div style="width:340px" , align="left"> <span class="dfHeader">Question</span></div>',
                        '<div style="width:260px", align="left"><span class="dfHeader">Predictions</span></div>']
                    examplePrint = exampleQuestionDF_copy.replace(r'\n', '<br>', regex=True)
                    examplePrint.reset_index(inplace=True, drop=True)
                    examplePrint.index = [i + 1 for i in range(len(examplePrint.index))]

                    # for i in range(len(textbookPassage)):
                    #     df.loc[index] = [
                    #         '<div style="width:600px", align="justify"><i>' + textbookPassage[i] + '</i></div>',
                    #         '<div style="width:500px", align="justify">' + QG[i] + '</div>']
                    #     index += 1

                    # html_df.index()
                    html_df = examplePrint.to_html(escape=False, index=False)
                    # reduce index size
                    # html_df = html_df.replace('<tr style="text-align: right;">\n      <th></th>\n', '<tr style="text-align: right;">\n      <th><div style="width:1px" , align="left"></div></th>\n')

                    st.markdown(html_df, unsafe_allow_html=True)
                    # st.dataframe(examplePrint)

                    # st.write(examplePrint.item().replace('\n', '  \n'))
                associatePerformance()

            with col2:
                mainOverallPerformance(chart_data1)
                professionalPerformance()

        # Domain Performance===

        option1 = st.selectbox(
            "Select Difficulty Level:",
            ['All', 'Cisco Certified Network Associate (CCNA)', 'Cisco Certified Network Professional (CCNP)'],
            key=questionType + '1'
        )
        combinedDF_copy2 = combinedDF_internal.copy()
        if option1 == 'Cisco Certified Network Associate (CCNA)':
            combinedDF_copy2 = combinedDF_copy2[combinedDF_copy2['Book'] != 'ccnp']
        elif option1 == 'Cisco Certified Network Professional (CCNP)':
            combinedDF_copy2 = combinedDF_copy2[combinedDF_copy2['Book'] == 'ccnp']
        else:
            pass

        combinedDF_copy3 = combinedDF_copy2.copy()
        associateDomains = combinedDF_copy3[combinedDF_copy3['Book'] != 'ccnp']['Domain'].unique().tolist()
        associateDomains = [i + ' - CCNA (Associate)' for i in associateDomains]

        professionalDomains = combinedDF_copy3[combinedDF_copy3['Book'] == 'ccnp']['Domain'].unique().tolist()
        professionalDomains = [i + ' - CCNP (Professional)' for i in professionalDomains]

        # domainList = combinedDF_copy2['Domain'].unique().tolist()
        domainList = associateDomains + professionalDomains

        option2 = st.selectbox(
            "Select Networking Domain:",
            ['All'] + domainList, key=questionType + '2'
        )

        option2 = option2.split(' - ')[0].strip()

        globalModelOrderForDisplay_FILTER_DOMAIN = ['[OpenAI] o1-preview', '[OpenAI] o1-mini',
                                                    '[Alibaba] Calme-2.4-rys-78b', '[Alibaba] Qwen2-72B-Instruct',
                                                     '[Google] Gemma-2-27b-it',
                                                    '[Meta] Llama-3.1-70B-Instruct', '[Nvidia] Llama-3.1-Nemotron-70B']

        # remove some models as seem like too many
        combinedDF_copy2 = combinedDF_copy2[
            combinedDF_copy2['ModelName'].isin(globalModelOrderForDisplay_FILTER_DOMAIN)]

        if option2 == 'All':

            a = combinedDF_copy2.groupby(['Domain', 'ModelName'])['EM'].sum().reset_index()
            b = combinedDF_copy2.groupby(['Domain', 'ModelName'])['EM'].count().reset_index()
            result = pd.merge(a, b, how="inner", on=["Domain", "ModelName"])
            result['Acc'] = (result['EM_x'] / result['EM_y'])

            result.sort_values(by=["ModelName", "Acc"], ascending=[False, False], inplace=True)
            # result.sort_values(by=["Acc"], ascending=[False], inplace=True)
            # df = px.data.tips()px
            # fig = px.bar(result, x="Domain", y="Acc", orientation='v', color='ModelName', text_auto=False)
            fig = px.histogram(result, x="Domain", y="Acc",
                               color='ModelName', barmode='group',
                               category_orders={"ModelName": globalModelOrderForDisplay},
                               color_discrete_map=color_discrete_map)
            # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
            fig.update_layout(title_text='Results by Technical Domain', title_x=0.05)
            fig.update_layout(xaxis={'categoryorder': 'total descending'})  # add only this line to sort
            fig.update_layout(yaxis_title="Acc")  # add only this line to sort
            fig.update_xaxes(tickangle=20)
            # fig.update_xaxes(ticklabelposition='outside left')

            fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace("sum of", "")))
            fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace("sum of", "")))

            fig.update_layout(yaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)
        else:
            # filter by domain
            combinedDF_copy2 = combinedDF_copy2[combinedDF_copy2['Domain'] == option2]
            a = combinedDF_copy2.groupby(['ChapterTitle', 'ModelName'])['EM'].sum().reset_index()
            b = combinedDF_copy2.groupby(['ChapterTitle', 'ModelName'])['EM'].count().reset_index()
            result = pd.merge(a, b, how="inner", on=["ChapterTitle", "ModelName"])

            result['Acc'] = (result['EM_x'] / result['EM_y'])

            result.sort_values(by=["ModelName", "Acc"], ascending=[False, False], inplace=True)
            # result.sort_values(by=["Acc"], ascending=[False], inplace=True)
            # df = px.data.tips()px
            # fig = px.bar(result, x="Domain", y="Acc", orientation='v', color='ModelName', text_auto=False)
            fig = px.histogram(result, x="ChapterTitle", y="Acc",
                               color='ModelName', barmode='group',
                               category_orders={"ModelName": globalModelOrderForDisplay},
                               color_discrete_map=color_discrete_map)
            # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
            fig.update_layout(title_text='Results by Sub-area', title_x=0.05)
            fig.update_layout(xaxis={'categoryorder': 'total descending'})  # add only this line to sort
            fig.update_layout(yaxis_title="Acc")  # add only this line to sort
            fig.update_xaxes(tickangle=20)
            # fig.update_xaxes(ticklabelposition='outside left')

            fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace("sum of", "")))
            fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace("sum of", "")))

            fig.update_layout(yaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)
        # agree = st.checkbox("Show Detailed Topics")
        #
        # if agree:
        #     st.write("Great!")
        # else:

        # _, col2_header, _ = st.columns([1, 2, 3])
        # with col2_header:
        #     st.subheader ("Overall:", divider=False)

        st.markdown("""
            <style>
            [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
                gap: 0rem;
            }
            </style>
            """, unsafe_allow_html=True)


# tabReady = 0
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
# with st.sidebar:

#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#     # st.selectbox('Choose Model', ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-4o-mini' 'GPT-3.5-Turbo'])
#
#     buttonPressed = st.button('Run Cisco LLM Benchmarking')


# if tabReady==1:
# Insert containers separated into tabs:
# tab1, tab2, tab3 = st.tabs(["Inference", "Results", "Chat"])
# tab1, tab2, tab3, tab4 , tab5 , tab6   = st.tabs(["Inference", "Overall Results", "Recollection", "Conceptual Knowledge", "Configuration Ability", "Problem Solving"])
def questionTypePerformance():
    # Question Type Performance===
    # if questionType == 'Overall':
    # _, col2_header, _ = st.columns([1, 2, 3])
    # with col2_header:
    #     st.subheader ("Overall:", divider=False)
    combinedDF_copy = combinedDF.copy()
    a = combinedDF_copy.groupby(['QuestionType', 'ModelName'])['EM'].sum().reset_index()
    b = combinedDF_copy.groupby(['QuestionType', 'ModelName'])['EM'].count().reset_index()
    result = pd.merge(a, b, how="inner", on=["QuestionType", "ModelName"])
    result['Acc'] = (result['EM_x'] / result['EM_y'])

    result.rename(columns={"QuestionType": "Capability Type"}, inplace=True)
    fig = px.histogram(result, x="Capability Type", y="Acc",
                       color='ModelName', height=436, barmode='group', category_orders={
            "ModelName": globalModelOrderForDisplay}, color_discrete_map=color_discrete_map)
    # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
    # fig.update_layout(title_text='Question Type Performance', title_x=0.05)
    # fig.update_layout(xaxis={'categoryorder': })  # add only this line to sort
    fig.update_xaxes(categoryorder='array',
                     categoryarray=["Recollection", "Conceptual Knowledge", "Configuration Ability",
                                    "Problem Solving"])
    fig.update_layout(yaxis_title="Acc")  # add only this line to sort

    fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace("sum of", "")))
    fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace("sum of", "")))

    # fig.update_traces(texttemplate='%{y:.1f}')
    # fig.update_layout(xaxis_tickformat='%')
    fig.update_layout(yaxis_tickformat='.1%')

    # Plot!
    st.plotly_chart(fig, use_container_width=False)


main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ['LLM Benchmarking', 'Capability Evaluation', 'Our Research', 'Summary'])

# tab1, tab2, tab3, tab4 , tab5 , tab6, tab7   = st.tabs([ "Introduction", "Overall Results (Zero-Shot)", "Recollection", "Conceptual Knowledge", "Configuration Ability", "Problem Solving", "[Research] Automated Question Generation"])
with main_tab1:
    tab1, tab2 = st.tabs(["Introduction", "Overall Results (Zero-Shot)"])

    # createTab(tab1, 'Introduction', combinedDF, modelDF)
    with tab1:
        # st.markdown('<h5>LLM Baselines</h5>', unsafe_allow_html=True)
        # st.markdown('<p class="big-font">Knowledge Source: <u>Community Forum</u></p>',unsafe_allow_html=True)
        with open("main1_pic.svg") as logo_file:
            logo = logo_file.read()
        st.image(logo, width=800)
        st.markdown('---')
        st.markdown('<h5>Technical Support Domain:</h5>', unsafe_allow_html=True)
        # with open("ciscoLogo.svg") as logo_file:
        #     logo = logo_file.read()
        # st.image(logo, width=100)
        with open("main2_pic.svg") as logo_file:
            logo = logo_file.read()
        st.image(logo, width=900)
        st.markdown('---')
    with tab2:
        createTab(tab2, 'Overall', combinedDF, modelDF)
with main_tab2:
    tab22, tab3, tab4, tab5, tab6 = st.tabs(
        ["Introduction", "Recollection", "Conceptual Knowledge", "Configuration Ability", "Problem Solving"])
    with tab22:
        with open("main2_capability.svg") as logo_file:
            logo = logo_file.read()
        st.image(logo, width=1000)

        st.markdown('---')

        # with col1_title_capability:
        st.markdown('<h3>Overall Results</h3>', unsafe_allow_html=True)
        _, col1_metric_capability, col2_metric_capability, col3_metric_capability, col4_metric_capability, _ = st.columns(
            spec=[0.08, 0.2, 0.2, 0.2, 0.2, 0.1])
        # with col1_title_capability:
        #     st.markdown('<h3>Overall Results</h3>', unsafe_allow_html=True)

        # find the best model acc for each category
        tempt_a = combinedDF.groupby(['QuestionType', 'ModelName'])['EM'].sum().reset_index()
        tempt_b = combinedDF.groupby(['QuestionType', 'ModelName'])['EM'].count().reset_index()
        tempt_a['total'] = tempt_b['EM']
        tempt_a['acc'] = tempt_a['EM'] / tempt_a['total']

        tempt_a = tempt_a.groupby(['QuestionType']).acc.max().reset_index()
        # temptQuestionTypeDF = combinedDF[combinedDF['QuestionType']=='Recollection']
        # recollection_acc = sum(['EM'])/
        col1_metric_capability.metric("Best Result (Recollection)", "  " + str(
            normal_round(tempt_a[tempt_a['QuestionType'] == 'Recollection']['acc'].item() * 100, 1)) + '%')
        col2_metric_capability.metric("Best Result (Conceptual Knowledge)", "  " + str(
            normal_round(tempt_a[tempt_a['QuestionType'] == 'Conceptual Knowledge']['acc'].item() * 100, 1)) + '%')
        col3_metric_capability.metric("Best Result (Configuration Ability)", "  " + str(
            normal_round(tempt_a[tempt_a['QuestionType'] == 'Configuration Ability']['acc'].item() * 100, 1)) + '%')
        col4_metric_capability.metric("Best Result (Problem Solving)", "  " + str(
            normal_round(tempt_a[tempt_a['QuestionType'] == 'Problem Solving']['acc'].item() * 100, 1)) + '%')
        # col1_metric_capability.metric("Recollection Best Result", '80.7%')
        # col2_metric_capability.metric("Total Domains", "   " + str(len(modelDF_internal['[OpenAI] GPT-4o']['Domain'].unique())))
        # col3_metric_capability.metric("Total Topics",
        #                    "   " + str(len(modelDF_internal['[OpenAI] GPT-4o']['ChapterTitle'].unique())))
        # col4_metric_capability.metric("Total Questions", "  " + str(len(modelDF_internal['[OpenAI] GPT-4o'])))
        questionTypePerformance()
        # st.markdown('<h5>Generating Multiple-choice Questions (MCQ) from Organizational Knowledge Sources</h5>', unsafe_allow_html=True)

    createTab(tab3, 'Recollection', combinedDF, modelDF)
    createTab(tab4, 'Conceptual Knowledge', combinedDF, modelDF)
    createTab(tab5, 'Configuration Ability', combinedDF, modelDF)
    createTab(tab6, 'Problem Solving', combinedDF, modelDF)

with main_tab3:
    tab7, tab8 = st.tabs(['Introduction', 'Automated Question Generation'])
    # tab7, tab8 = st.tabs(['Introduction', 'Automated Question Generation'])
    with tab7:
        st.markdown('<h5>Motivation</h5>', unsafe_allow_html=True)
        with open("researchPic.svg") as logo_file:
            logo = logo_file.read()
        st.image(logo, width=800)
        st.markdown('&nbsp;',
                    unsafe_allow_html=True)

    with tab8:
        st.markdown('<h5>Generating Multiple-choice Questions (MCQ) from Organizational Knowledge Sources</h5>',
                    unsafe_allow_html=True)
        # st.markdown('&nbsp;', unsafe_allow_html=True)
        # st.markdown('<p class="big-font"><b>Generating Questions from Knowledge Sources</b></p>', unsafe_allow_html=True)
        st.markdown('<p class="big-font">Knowledge Source: <u>Community Forum</u></p>',
                    unsafe_allow_html=True)

        with open("QG_forum.svg") as logo_file:
            logo = logo_file.read()
        st.image(logo, width=1200)

        # st.write(combinedDF[0:2])
        textbookPassage = [
            '\"The interface ID (IID) value to follow the just-learned IPv6 prefix. Before using the address, use DAD to ensure that no other host is already using the same address. Figure 28-8 depicts the first two steps while noting the two most common ways a host completes the address. Hosts can use modified EUI-64 rules, as discussed in the section, “Generating a Unique Interface ID Using Modified EUI-64,” in Chapter 27, “Implementing IPv6 Addressing on Routers,” or a random number. Figure 28-8 Host IPv6 Address Formation Using SLAAC Combining SLAAC with Stateless DHCP When using SLAAC, a host uses three tools\"',
            '\"Chapter 14 1. D. When using classful IP addressing concepts as described in Chapter 13, “Analyzing Subnet Masks,” addresses have three parts: network, subnet, and host. For addresses in a single classful network, the network parts must be identical for the numbers to be in the same network. For addresses in the same subnet, both the network and subnet parts must have identical values. The host part differs when comparing different addresses in the same subnet. 2. B and D. In any subnet, the subnet ID is the smallest number in the range, the subnet broadcast address is the largest\"']

        QG = [
            'When using SLAAC, a host uses: <br>a) SLAAC to learn the prefix length. <br>b) NDP messages to learn the prefix length. <br>c) stateless DHCP to learn the IPv6 addresses of any DNS servers. <br>d) all of the above',
            'To find the subnet broadcast address, you need to find the:<br>a) subnet ID. <br>b) mask. <br>c) prefix. <br>d) host.']
        # df = pd.DataFrame(columns=['<div style="width:130px">Textbook Passage</div>', '<div style="width:150px">Question Generated (with choices)</div>'], index=[1,2])
        # df = pd.DataFrame(columns=['<H6 align="left">Textbook Passage</H6>',
        #                            '<H6 align="left">Question Generated (with choices)</H6>'], index=[1, 2])
        df = pd.DataFrame(columns=['<div align="left"> <span class="dfHeader">Passage</span></div>',
                                   '<div  align="left"> <span class="dfHeader">Multiple-choice Question Generated</span></div>'],
                          index=[1, 2])

        index = 1
        for i in range(len(textbookPassage)):
            df.loc[index] = ['<div style="width:600px", align="justify"><i>' + textbookPassage[i] + '</i></div>',
                             '<div style="width:500px", align="justify">' + QG[i] + '</div>']
            index += 1
        st.markdown('&nbsp;',
                    unsafe_allow_html=True)
        st.markdown('&nbsp;',
                    unsafe_allow_html=True)
with main_tab4:
    st.markdown('<h5>Summary</h5>', unsafe_allow_html=True)
    with open("main2_summary.svg") as logo_file:
        logo = logo_file.read()
    st.image(logo, width=1200)
    st.markdown('&nbsp;',
                unsafe_allow_html=True)

    # st.table(df)

    # st.markdown('<p class="big-font">Knowledge Source 2: <u>Document Passages</u></p>',
    #             unsafe_allow_html=True)
    # st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
# st.markdown('&nbsp;',
#             unsafe_allow_html=True)
# st.markdown('&nbsp;',
#             unsafe_allow_html=True)



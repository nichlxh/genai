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

st.set_page_config(page_title="LLM Benchmarking", layout="wide", page_icon='ciscoLogo.svg')
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

st.title("LLM Benchmarking")
logCol1, logCol2, logCol3, logCol4, logCol5, _, _ = st.columns(7)
with logCol1:
    # st.image('mainLogo.png', width=500)
    with open("mainLogo.svg") as logo_file:
        logo = logo_file.read()
    st.image(logo, width=550)
# with logCol2:
#     st.image('ciscoLogo.png', width=330)
# with logCol3:


st.caption("🚀 Evaluating Large Language Models on Question Answering for Networking")

# ordered is followed for website printing + also used for others to process all the models
globalModelOrder = ['gpt-4o', 'calme-2.4-rys-78b', 'gpt-4-turbo', 'gpt-4', 'gpt-4o-mini', 'gpt-3.5-turbo',
                    'gemma-2-27b-it', 'Meta-Llama-3.1-70B-Instruct', 'Qwen2-72B-Instruct', 'RYS-XLarge']
# from openai import OpenAI
import streamlit as st

# read in all model pred DF
combinedDF = pd.DataFrame()

modelDF = {}
for modelName in globalModelOrder:
    # for modelName in ['gpt-4o','gpt-4o-mini','gpt-4-turbo','gpt-4','gpt-3.5-turbo']:
    modelDF[modelName] = pd.read_excel('outputs/ciscoBooks_QA_combined_pred_' + modelName + '_Cleaned2.xlsx')
    # modelDF[modelName] = modelDF[modelName].replace('‚Äôs','\'s', regex=True)
    modelDF[modelName]['ModelName'] = modelName

    combinedDF = pd.concat([combinedDF, modelDF[modelName]], ignore_index=True)
    # combinedDF = combinedDF.append(modelDF[modelName], ignore_index=True)
    # print(len(combinedDF))

combinedDF['QuestionType'] = combinedDF['predictedQuestionType']
combinedDF.drop(['predictedQuestionType'], axis=1, inplace=True)


#

def createTab(tabChoice, questionType, combinedDF, modelDF):
    # filter by question type section
    combinedDF_internal = combinedDF.copy()
    modelDF_internal = copy.deepcopy(modelDF)
    if questionType != 'Overall':  # filter except first tab
        combinedDF_internal = combinedDF_internal[combinedDF_internal['QuestionType'] == questionType]
        for modelName in globalModelOrder:
            modelDF_internal[modelName]['QuestionType'] = modelDF_internal[modelName]['predictedQuestionType']
            modelDF_internal[modelName].drop(['predictedQuestionType'], axis=1, inplace=True)

            modelDF_internal[modelName] = modelDF_internal[modelName][
                modelDF_internal[modelName]['QuestionType'] == questionType]

    with tabChoice:
        chart_data1 = pd.DataFrame(columns=['Accuracy', 'Model'])

        accScore = []
        index = 0
        for modelName in globalModelOrder:
            acc = sum(modelDF_internal[modelName]['EM']) / len(modelDF_internal[modelName]['EM'])
            # acc = round(acc*100,1)
            accScore.append(acc)

            chart_data1.loc[index] = acc, modelName
            index += 1

        # import decimal
        # # import decimal
        # decimal.getcontext().rounding = decimal.ROUND_CEILING

        import math  # round up when >0.5 else round down
        def normal_round(n, decimals=0):
            expoN = n * 10 ** decimals
            if abs(expoN) - abs(math.floor(expoN)) < 0.5:
                return math.floor(expoN) / 10 ** decimals
            return math.ceil(expoN) / 10 ** decimals

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

        _, col1_metric, col2_metric, col3_metric, col4_metric, _ = st.columns(6)
        col1_metric.metric("Best Model Performance", "  " + str(normal_round(max(accScore) * 100, 1)) + '%')
        col2_metric.metric("Total Domains", "   " + str(len(modelDF_internal['gpt-4o']['Domain'].unique())))
        col3_metric.metric("Total Topics", "   " + str(len(modelDF_internal['gpt-4o']['ChapterTitle'].unique())))
        col4_metric.metric("Total Questions", "  " + str(len(modelDF_internal['gpt-4o'])))

        # with col1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"Model": globalModelOrder})
            fig.update_layout(title_text='Overall Performance', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)
            # st.markdown("""
            #             <style>
            #             [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            #                 gap: 0rem;
            #             }
            #             </style>
            #             """, unsafe_allow_html=True)

            # Associate performance
            chart_data3 = pd.DataFrame(columns=['Accuracy', 'Model'])

            accScore = []
            index = 0
            for modelName in globalModelOrder:
                df = modelDF_internal[modelName].copy()
                df = df[df['Book'] != 'ccnp']
                acc = sum(df['EM']) / len(df['EM'])
                accScore.append(acc)

                chart_data3.loc[index] = acc, modelName
                index += 1

            fig = px.bar(chart_data3, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"Model": globalModelOrder})
            fig.update_layout(title_text='Overall Performance - Associate', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            # Question Type Performance===
            if questionType == 'Overall':

                # _, col2_header, _ = st.columns([1, 2, 3])
                # with col2_header:
                #     st.subheader ("Overall:", divider=False)
                combinedDF_copy = combinedDF_internal.copy()
                a = combinedDF_copy.groupby(['QuestionType', 'ModelName'])['EM'].sum().reset_index()
                b = combinedDF_copy.groupby(['QuestionType', 'ModelName'])['EM'].count().reset_index()
                result = pd.merge(a, b, how="inner", on=["QuestionType", "ModelName"])
                result['Acc'] = (result['EM_x'] / result['EM_y'])

                fig = px.histogram(result, x="QuestionType", y="Acc",
                                   color='ModelName', height=436, barmode='group', category_orders={
                        "ModelName": globalModelOrder})
                # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
                fig.update_layout(title_text='Question Type Performance', title_x=0.05)
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
            else:
                # combinedDF_copy = combinedDF_internal.copy()
                # if questionType == 'Recollection':
                # examplePrint = combinedDF_copy[[ 'FullQuestion', 'Answer']]

                # st.markdown("""
                #     <style>
                #         .stTable tr {
                #             height: 50px; # use this to adjust the height
                #         }
                #     </style>
                # """, unsafe_allow_html=True)
                # st.table(examplePrint)
                # st.markdown('<span style="word-wrap:break-word;">' + examplePrint.item()  + '</span>', unsafe_allow_html=True)
                col_question1, col_button = st.columns(spec=[0.7, 0.3])

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
                    answerButton = st.checkbox("Show Answers", key=questionType + 'Button1')

                # answerButton = st.toggle("Show Answers", key=questionType + 'Button1')
                with st.container(height=380, border=False):

                    # ================================extract example questions that are all wrong==================================
                    combinedDF_copy10 = combinedDF.copy()
                    # a = combinedDF_copy10.groupby(['Book', 'ChapterID', 'QID']).size().reset_index(name='Counts')
                    # a['Counts']

                    # a = combinedDF_copy10.groupby(['Book', 'ChapterID', 'QID', 'EM']).size().reset_index(name='Counts')

                    a = combinedDF_copy10.groupby(
                        ['Book', 'ChapterID', 'QID', 'QuestionType', 'EM']).size().reset_index(
                        name='Counts')
                    totalNumberOfModelsConsiderd = len(globalModelOrder)
                    #
                    b = a[(a['Counts'] == totalNumberOfModelsConsiderd) & (a['EM'] == 0)]

                    c = b.groupby(['QuestionType']).size().reset_index(name='Counts')
                    # print(c)

                    assert len(c) == 4  # must have wrong questions for each section, currently 4.

                    exampleQuestionDF = pd.DataFrame(columns=['FullQuestion', 'modelOutputsCleaned', 'QuestionType'])
                    exampleQuestionDF_index = 0

                    temptList = []  # checking
                    for index, row in a.iterrows():
                        if row['EM'] == 0 and row['Counts'] == totalNumberOfModelsConsiderd:
                            # print(row)
                            filtered = combinedDF_copy10[
                                (combinedDF_copy10['Book'] == row['Book']) & (
                                            combinedDF_copy10['ChapterID'] == row['ChapterID']) & (
                                        combinedDF_copy10['QID'] == row['QID'])]

                            # as we take only 1 of the many questions by diff models, we double check that e.g. 9 models should have 1 same full question
                            assert len(set(filtered['FullQuestion'].tolist())) == 1
                            question = filtered['FullQuestion'].tolist()[0]

                            assert len(set(filtered['QuestionType'].tolist())) == 1
                            collapsedQuestionType = filtered['QuestionType'].tolist()[0]

                            assert len(set(filtered['Answer'].tolist())) == 1
                            answer = filtered['Answer'].tolist()[0]
                            # modelPredText
                            # combine model answers into a single big text
                            if answerButton:
                                # color wrong
                                modelOutputsCleaned = '<span class="modelPredText">' + filtered[
                                    'ModelName'] + ': </span>' + '<span class="redText">' + filtered[
                                                          'predCleaned_answer2'] + '</span>'
                                # color green question
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
                                exampleQuestionDF_index] = question, modelOutputsCleaned, collapsedQuestionType

                            exampleQuestionDF_index += 1

                    from collections import Counter
                    # checking
                    Counter(temptList)
                    # ================================extract example questions that are all wrong==================================

                    exampleQuestionDF_copy = exampleQuestionDF.copy()
                    exampleQuestionDF_copy = exampleQuestionDF_copy[
                        exampleQuestionDF_copy['QuestionType'] == questionType]
                    exampleQuestionDF_copy = exampleQuestionDF_copy.reset_index(drop=True)
                    exampleQuestionDF_copy.drop(columns=['QuestionType'], inplace=True)
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
                        # index+=1 # as it starts from 1
                        exampleQuestionDF_copy.loc[index] = [
                            # '<div style="width:450px", align="justify"><h6 style="font-size:1vw, font-weight:normal">' + row['FullQuestion'] + '</h6></div>',
                            '<div align="justify"><span class="questionText">' + row['FullQuestion'] + '</span></div>',
                            '<div align="justify"><span class="questionText">' + row[
                                'modelOutputsCleaned'] + '</span></div>']

                    # exampleQuestionDF_copy.columns = columns=['<p style="text-align:left"><b>Question</b></p>',
                    #            '<p style="text-align:left"><b>Predictions</b></p>']
                    exampleQuestionDF_copy.columns = columns = [
                        '<div style="width:340px" , align="left"> <span class="dfHeader">Question</span></div>',
                        '<div style="width:250px", align="left"><span class="dfHeader">Predictions</span></div>']
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

            # Professional Performance
            chart_data4 = pd.DataFrame(columns=['Accuracy', 'Model'])

            accScore = []
            index = 0
            for modelName in globalModelOrder:
                df = modelDF_internal[modelName].copy()
                df = df[df['Book'] == 'ccnp']
                acc = sum(df['EM']) / len(df['EM'])
                accScore.append(acc)

                chart_data4.loc[index] = acc, modelName
                index += 1

            fig = px.bar(chart_data4, x="Accuracy", y="Model", orientation='h', color='Model',
                         category_orders={"Model": globalModelOrder})
            fig.update_layout(title_text='Overall Performance - Professional', title_x=0.1)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # add only this line to sort
            # fig.update_traces(texttemplate='%{y:.1f}')
            # fig.update_layout(xaxis_tickformat='%')
            fig.update_layout(xaxis_tickformat='.1%')

            # Plot!
            st.plotly_chart(fig, use_container_width=False)

        # Domain Performance===

        option1 = st.selectbox(
            "Select Difficulty Level:",
            ['All', 'Associate', 'Professional'], key=questionType + '1'
        )
        combinedDF_copy2 = combinedDF_internal.copy()
        if option1 == 'Associate':
            combinedDF_copy2 = combinedDF_copy2[combinedDF_copy2['Book'] != 'ccnp']
        elif option1 == 'Professional':
            combinedDF_copy2 = combinedDF_copy2[combinedDF_copy2['Book'] == 'ccnp']
        else:
            pass

        combinedDF_copy3 = combinedDF_copy2.copy()
        associateDomains = combinedDF_copy3[combinedDF_copy3['Book'] != 'ccnp']['Domain'].unique().tolist()
        associateDomains = [i + ' - Associate' for i in associateDomains]

        professionalDomains = combinedDF_copy3[combinedDF_copy3['Book'] == 'ccnp']['Domain'].unique().tolist()
        professionalDomains = [i + ' - Professional' for i in professionalDomains]

        # domainList = combinedDF_copy2['Domain'].unique().tolist()
        domainList = associateDomains + professionalDomains

        option2 = st.selectbox(
            "Select Networking Domain:",
            ['All'] + domainList, key=questionType + '2'
        )

        option2 = option2.split(' - ')[0].strip()

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
                               color='ModelName', barmode='group', category_orders={"ModelName": globalModelOrder})
            # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
            fig.update_layout(title_text='Domain Performance', title_x=0.05)
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
                               color='ModelName', barmode='group', category_orders={"ModelName": globalModelOrder})
            # fig = px.bar(chart_data1, x="Accuracy", y="Model", orientation='h', color='Model')
            fig.update_layout(title_text='Domain Topic Performance', title_x=0.05)
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

tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overall Results (Zero-Shot)", "Recollection", "Conceptual Knowledge", "Configuration Ability", "Problem Solving",
     "[Experimental] Automated Question Generation"])

createTab(tab2, 'Overall', combinedDF, modelDF)
createTab(tab3, 'Recollection', combinedDF, modelDF)
createTab(tab4, 'Conceptual Knowledge', combinedDF, modelDF)
createTab(tab5, 'Configuration Ability', combinedDF, modelDF)
createTab(tab6, 'Problem Solving', combinedDF, modelDF)

with tab7:
    st.markdown('<h5>Generating Multiple-choice Questions (MCQ) from Organizational Knowledge Sources</h5>',
                unsafe_allow_html=True)
    # st.markdown('&nbsp;', unsafe_allow_html=True)
    # st.markdown('<p class="big-font"><b>Generating Questions from Knowledge Sources</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">Knowledge Source 1: <u>Community Forum</u></p>',
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

    # st.table(df)

    st.markdown('<p class="big-font">Knowledge Source 2: <u>Document Passages</u></p>',
                unsafe_allow_html=True)
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
    st.markdown('&nbsp;',
                unsafe_allow_html=True)
    st.markdown('&nbsp;',
                unsafe_allow_html=True)

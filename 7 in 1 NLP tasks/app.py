import streamlit as st
import spacy_streamlit
import spacy
nlp=spacy.load('en_core_web_sm')
from PIL import Image
import docx2txt
from PyPDF2 import PdfFileReader
from textblob import TextBlob
from collections import Counter
from spacy import displacy
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import neattext as nt
import neattext.functions as nfx
import streamlit.components.v1 as stc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import altair as alt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import neattext.functions as nfx
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import requests
from clean import preprocessing
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tensorflow as tf
#from QnA import *





def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text

def text_analyzer(text):
    docx=nlp(text)
    all_data=[(token.text,token.pos_,token.lemma_,token.is_stop) for token in docx]
    df=pd.DataFrame(all_data,columns=['Token','PoS','Root Word','Is_Stopword'])
    return df

def get_most_common_tokens(text,num=5):
    tokens=Counter(text.split())
    common_token=dict(tokens.most_common(num))
    return common_token

def sentiment(text):
    blob=TextBlob(text)
    return blob.sentiment

def plot_wordcloud(my_text):
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    return entities

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
@st.cache
def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx,style="ent")
    html = html.replace("\n\n","\n")
    result = HTML_WRAPPER.format(html)
    return result

def sumy_summarizer(docx,num=2):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,num)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


punctuation = punctuation + '\n' + '\n\n'
stopwords = list(STOP_WORDS)
def freq_summarization(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    summary = [word.text for word in summary]
    summary = ' '.join(summary)
    return summary


# def embed_elmo2(module):
#     with tf.Graph().as_default():
#         sentences = tf.placeholder(tf.string)
#         embed = hub.Module(module)
#         embeddings = embed(sentences)
#         session = tf.train.MonitoredSession()
#     return lambda x: session.run(embeddings, {sentences: x})
# embed_fn = embed_elmo2("https://tfhub.dev/google/elmo/2")

# Evaluate Summary
from rouge import Rouge
def evaluate_summary(summary,reference):
    r = Rouge()
    eval_score = r.get_scores(summary,reference)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df



def main():
    st.set_page_config(page_title="20 in 1 NLP tasks",layout='wide')
    #st.title('NER recognition app')
    options=['Home','Analysis','Custom text cleaning','Question and answering','Text summarization','Email extractor','Spelling correction',
             'Text generation','About']
    choice=st.sidebar.selectbox('Chose accordingly',options)


    if choice=='Home':
        image=Image.open('1_a3xerDP7jqQglKxfIxfxVw.jpeg')
        st.image(image)
        st.header('Multi **NLP** tasks in a single window')
        st.write("""
        # This web App contains different text analysis with visual representation and advance tasks like QnA and Text Generation
        """)




    elif choice=='Analysis':
        st.subheader('Upload document')

        doc_file = st.file_uploader('', type=['csv', 'pdf', 'text', 'docx'])

        if doc_file is not None:
            file_details = doc_file.type
            if file_details == 'text/plain':
                raw_text = str(doc_file.read(), 'utf-8')
            elif file_details == 'application/pdf':
                raw_text = read_pdf(doc_file)

            else:
                raw_text = docx2txt.process(doc_file)

        elif doc_file is None:
            st.subheader('Or enter your input')
            raw_text = st.text_area(' ')

        
        if st.sidebar.checkbox('Analyze'):
            num_of_most_common=st.sidebar.number_input('Most common tokens',5,15)
            with st.beta_expander('Original text'):
                st.write(raw_text)

            with st.beta_expander('Basic Text Analysis'):
                data=text_analyzer(raw_text)
                st.dataframe(data)


            col1,col2=st.beta_columns(2)

            with col1:
                with st.beta_expander('Word Stats'):
                    st.info('Words statistics')
                    doc=nt.TextFrame(raw_text)
                    st.write(doc.word_stats())
                with st.beta_expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    st.write(keywords)

                with st.beta_expander("Sentiment"):
                    sent_result = sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.beta_expander("Plot Word Freq"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    plt.bar(keywords.keys(), top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.beta_expander('Plot of part of speech'):
                    fig=plt.figure()
                    sns.countplot(data['PoS'])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                with st.beta_expander('Word Cloud Visualization'):
                    plot_wordcloud(raw_text)

        if st.sidebar.checkbox('Name Entity Recognition'):
            doc = nlp(raw_text)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels,
                                          attrs=['text', 'label_', 'start', 'end'])




    elif choice=='Custom text cleaning':
        st.subheader('Custom text cleaning')
        doc_file = st.file_uploader('', type=['csv', 'pdf', 'text', 'docx'])

        if doc_file is not None:
            file_details = doc_file.type
            if file_details == 'text/plain':
                raw_text = str(doc_file.read(), 'utf-8')
            elif file_details == 'application/pdf':
                raw_text = read_pdf(doc_file)

            else:
                raw_text = docx2txt.process(doc_file)

        elif doc_file is None:
            st.subheader('Or enter your input')
            raw_text = st.text_area(' ')

        normalization = st.sidebar.checkbox('Text normalization')
        clean_stopwards = st.sidebar.checkbox('Remove stopwords')
        clean_punctuation = st.sidebar.checkbox('Remove punctuation')
        clean_numreric = st.sidebar.checkbox('Remove numbers')
        clean_special = st.sidebar.checkbox('Remove special characters')
        clean_url = st.sidebar.checkbox('Clean URLs')

        if st.button('Start process'):



            col1,col2=st.beta_columns(2)
            with col1:
                with st.beta_expander('Original text'):
                    st.write('The length is :',len(raw_text))
                    st.write(raw_text)

            with col2:
                with st.beta_expander('Processed text'):
                    if normalization:
                        raw_text=raw_text.lower()
                    if clean_stopwards:
                        raw_text=nfx.remove_stopwords(raw_text)
                    if clean_url:
                        raw_text=nfx.remove_urls(raw_text)
                    if clean_special:
                        raw_text=nfx.remove_special_characters(raw_text)
                    if clean_punctuation:
                        raw_text=nfx.remove_punctuations(raw_text)
                    if clean_numreric:
                        raw_text=nfx.remove_numbers(raw_text)
                    st.write('The length is :',len(raw_text))
                    st.write(raw_text)




    elif choice=='Text summarization':
        st.subheader('Extractive text summarization')
        doc_file = st.file_uploader('Upload', type=['csv', 'pdf', 'text', 'docx'])
        #

        if doc_file is not None:
            file_details = doc_file.type
            if file_details == 'text/plain':
                raw_text = str(doc_file.read(), 'utf-8')
            elif file_details == 'application/pdf':
                raw_text = read_pdf(doc_file)
            else:
                raw_text = docx2txt.process(doc_file)
        elif doc_file is None:
            raw_text = st.text_area('Or enter your input manually')

        if st.button("Summarize"):
            with st.beta_expander("Original Text"):
                st.write(raw_text)
            c1, c2 = st.beta_columns(2)

            with c1:
                with st.beta_expander("LexRank Summary"):
                    my_summary = sumy_summarizer(raw_text)
                    document_len = {"Original": len(raw_text),
                                    "Summary": len(my_summary)}
                    st.write(document_len)
                    st.write(my_summary)

                    st.info("Rouge Score")
                    eval_df = evaluate_summary(my_summary, raw_text)
                    #st.dataframe(eval_df.T)
                    eval_df['metrics'] = eval_df.index
                    c = alt.Chart(eval_df).mark_bar().encode(
                        x='metrics', y='rouge-1')
                    st.altair_chart(c)

            with c2:
                with st.beta_expander("Frequency based summary"):
                    summary=freq_summarization(raw_text)
                    document_len = {"Original": len(raw_text),
                                    "Summary": len(summary)}
                    st.write(document_len)
                    st.write(summary)
                    st.info("Rouge Score")
                    eval_df = evaluate_summary(summary, raw_text)
                    #st.dataframe(eval_df.T)
                    eval_df['metrics'] = eval_df.index
                    c = alt.Chart(eval_df).mark_bar().encode(
                        x='metrics', y='rouge-1')
                    st.altair_chart(c)




    # elif choice=='Document similarity':
    #     st.subheader('Document similarity check')

    #     doc_file_1 = st.file_uploader('Upload first document', type=['csv', 'pdf', 'text', 'docx'])
    #     if doc_file_1 is not None:
    #         file_details = doc_file_1.type
    #         if file_details == 'text/plain':
    #             raw_text_1 = str(doc_file_1.read(), 'utf-8')
    #         elif file_details == 'application/pdf':
    #             raw_text_1 = read_pdf(doc_file_1)
    #         else:
    #             raw_text_1 = docx2txt.process(doc_file_1)
    #     elif doc_file_1 is None:
    #         raw_text_1 = st.text_area('Upload first document manually')

    #     doc_file_2 = st.file_uploader('Upload second document', type=['csv', 'pdf', 'text', 'docx'])
    #     if doc_file_1 is not None:
    #         file_details = doc_file_2.type
    #         if file_details == 'text/plain':
    #             raw_text_2 = str(doc_file_2.read(), 'utf-8')
    #         elif file_details == 'application/pdf':
    #             raw_text_2 = read_pdf(doc_file_2)
    #         else:
    #             raw_text_2 = docx2txt.process(doc_file_2)
    #     elif doc_file_2 is None:
    #         raw_text_2 = st.text_area('Upload second document manually')

    #     a=embed_fn([raw_text_1])
    #     b=embed_fn([raw_text_2])
    #     cosine=cosine_similarity(a,b)[0][0]*100
    #     if st.button('Calculate similarity'):
    #         st.write(f'The similarity is {round(cosine,2)} %')




    elif choice=='Email extractor':
        st.subheader('Email extractor')
        doc_file = st.file_uploader('Upload', type=['csv', 'pdf', 'text', 'docx'])
        if doc_file is not None:
            file_details = doc_file.type
            if file_details == 'text/plain':
                raw_text = str(doc_file.read(), 'utf-8')
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            elif file_details == 'application/pdf':
                raw_text = read_pdf(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            else:
                raw_text = docx2txt.process(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
        elif doc_file is None:
            raw_text = st.text_area('Enter your input')


        tasks_list = ["Emails"]
        task_option = st.sidebar.multiselect("Task", tasks_list, default="Emails")
        task_mapper = {"Emails": nfx.extract_emails(raw_text)}

        all_results = []
        for task in task_option:
            result = task_mapper[task]
            # st.write(result)
            all_results.append(result)
        st.write(all_results)

        with st.beta_expander("Results As DataFrame"):
            result_df = pd.DataFrame(all_results).T
            result_df.columns = task_option
            st.dataframe(result_df)
            #make_downloadable_df(result_df)

    elif choice=='Spelling correction':
        st.subheader('Spell checker and corrector')
        doc_file = st.file_uploader('Upload', type=['csv', 'pdf', 'text', 'docx'])
        if doc_file is not None:
            file_details = doc_file.type
            if file_details == 'text/plain':
                raw_text = str(doc_file.read(), 'utf-8')
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            elif file_details == 'application/pdf':
                raw_text = read_pdf(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            else:
                raw_text = docx2txt.process(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
        elif doc_file is None:
            raw_text = st.text_area('Enter your input')

        spell = SpellChecker()
        misspelled_word_list = raw_text.split()
        misspelled_word = spell.unknown(misspelled_word_list)
        b = spell.correction(raw_text)
        if st.button('Get corrected output'):
            st.write(b)
        if st.button('Analyze'):
            for word in misspelled_word:
                if word != spell.correction(word):
                    st.write('Original word:', word)
                    st.write('correct word:', spell.correction(word))
                    st.write('Suggested words:', spell.candidates(word))
                    #st.write('\n')





    elif choice=='Question and answering':
        st.subheader('Question and Answering system')

        doc_file=st.file_uploader('Upload',type=['csv','pdf','text','docx'])
        #


        if doc_file is not None:
            file_details=doc_file.type
            if file_details=='text/plain':
                raw_text=str(doc_file.read(),'utf-8')
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            elif file_details=='application/pdf':
                raw_text=read_pdf(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
            else:
                raw_text=docx2txt.process(doc_file)
                if st.checkbox('Display original text'):
                    st.write(raw_text)
        elif doc_file is None:
            raw_text = st.text_area('Enter your input')

        st.subheader('Enter your question')
        question=st.text_area('What"s in your mind?')


        # if st.button('Generate answer'):
        #
        #     qna=QnA(question,raw_text)
        #     st.write(qna)

    elif choice=='Text generationText generation':
        pass

    else:
        st.header('About')
        st.write('''
        # This web application is built by *Arindam Mondal* , a student of Masters in Data Analytics.''')










if __name__=='__main__':
    main()
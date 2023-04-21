import os 
import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
import seaborn as sns

# !python -m spacy download es_dep_news_trf
# !python -m spacy download en_core_web_trf
# nltk.download()
import spacy 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from chamd import ChatReader
reader = ChatReader()

# ---------------------------------------------------------------------------- #
#              Function 1: Extract .chat information by researcher             #
# ---------------------------------------------------------------------------- #
def raw_chat_to_dataframe (data_folder_raw):

    # a) Collect all the .cha files
    os.chdir(data_folder_raw)
    chat_files = [i for i in os.listdir() if i.endswith(".cha")]
    
    # b) Load all the .cha files 
    # c) For each file, extract the useful information and convert to dataframe
    output = []
    for i in chat_files:
        try: 
            chat_data = reader.read_file(i)
            chat_df   = pd.DataFrame({
                "data": i,
                "lan_researcher": [i.metadata["corpus"].text for i in chat_data.lines],
                "role": [i.metadata["role"].text for i in chat_data.lines], 
                "age": [i.metadata["age"].text for i in chat_data.lines],
                "sex": [i.metadata["sex"].text for i in chat_data.lines],
                "text": [i.text for i in chat_data.lines]
            })

            # d) Reshape the dataframe to desirable format
            chat_df = chat_df.query("role == 'Target_Child'")
            chat_df = pd.DataFrame({
                "data": i,
                "lan_researcher": chat_df["lan_researcher"].unique(),
                "role": chat_df["role"].unique(),
                "age": chat_df["age"].unique(),
                "sex": chat_df["sex"].unique(),
                "text": "| ".join(chat_df["text"])
            })

            chat_df["age"] = chat_df["age"].str.split(";", expand=True).loc[:,0]
            chat_df["age"] = chat_df["age"].astype(int)
            chat_df["age_category"] = pd.cut(chat_df["age"], bins=[3,6,9,12], labels=["3-6", "6-9", "9-12"])
            chat_df["text"] = chat_df["text"].str.lower()

            output.append(chat_df)

        except: 
            pass
    
    # e) Merge all converted dataframes into one
    # f) Remove the duplicated records
    output = pd.concat(output)
    output = output.drop_duplicates()
    output = output.sort_values(by="data")

    print(output["lan_researcher"].unique()[0], ": ", 
          "N(unique) =", len(output))
    
    output = output.reset_index(drop=True)
    return output

# ---------------------------------------------------------------------------- #
#               Function 2: Generate summary tables by languages               #
# ---------------------------------------------------------------------------- #
def summary_table (dataframe_concate, language):
    n_total = dataframe_concate.data.nunique()
    n_sex_female = dataframe_concate.sex.value_counts()['female']
    n_sex_male = dataframe_concate.sex.value_counts()['male']
    n_sex_unkown = n_total - n_sex_female - n_sex_male
    n_36 = dataframe_concate.age_category.value_counts()['3-6']
    n_69 = dataframe_concate.age_category.value_counts()['6-9']
    n_912 = dataframe_concate.age_category.value_counts()['9-12']
    age_mean = round(dataframe_concate.age.mean(), 1)
    age_std = round(dataframe_concate.age.std(), 1)

    records_summary_dict = {language: {
        "N": dataframe_concate.data.nunique(),
        "N (gender = female/male/unknown)": f"{n_sex_female} / {n_sex_male} / {n_sex_unkown}",
        "N (age = 3-6/ 6-9 / 9-12)": f"{n_36} / {n_69} / {n_912}",
        "Age (mean ± std)": f"{age_mean} ± ({age_std})"
    }}

    records_summary_df = pd.DataFrame(
        records_summary_dict,
        index=["N", "N (gender = female/male/unknown)", "N (age = 3-6/ 6-9 / 9-12)", "Age (mean ± std)"]
    )
    return records_summary_df


# ---------------------------------------------------------------------------- #
#                   Function 3: RQ1 - Average sentence length                  #
# ---------------------------------------------------------------------------- #
def q1_stentence_length (sample_one_row_paragraph):
    '''
    Function for Research Question 1
    1. For each paragraph, split into sentences
    2. For each paragraph, count the number of sentences 
    3. For each sentence, count the number of words
    4. For each paragraph, there will therefore be multiple sentence lengths 
    5. Take the mean of the multiple sentence lengths
    6. Result = 2 & 5
    '''
    paragraph_to_sentences = sample_one_row_paragraph.split("| ")
    n_sentence = len(paragraph_to_sentences)
    count_length_all = np.array([i.count(" ") for i in paragraph_to_sentences])
    count_length_mean = np.mean(count_length_all)
    count_length_mean = round(count_length_mean, 3)

    return [n_sentence, count_length_mean]


# ---------------------------------------------------------------------------- #
#               Function 4: RQ2 - Most used part of speech (POS)               #
# ---------------------------------------------------------------------------- #
def q2_sentence_cleaning (df_dample, spa_or_eng_model):
    '''
    Function - cleaning for Research Question 2
    0. Model download - Fast: 
        !python -m spacy download es_dep_news_trf
        !python -m spacy download en_core_web_sm
    
    0. Model download - Accurate: 
        !python -m spacy download es_dep_news_trf
        !python -m spacy download en_core_web_trf

    1. For dataframe, split all the paragraphs into individual sentences
    2. Concatenate all sentences into a list
    3. For each sentence, split into word tokens
    4. For each set of word tokens, remove the unnecessary words (stopwords / symbols)
    5. For each set of word tokens, remove the word with less than 2 letters
    6. For each set of cleaned word tokens, concatenate back to sentences
    '''

    stopwords = spa_or_eng_model.Defaults.stop_words
    symbols = [',', '.', '!', '?', ':']

    paragraphs_to_sentences = df_dample.text.apply(lambda x: x.split("| "))
    sentence_list = list(np.concatenate(paragraphs_to_sentences).flat)
    word_token = [word_tokenize(i) for i in sentence_list]
    word_token_clean = list(map(lambda x: [i for i in x if not i in stopwords], word_token))
    word_token_clean = list(map(lambda x: [i for i in x if not i in symbols], word_token_clean))
    word_token_clean = list(map(lambda x: [i for i in x if len(i) >1], word_token_clean))
    word_token_clean = [i for i in word_token_clean if i!= []]

    cleaned_words_to_sentences = [" ".join(i) for i in word_token_clean]

    print("Preview: cleaned sentences")
    print(cleaned_words_to_sentences[1:5])

    return cleaned_words_to_sentences

def q2_part_of_speech (cleaned_sentences_list, spa_or_eng_model):
    '''
    Function - POS matching & counting for Research Question 2
    1. Transform the cleaned sentences with the language model
    2. For each word in each sentence, detect the POS 
    3. Store all the possible POSs in a list
    4. For each unique POS, group the related words together
    5. For each unique POS, count the number of related words
    6. Result = 4 & 5
    '''
    
    transformed_sentences = [spa_or_eng_model(i) for i in cleaned_sentences_list]
    words_pos_pair = list(map(lambda x: [(i, i.pos_) for i in x], transformed_sentences))
    pos = list(map(lambda x: [i[1] for i in x], words_pos_pair))
    pos = list(np.concatenate(pos).flat)
    pos_unique = list(set(pos))

    pos_group_dict = {i:[] for i in pos_unique}
    for pos_set in words_pos_pair:
        for i in pos_set:
            i_word, i_pos = i[0], i[1]
            pos_group_dict[i_pos].append(i_word)
    
    pos_count = [len(pos_group_dict[i]) for i in pos_unique]
    pos_count_dict =  dict(zip(pos_unique, pos_count))
    
    output = [pos_group_dict, pos_count_dict]
    return output








# --------------------------------- Reference -------------------------------- #
# https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/Multilingual/Spanish/03-POS-Keywords-Spanish.html#spacy-and-natural-language-processing-nlp
# https://spacy.io/usage/models
# https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy


# ---------------------------------- Archive --------------------------------- #
# aug_06_test = "/Users/noel/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/charlotte_thesis/Spanish-Aguilar/0602.cha"
# chat_06_test = reader.read_file(aug_06_test)
# print(chat_06_test.lines[1].metadata)

# test_df = pd.DataFrame({
#     "lan_researcher": [i.metadata["corpus"].text for i in chat_06_test.lines],
#     "role": [i.metadata["role"].text for i in chat_06_test.lines], 
#     "age": [i.metadata["age"].text for i in chat_06_test.lines],
#     "sex": [i.metadata["sex"].text for i in chat_06_test.lines],
#     "text": [i.text for i in chat_06_test.lines]
# })

# test_df = test_df.query("role == 'Target_Child'")
# test_df = pd.DataFrame({
#     "lan_researcher": test_df["lan_researcher"].unique(),
#     "role": test_df["role"].unique(),
#     "age": test_df["age"].unique(),
#     "sex": test_df["sex"].unique(),
#     "text": "; ".join(test_df["text"])
# })

# test_df

# def q2_part_of_speech_nltk (df_dample, stopwords_list):
#     '''
#     Function for Research Question 2
#     1. For dataframe, split all the paragraphs into individual sentences
#     2. Concatenate all sentences into a list
#     3. For each sentence, split into word tokens
#     4. For each set of word tokens, remove the unnecessary words (stopwords_list or any signs)
#     5. For each set of word tokens, detect the POS 
#     6. From all pairs of word-pos, extract all possible POSs
#     7. For each unique POS, group the related word tokens together
#     8. For each unique POS, count the number of related tokens
#     9. Result = 7 & 8
#     ''' 
#     paragraphs_to_sentences = df_dample.text.apply(lambda x: x.split("| "))
#     sentence_list = list(np.concatenate(paragraphs_to_sentences).flat)

#     word_token = [word_tokenize(i) for i in sentence_list]
#     word_token_clean = list(map(lambda x: [i for i in x if not i in stopwords_list], word_token))
#     word_token_clean = [i for i in word_token_clean if i!= ['.']]
#     word_token_clean = list(map(lambda x: [i for i in x if i!='.'], word_token_clean))
#     word_token_clean = list(map(lambda x: [i for i in x if i!=','], word_token_clean))

#     word_pos_pair = [nltk.pos_tag(i) for i in word_token_clean] 
#     word_pos_list = list(map(lambda x: [i[1] for i in x], word_pos_pair))
#     word_pos_list = list(np.concatenate(word_pos_list).flat)
#     word_pos_unique = list(set(word_pos_list))

#     pos_group_dict = {i:[] for i in word_pos_unique}
#     for pos_set in word_pos_pair:
#         for i in pos_set:
#             i_word, i_pos = i[0], i[1]
#             pos_group_dict[i_pos].append(i_word)

#     pos_unique = pos_group_dict.keys()
#     pos_unique_count = [len(pos_group_dict[i]) for i in pos_unique]
#     pos_count_dict =  dict(zip(pos_unique, pos_unique_count))

#     output = [pos_group_dict, pos_count_dict]
#     return output

# stopwords_eng = set(stopwords.words("english"))
# pos_words_eng = q2_part_of_speech_english(df_sample_eng, stopwords_eng)[0]
# pos_count_eng = q2_part_of_speech_english(df_sample_spa, stopwords_spa)[1]
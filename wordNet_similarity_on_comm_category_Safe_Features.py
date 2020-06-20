import nltk
import csv
import re
import os
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import sys
from itertools import islice

import gensim
import gensim.models.keyedvectors as word2vec

path_to_model = "./models/english-bidirectional-distsim.tagger"
path_to_jar = "./stanford-postagger-2016-10-31/stanford-postagger.jar"

st_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return 'n'
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def wordNet_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    
    # sentence1 = pos_tag(word_tokenize(sentence1))
    sentence1=st_tagger.tag(word_tokenize(sentence1))
    
    # sentence2 = pos_tag(word_tokenize(sentence2))
    sentence2=st_tagger.tag(word_tokenize(sentence2))

    
    # Get the synsets for the tagged words
    #################################################

    # synsets1=[]
    # synsets2=[]
    # for tagged_word in sentence1:
    #     print(tagged_word)
    #     tagged_word = list(tagged_word)
    #     synsets1.append(tagged_to_synset(tagged_word[0],tagged_word[1]))
    # for tagged_word in sentence2:
    #     print(tagged_word)
    #     tagged_word = list(tagged_word)
    #     print(tagged_word)
    #     synsets2.append(tagged_to_synset(tagged_word[0],tagged_word[1]))

    # The code above is the elaboration of code below
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones in the synonym set
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
###########################################################################
    # for syn1 in synsets1:
    #     arr_simi_score = []
    #     print('=========================================')
    #     print(syn1)
    #     print('----------------')
    # for syn2 in synsets2:
    #     print(syn2)
    #     simi_score = syn1.path_similarity(syn2)
    #     print(simi_score)
    #     if simi_score is not None:
    #         arr_simi_score.append(simi_score)
    #         print('----------------')
    #         print(arr_simi_score)
    #     if(len(arr_simi_score) > 0):
    #         best = max(arr_simi_score)
    #         print(best)
    #         score += best
    #         count += 1
    #         # Average the values
    #         print('score: ', score)
    #         print('count: ', count)
    #         score /= count

###########################################################################

    for syn1 in synsets1:
        arr_simi_score = []
        # print('=========================================')
        print("Each word from Synonym se1",syn1)
        # print('----------------')
        for syn2 in synsets2:
            print("Each word from Synonym se2",syn2)
            # simi_score = syn1.path_similarity(syn2)
            simi_score = syn1.wup_similarity(syn2)
            print("word to word path_similarity score",simi_score)
            if simi_score is not None:
                arr_simi_score.append(simi_score)
                print('----------------')
                print(arr_simi_score)
        if(len(arr_simi_score) > 0):
            best = max(arr_simi_score)
            print("best score so far", best)
            score += best
            count += 1
    # Average the values
    print('score: ', score)
    print('count: ', count)
    if count!=0:
        score /= count
    else:
        score=0.0
    return score



def text_cleaning(app_desc):
    app_desc = str(app_desc)
    app_desc=app_desc.replace('\\n','')  # Cool Cleaning stuff use of '\\'
    app_desc=re.sub(r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', app_desc, flags=re.MULTILINE)
    app_desc = re.sub(r'<.*?>', ' ', app_desc)
    app_desc=re.sub(r'[^[a-zA-z ]+]*', ' ',app_desc)
    app_desc=re.sub(r'[^\w]', ' ',app_desc)
    app_desc = re.sub(r'\s+', ' ', app_desc).strip()
    return app_desc


 
"""STOP WORD removal"""
def stop_word_removal(text):
    stop_words = set(stopwords.words('english')) 
    # word_tokens = word_tokenize(text) 
    text = [w for w in text if not w in stop_words] 
    # text=str(text)
    print(text)
    return text 




""" Writing a file by using writer object"""
def fileopener(filename_):
    csvfile=open(filename_,'a',newline='')
    fieldnames = ['AppID','Functional_similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    return writer   

if __name__ == '__main__':
    
    dir_name='.\\Dataset\\GooglePlay2020\\Communication\\comm_top_free'
    with open('.\\Dataset\\GooglePlay2020\\Communication\\comm_top_free\\cleanedText_competitors.csv', encoding='utf-8',errors='ignore') as f_a:
        f_csv_a = csv.DictReader(f_a)
        for line_a in f_csv_a:
            app_id=line_a['AppID']
            print("+++",app_id)
            filename= dir_name+'\\'+'target_'+line_a['AppID']+'_wordNet_cleanedText.csv'
            csv_writer=fileopener(filename)
            app_desc_x=line_a['Functional_Features']
            app_desc_x=text_cleaning(app_desc_x)
            # app_desc_x=sentence_processing(app_desc_x.lower())
            with open('.\\Dataset\\GooglePlay2020\\Communication\\comm_top_free\\cleanedText.csv', encoding='utf-8',errors='ignore') as f_b:
                f_csv_b = csv.DictReader(f_b)
                for line_b in f_csv_b:
                    app_id_b=line_b['AppID']
                    print(app_id_b)
                    app_desc_y=line_b['Functional_Features']
                    app_desc_y=text_cleaning(app_desc_y)
                    # app_desc_y=sentence_processing(app_desc_y.lower())
                    app_simi_score=wordNet_similarity(app_desc_x,app_desc_y)
                    print("sim(description_1,description_2) = ", app_simi_score,"/1.")
                    csv_writer.writerow({'AppID':line_b['AppID'],'Functional_similarity':app_simi_score})




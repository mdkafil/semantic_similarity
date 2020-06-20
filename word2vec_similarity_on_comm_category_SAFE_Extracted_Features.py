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


path_to_model = "./stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger"
path_to_jar = "./stanford-postagger-2016-10-31/stanford-postagger.jar"
st_tagger = StanfordPOSTagger(path_to_model, path_to_jar)


def text_cleaning(app_desc):
    app_desc=re.sub(r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', app_desc, flags=re.MULTILINE)
    app_desc = re.sub('<.*?>', ' ', app_desc)
    app_desc=re.sub(r'[^\w]',' ',app_desc)
    app_desc = re.sub( '\s+', ' ', app_desc ).strip()
    return app_desc

def w2v(s1,s2,wordmodel):
        if(s1==s2):
            return 1.0

        s1wordsset=set(s1)
        s2wordsset=set(s2)
        # print(s1wordsset)
        print(s2wordsset)
        
        if len(s1wordsset & s2wordsset)==0:
            return 0.0

        #if word not in the vocabulary
        vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
        s1 = [w for w in s1 if w in vocab]
        s2 = [w for w in s2 if w in vocab]  

        return wordmodel.n_similarity(s1, s2)


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
 
    return None  #"""NONE creates error for wordnet uknown types"""
    # return 'n'
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return word
    except:
        return None
 
def sentence_processing(text):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    #StanfordPOSTagger gives more accurate POS tag than NLTK pos_tag
    #pos_tag(word_tokenize(sentence1)) gives sometimes incorrect 
    #for example "snap selfies." and "snap selfies"
   #####################################################################

    # sentence1 = pos_tag(word_tokenize(sentence1))

    try:
        text=st_tagger.tag(word_tokenize(text))
    except:
        text = pos_tag(word_tokenize(text))


    # sentence2=st_tagger.tag(word_tokenize(sentence2))
    # # sentence2 = [nltk.WordNetLemmatizer().lemmatize(*tagged_word) for tagged_word in sentence2]
    # print(sentence2)
    
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
    
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in text]
    # synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones in the synonym set
    text = [ss for ss in synsets1 if ss]
    # synsets2 = [ss for ss in synsets2 if ss]
 
    

    ##REMOVE STOP WORDS###
    stop_words = set(stopwords.words('english')) 
    text = [w for w in text if not w in stop_words] 
    # text1=str(text1)

    ###REMOVE DUPLICATES
    text = list(dict.fromkeys(text))

    ###REMOVE words in my custom dictionary
    my_custom_dict =["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
    "http",
    "https",
    "inc",
    "etc",
    "able",
    "typical",
    "yet",
    "otherwise",
    "welcome",
    "none",
    "done",
    "ago",
    "recently",
    "still",
    "wait",
    "today",
    "soon",
    "always", 
    "app",
    "also", 
    "even",
    "ever",
    "available", 
    "please",
    "much",
    "almost",
    "many"]
    
    text = [w for w in text if not w in my_custom_dict] 
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
    with open('.\\Dataset\\GooglePlay2020\\Communication\\comm_top_free\\extractedFeatures_competitors.csv', encoding='utf-8',errors='ignore') as f_a:
        f_csv_a = csv.DictReader(f_a)
        for line_a in f_csv_a:
            app_id=line_a['AppID']
            print("+++",app_id)
            filename= dir_name+'\\'+'target_'+line_a['AppID']+'_SAFE_Features.csv'
            csv_writer=fileopener(filename)
            app_desc_x=line_a['Functional_Features']
            app_desc_x=text_cleaning(app_desc_x)
            app_desc_x=sentence_processing(app_desc_x.lower())
            with open('.\\Dataset\\GooglePlay2020\\Communication\\comm_top_free\\extractedFeatures.csv', encoding='utf-8',errors='ignore') as f_b:
                f_csv_b = csv.DictReader(f_b)
                for line_b in f_csv_b:
                    app_id_b=line_b['AppID']
                    print(app_id_b)
                    app_desc_y=line_b['Functional_Features']
                    app_desc_y=text_cleaning(app_desc_y)
                    app_desc_y=sentence_processing(app_desc_y.lower())
                    app_simi_score=w2v(app_desc_x,app_desc_y,wordmodel)
                    print("sim(description_1,description_2) = ", app_simi_score,"/1.")
                    csv_writer.writerow({'AppID':line_b['AppID'],'Functional_similarity':app_simi_score})

   
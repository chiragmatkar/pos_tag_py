
# coding: utf-8

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import itertools
from tqdm import tqdm
import en
from en import is_connective_word
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
wnl = WordNetLemmatizer()

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

def to_lowercase(label):
    return [word.lower() for word in label]

def generalize_style(style):
    AN  = ['AN_NP', 'AN_ING', 'AN_OF', 'AN_IRR', 'AN']
    VOS = ['VOS_IRR', 'VOS', 'VO']
    DES = ['DES', 'PS', 'DES / EVENT', 'GATEWAY']
    NA = ['NA']
    
    if style in AN:
        return 'AN'
    elif style in VOS:
        return 'VOS'
    elif style in DES:
        return 'DES'
    elif style in NA:
        return 'NA'



#data = data[data.Label != '']
#data.Label = map(lambda x: x.strip(','), data.Label)
#data.Split = map(str.split, data.Label)
#data.Split = map(to_lowercase, data.Split)


prepositions = 'aboard about above across after against along amid among anti around as at before behind                 below beneath beside besides between beyond but by concerning considering despite down during                 except excepting excluding following for from in inside into like minus near of off on onto                 opposite outside over past per plus regarding round save since than through to toward towards                 under underneath unlike until up upon versus via with within without'.split()

not_connectives = 'time year and of'.split()
action_mod = 'manually manual automatic automatically completely complete non-integrated'.split()

two_word_first = 'fill check log send take hand explain set'.split()
two_word_second = 'in out to up'.split()


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


tags = dict({
        'AN_NP': 'ANN',
        'AN_ING': 'ANING',
        'AN_OF': 'ANO',
        'AN_IRR': 'ANI',
        'VOS_IRR': 'VOI',
        'VOS': 'VO',
        'VO': 'VO',
        'NA': 'NA',
        'DES': 'DES',
        'PS': 'DES',
        'AN': 'AN',
        'DES / EVENT': 'DES',
        'GATEWAY': 'DES'        
    })

ordinal = '1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th'.split()
conjunctions = 'and but nor or so yet & /'.split()
#prepositions = 'to with in at by for as'.split()

def process_label(row):
    label = row['Label']
    style = row.Style
    obj = str(row['Business Object']).split()
    action = row.Action.split()
    fragment_a = [word[0:4] for word in action]
    fragment_o = [word[0:4] for word in obj]
    string = ''
    indicator = '<>'
    gen_style = generalize_style(style)
    
    #print("********",label)
    x = iter(label)
    next_word = next(x)
    position = 1
    previous = ''
    for word in label:
        store = ''
        next_word = next(x, None)
        #print('current word: %s, next word: %s' %(word, next_word))    
        
        if position > 1 and is_number(word):
            store = str(str(word) + str(indicator) + 'INT' + str(gen_style) + ', ')
        
        elif previous in two_word_first and word in two_word_second and gen_style == 'VOS':
            store = str(word + indicator + 'VOV-E, ')
        
        elif position == 1 and word in action_mod:
            store = str(word + indicator + 'MODIFIER' + gen_style + ', ')
        
        elif position == 1 and is_number(word):
            store = str(word + indicator + 'START_INT, ')
        
        elif word in ordinal:
            store = str(word + indicator + 'ORDINAL' + gen_style + ', ')

        elif next_word == 'of' and (word in action or word[0:4] in fragment_a):
            store = str(word + indicator + 'OF_VERB-' + gen_style + ', ')
        
        elif word in obj:
            store = str(word + indicator + tags[style] +'O, ')
        
        elif word in action or word[0:4] in fragment_a:
            store = str(word + indicator + tags[style] +'V, ')

        elif word == 'of' and  position == 2 and gen_style == 'AN':
            store = str(word + indicator + 'AN_OF' + ', ')
            
        elif word == 'of' and position > 2:
            store = str(word + indicator + 'OF-' + gen_style + ', ')

        elif word == 'and':
            store = str(word + indicator + 'AND-'+ gen_style +', ')
            
        elif is_connective_word(word) and word not in not_connectives:
            store = str(word) + str(indicator) +'C-' + str(gen_style) + ', ' 

        elif word in obj or word[0:4] in fragment_o:
            store = str(word + indicator + tags[style] +'O, ')

        elif wnl.lemmatize(word, 'v') in action:
            store = str(word + indicator + tags[style] +'V, ')

        elif wnl.lemmatize(word, 'n') in obj:
            store = str(word + indicator + tags[style] +'O, ')

        elif stemmer.stem(word)  in action:
            store = str(word + indicator + tags[style] +'V, ')
        
        elif set('[~!@#$%^*,()-=%_+{}":;\']+$').intersection(word):
            store = str(word) + str(indicator) + 'AD' + str(gen_style) +', '
        
        else:
            pass
            #store = word + indicator + 'misc-' + gen_style + ', '
        
        try:
            if en.verb.tense(word) == 'present participle' and previous == 'is':
                store = str(word + indicator + 'DESV-P, ')
        except: 
            pass
        
        position += 1
        previous = word
        string += store
           
    return string
    

def parse_data():
    result = []
    for i in tqdm(range(len(df))):
        result.append(process_label(df.iloc[i]))
        
            
    for j in range(len(df)):
       #print(result[j])
        df.iloc[j]['Tags'] = result[j]
    df.to_csv('CSV/SAP_HMM_BASIC_TAGS_V20.csv', sep=';')
    print('Parsing finished.')

#print(df)





if __name__ == "__main__":
    xlspath = os.path.join(os.path.dirname(__file__), "target/test_labels_hmm_tagger2.xlsx")
    data = pd.read_excel(xlspath, keep_default_na=False)
    columns = ['Label', 'Split', 'Style', 'Action', 'Business Object', 'Tags']
    df = data[columns]
    parse_data()

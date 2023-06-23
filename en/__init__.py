### CREDITS ##########################################################################################

# Copyright (c) 2007 Tom De Smedt.
# See LICENSE.txt for details.

__author__    = "Tom De Smedt"
__version__   = "1.9.4.2"
__copyright__ = "Copyright (c) 2007 Tom De Smedt"
__license__   = "GPL"

### NODEBOX ENGLISH LINGUISTICS ######################################################################

# The Nodebox English Linguistics library adds grammar inflection and semantic operations to NodeBox.
# You can use it to conjugate verbs, pluralize nouns, write out numbers, find dictionary descriptions
# and synonyms for words, summarise texts and parse grammatical structure from sentences.

# The library bundles WordNet, NLTK, Damian Conway's pluralisation rules, Jason Wiener's Brill tagger,
# several algorithms adopted from Michael Granger's Ruby Linguistics module, 
# John Wisemans implementation of the Regressive Imagery Dictionary, and
# Peter Norvig's spelling corrector.

######################################################################################################

from .article import article
from .commonsense import is_connective , is_basic_emotion , is_persuasive , noun_is_emotion
from .numerals import numerals
from .ordinal import ordinal
from .parser import chunk_rules , sentence_tag , sentence_chunk , sentence_traverse , sentence_find , tag_description
from .singular import singular
from .plural import plural
from .quantify import quantify
from .tags import is_html_tag , is_tag , strip_tags
from .verb import * 
import wn
from .rid import categorise
from .ogden import words


def is_number(value):
    return numerals.is_number(value)

def is_noun(word):
    return wn.is_noun(word)

def is_verb(word):
    return wn.is_verb(word)
    
def is_adjective(word):
    return wn.is_adjective(word)
                           
def is_adverb(word):
    return wn.is_adverb(word)    
        
def is_tag(value):
    return is_tag(value)

def is_html_tag(value):
    return is_html_tag(value)
    
def is_connective_word(word):
    return is_connective(word)

def is_basic_emotion(word):
    return is_basic_emotion(word)

def is_persuasive(word):
    return is_persuasive(word)    

class number:
    
    def ordinal(self, number):
        return ordinal.ordinal(number)
    
    def spoken(self, number):
        return numerals.spoken_number(number)
        
    def quantify(self, number, word):
        return quantify.quantify(word, number)

class list:
    
    def conjunction(self, list, generalize=False):
        return quantify.conjunction(list, generalize)
        
    def flatten(self, list):
        return wn.flatten(list)
    
class noun:
    
    def article(self, word):
        return article.article(word)

    def singular(self, word, custom={}):
        return singular.noun_singular(word, custom)
    
    def plural(self, word, classical=True, custom={}):
        return plural.noun_plural(word, classical, custom)

    def is_emotion(self, word, shallow=False, boolean=True):
        return noun_is_emotion(word, shallow, boolean)

class verb:
    
    def infinitive(self, word):
        return verb_infinitive(word)
    
    def conjugate(self, word, tense="infinitive", negate=False):
        return verb_conjugate(word, tense, negate)
    
    def present(self, word, person="", negate=False):
        return verb_present(word, person, negate)

    def present_participle(self, word):
        return verb_present_participle(word)
        
    def past(self, word, person="", negate=False):
        return verb_past(word, person, negate)

    def past_participle(self, word):
        return verb_past_participle(word)
        
    def tenses(self):
        return verb_all_tenses()
        
    def tense(self, word):
        return verb_tense(word)

    def is_tense(self, word, tense, negated=False):
        return verb_is_tense(word, tense, negated)

    def is_present(self, word, person="", negated=False):
        return verb_is_present(word, person, negated)

    def is_present_participle(self, word):
        return verb_is_present_participle(word)

    def is_past(self, word, person="", negated=False):
        return verb_is_past(word, person, negated)

    def is_past_participle(self, word):
        return verb_is_past_participle(word)

    def is_emotion(self, word, shallow=False, boolean=True):
        return commonsense.verb_is_emotion(word, shallow, boolean)
                        
class adjective:

    def plural(self, word, classical=True, custom={}):
        return plural.adjective_plural(word, classical, custom)

    def is_emotion(self, word, shallow=False, boolean=True):
        return commonsense.adjective_is_emotion(word, shallow, boolean)

class adverb:
    
    def is_emotion(self, word, shallow=False, boolean=True):
        return commonsense.adverb_is_emotion(word, shallow, boolean)

class sentence:
     
    def tag(self, sentence):
         return sentence_tag(sentence)
    
    def chunk(self, sentence):
        return sentence_chunk(sentence)
        
    def chunk_rules(self, list=None):
        if list == None:
            return chunk_rules
        else:
            chunk_rules = list
            
    def traverse(self, sentence, f):
        sentence_traverse(sentence, f)
        
    def find(self, sentence, pattern, chunked=True):
        return sentence_find(sentence, pattern, chunked)
        
    def tag_description(self, postag):
        return tag_description(postag)

class content:
    
    def strip_tags(self, txt, clean=True):
        return strip_tags(txt, clean)
        
    def keywords(self, str, top=10, nouns=True, singularize=True, filters=[]):
        return commonsense.sentence_keywords(str, top, nouns, singularize, filters)
        
    def categorise(self, str):
        return categorise(str)

number = number()
list = list()
noun = noun()
verb = verb()
adjective = adjective()
adverb = adverb()
sentence = sentence()
content = content()
        
def_prefixes = {
    "noun"      : noun, 
    "verb"      : verb, 
    "adjective" : adjective, 
    "adverb"    : adverb
}
defs = ["count_senses", "senses", "gloss", "lexname", 
        "hyponym", "hyponyms", "hypernym", "hypernyms", 
        "antonym", "meronym", "holonym", "meet", "absurd_gloss"]

for p in def_prefixes:
    for f in defs:
        print(f)
        #setattr(def_prefixes[p], f, eval("wordnet."+p+"_"+f))
        
basic = words
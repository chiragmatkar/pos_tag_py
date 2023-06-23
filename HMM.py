
# coding: utf-8

# <h1>Stratified KFold cross validation - concatenated datasets</h1>

# In[1]:

import pandas as pd
import numpy as np
import csv
import itertools
import math
import pickle
from tqdm import tqdm
from sklearn import metrics
from auxiliary_functions import * 
from collections import Counter
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

# Augmentation Functions
class augmentObject():

    def __init__(self, item):
        self.label = item.Label
        self.style = item.Style
        self.split = item.Split
        self.tags = item.Tags.split()

def parseAugmentList(data):
    result = []
    for index, row in data.iterrows():
        result.append(augmentObject(row))
    return result

def normalize(x):
    return x / sum(x)

def initialize_augmentation_matrix(columns):
    return pd.DataFrame(columns=columns)

def fill_augmentation_matrix(dataObjects, columns):
    augmentation_matrix = initialize_augmentation_matrix(columns)
    
    for item in dataObjects:
        row = pd.DataFrame(columns=columns, index=[item.label]).fillna(0.)
        for tag in item.tags:
            row[tag] = row[tag] + 1
        augmentation_matrix = augmentation_matrix.append(row)
    
    augmentation_matrix = augmentation_matrix.groupby(aug_matrix.index).sum()
    
    return augmentation_matrix

# Training Data Functions
class dataObject():

    def __init__(self, item):
        self.label = item.Label
        self.style = item.Style
        self.action = item.Action
        self.bobject = item['Business Object']
        self.split, self.tags = parseTags(item)
        
        
def parseTags(item):
    w, t = splitTags(item.Tags)
    return w, t

def splitTags(tags):
    w = []
    t = []
    tagsets = tags.split(',')
    tagsets = tagsets[0:-1]

    for tagset in tagsets:
        word = tagset.split('<>')[0].strip(' ')
        tag = tagset.split('<>')[1].strip(' ')
        w.append(word)
        t.append(tag)
    
    return w, t

def dataToObject(dataframe):
    objects = []
    for index, row in dataframe.iterrows():
        objects.append(dataObject(row))
    return objects        
        
# Hidden Markov Model
class HMM_R():

    def __init__(self, objects, augmentation):
        self.objects = objects
        self.A = None
        self.B = None
        self.Pi = None
        self.Q_count = None
        self.O_count = None
        self.O = None
        self.Q = None
        self.show_path = False
        self.aug = augmentation
        self.buildModel()
       
    def buildModel(self):
        # Retrieve states (Q) and observations  (O) and their counts. 
        self.Q = sorted(list(set(sum([x.tags for x in self.objects], []))))
        self.O = sorted(list(set(sum([x.split for x in self.objects], []))))
        # Tag sequence as presented to the model
        self.Q_seq = [x.tags for x in self.objects]

        # Word sequence as presented to the model
        self.W_seq = [x.split for x in self.objects]
        
        # Initialize empty transition matrix A and emission matrix B
        self.A = pd.DataFrame(columns=self.Q, index=self.Q).fillna(0.)
        self.B = pd.DataFrame(columns=self.Q, index=self.O).fillna(0.)
        self.Pi = pd.DataFrame(columns=['P'], index=self.Q).fillna(0.)
        
        ### Fill Transition Matrix A ###
        # Count the transitions from the current to the next in B_mat and divide by the total.
        for sub in self.Q_seq:
            x = iter(sub)
            count = 0
            next(x)
            while(True):
                try:
                    current = sub[count]
                    count += 1
                    trans = next(x)
                    self.A[trans][current] += 1
                except:
                    break
        
        self.A = self.A.apply(normalize, axis=0)
        
        #for q in self.Q:
        #    self.A[q] = self.A[q] / float(sum(self.A[q]))
            
        self.A = self.A.fillna(0.)
        self.A['VOV']['VOV'] = 0.
        self.A['misc-VOS']['ANNO'] = 0.
        self.A['ANNO']['misc-VOS'] = 0.
        self.A['ADAN']['ADAN'] = 0.
        self.A['ADVOS']['ADVOS'] = 0.
        self.A['ADNA']['ADNA'] = 0.
        self.A['ADDES']['ADDES'] = 0.
        
        ### Fill Emission Matrix B ###
        # Count all occurrences
        #self.C = pd.DataFrame(columns=self.B.columns, index=self.O).fillna(0.)
                
        for q, w in zip(self.Q_seq, self.W_seq):
            for qs, ws in zip(q,w):
                self.B[qs][ws] += 1
        self.B = self.B.apply(normalize, axis=1)

        
        ### Start Probability Matrix Pi ###
        for q in self.Q_seq:
            self.Pi['P'][q[0]] += 1
        
        #for q in self.Q:
        #    self.Pi['P'][q] = 1./len(self.Q)
        #self.Pi['P'] = self.Pi['P'] / float(len(self.Q))
        
        self.Pi['P'] = self.Pi['P'] / float(sum(self.Pi['P']))
        
        # Room for constraints
        self.Pi.ix['VOO'] = 0
        self.Pi.ix['misc-VOS'] = 0
        self.Pi.ix['INTVOS'] = 0
        self.Pi.ix['INTAN'] = 0
      
        # Initialize Viterbi parameters
        self.transProb = self.A.as_matrix()
        self.initialProb = self.Pi.as_matrix(['P'])
        self.obsProb = self.B
        self.tags = self.A.columns
        self.N = len(self.Q)
        
    
    def Obs(self, obs):
        shape = len(self.obsProb.columns)
        return self.obsProb.ix[obs].as_matrix().reshape(shape,1)
    
    def viterbi(self, obs):
        trellis = np.zeros((self.N, len(obs)), dtype='float64')
        backpt = np.ones((self.N, len(obs)), 'int32') * -1
        try: 
            trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))
        except:
            trellis[:, 0] = np.squeeze(self.initialProb * np.ones((len(self.initialProb), 1)))

        for t in xrange(1, len(obs)):
            try:
                T = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            except:
                T = (trellis[:, t-1, None].dot(np.ones((len(self.initialProb), 1)).T) * self.transProb).max(0)
            if not np.any(T):
                T = (trellis[:, t-1, None].dot(np.ones((len(self.initialProb), 1)).T) * self.transProb).max(0)
            
            trellis[:, t] = T
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
            
        tokens = [trellis[:, -1].argmax()]
        for i in xrange(len(obs)-1, 0, -1):
            tokens.append(backpt[tokens[-1], i])
        return [self.tags[i] for i in tokens[::-1]]


    #def decodeToTag(self, tokens):
    #    return [self.tags[i] for i in tokens]

def errors_to_csv(errors, filename):
    with open(filename + '.csv', 'wb') as csvfile:
        fieldnames = ['Label', 'True Style', 'Predicted Style', 'True Tags', 'Predicted Tags']
        writer = csv.DictWriter(csvfile, delimiter =';', fieldnames=fieldnames)
        writer.writeheader()
        for item in errors:
            writer.writerow({'Label': item[0], 'True Style': item[1], 'Predicted Style': item[2], 'True Tags': item[3], 'Predicted Tags': item[4]})

        writer.writerow({'Label': sum(CM)})
    
def actionObject(samples, model):
    label_list = []
    true_list = []
    pred_list = []
    
    for obj in samples:
        pred = model.viterbi(obj.split)
        word = obj.split
        true = obj.tags
        
        label_list.append(obj.label)
        true_list.append(true)
        pred_list.append(pred)
        
        #print str('Label: ' + str(word) + '\npredicted: ' + str(pred) +\
                  #'\nactual: ' + str(true) + '\nType: ' + str(obj.style) + '\n\n')
    return label_list, true_list, pred_list

def tag_style(tag):
    if tag in VOS:
        return 'VOS'
    elif tag in NA:
        return 'NA'
    elif tag in DES:
        return 'DES'
    elif tag in AN:
        return 'AN'
    else: 
        raise 'Unclassified tag'

def vote_style(tags,style):
    styles = {'VOS': 0, 'AN': 0, 'NA': 0, 'DES': 0}
    
    if style == 'NA':
        result = 0
        for tag in tags:
            if tag in OBJECT: 
                result += 0
            else:
                result += 1
        if result == 0:
            return 'NA'

    for tag in tags:
        try:
            style = tag_style(tag)
            styles[style] += 1
        except:
            pass
    return max(styles, key=styles.get) 


# In[ ]:

OBJECT = 'VOO VOIO misc-VOS ANO ANNO ANIO ANINGO ANOO misc-AN NAO misc-NA DESO misc-DES'.split()
VOS = ['MODIFIERVOS','3RDSPVOS','VOO','ADVOS','AND-VOS','C-VOS','VOV','misc-VOS','AD_VOS','MISC_VOS','OF-VOS','VOIV','VOIO','OF_VERB-VOS', 'VOV-E']
AN = ['MODIFIERAN','3RDSPAN','misc-AN','ADAN','AND-AN','OF-AN','C-AN','AN_OF','AD_AN','ANO','ANV', 'ANNO', 'ANNV','ANIO','ANIV','ANINGO','ANOV','ANINGV','ANOO','OF_VERB-AN']
NA = ['MODIFIERNA','3RDSPNA','NAO','NAV','OF_NA','AND-NA','AD_NA','C-NA','misc-NA','OF-NA','ADNA','OF_VERB-NA',]
DES = ['MODIFIERDES','3RDSPDES','OF-DES','misc-DES','AD_DES','AND-DES','ADDES','C-DES','DESV','DESO','OF_VERB-DES', 'DESV-P']
STYLES = ['AN', 'NA', 'VOS', 'DES']

#Load Dataset and augmentation
AI = pd.read_csv('CSV/COMPLETE_V19.csv', sep=';', keep_default_na=False)
AUG = dataToObject(pd.read_csv('CSV/aug_list.csv', sep=';', keep_default_na=False))
AI_objects = dataToObject(AI)
skf = StratifiedKFold(AI.Style, n_folds=10, random_state=100)

#Evaluation results
P = []
R = []
F = []
CM = []
ERRORS = []

for train,test in tqdm(skf):  
    cm = pd.DataFrame(index=STYLES, columns=STYLES).fillna(0)
    
    #Model generation and testfold creation.
    m_fold = HMM_R([AI_objects[i] for i in train] + AUG, None)
    test_fold = [AI_objects[i] for i in test]
    
    for item in test_fold:
        label = item.split
        style = generalize_style(item.style)
        predicted_tags = m_fold.viterbi(label) 
        predicted = vote_style(predicted_tags, style)
        
        if style == predicted:
            model_predictions.append((label, predicted_tags, item.tags, style))
        
        cm[predicted][style] += 1
        if style != predicted:
            ERRORS.append((label, style, predicted, item.tags, predicted_tags))
    
    #Evaluation Steps
    for style in cm.columns:
        p = round(cm[style][style] / float(sum(cm.ix[style])),3)
        if math.isnan(p):
            p = 0.
        r = round(cm[style][style] / float(sum(cm.T.ix[style])),3)
        if math.isnan(r):
            r = 0.
        try:
            f = round((2.*p*r)/(p+r),2)
        except:
            f = 0

        P.append(p)
        R.append(r)
        F.append(f)
        
    CM.append(cm)


# In[ ]:

def mean_(means, counts):
    return np.dot(means, counts)/sum(counts)

P_AN = np.mean(P[0::4])
P_NA = np.mean(P[1::4])
P_VOS = np.mean(P[2::4])
P_DES = np.mean(P[3::4])
R_AN = np.mean(R[0::4])
R_NA = np.mean(R[1::4])
R_VOS = np.mean(R[2::4])
R_DES = np.mean(R[3::4])
F_AN = np.mean(F[0::4])
F_NA = np.mean(F[1::4])
F_VOS = np.mean(F[2::4])
F_DES = np.mean(F[3::4])

AI_AN = AI.Style.value_counts().AN_NP + AI.Style.value_counts().AN_OF + AI.Style.value_counts().AN_IRR +AI.Style.value_counts().AN_ING + AI.Style.value_counts().AN
AI_NA = AI.Style.value_counts().NA
AI_VOS = AI.Style.value_counts().VOS + AI.Style.value_counts().VOS_IRR + AI.Style.value_counts().VO
AI_DES = AI.Style.value_counts().PS + AI.Style.value_counts().DES + AI.Style.value_counts()['DES / EVENT']

P_means = [P_AN,  P_VOS, P_DES]
R_means = [R_AN,  R_VOS, R_DES]
F_means = [F_AN,  F_VOS, F_DES]
counts = [AI_AN,  AI_VOS, AI_DES]

print('\tP\tR\tF')
print('AN:\t%.2f\t%.2f\t%.3f') %(P_AN, R_AN, F_AN)
print('NA:\t%.2f\t%.2f\t%.3f') % (P_NA, R_NA, F_NA)
print('VOS:\t%.2f\t%.2f\t%.3f') % (P_VOS, R_VOS, F_VOS)
print('DES:\t%.2f\t%.2f\t%.3f') % (P_DES, R_DES, F_DES)
print('Mean:\t%.2f\t%.2f\t%.3f') % (mean_(P_means, counts),mean_(R_means, counts),mean_(F_means, counts))


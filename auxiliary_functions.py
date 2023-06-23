
# coding: utf-8

# In[1]:

from sklearn.feature_extraction import DictVectorizer

def get_style(prediction):
    count = {'VOS': 0, 'AN': 0, 'DES': 0, 'NA': 0}
    check = {'VOS': ['VOIO', 'VOIV', 'VOO', 'VOV', 'OF_VERB'], 
             'AN': ['ANNV', 'ANNO', 'ANINGV', 'ANNINGO', 'ANOV', 'ANOO', 'ANIV', 'ANIO', 'ANO', 'ANV'], 
             'DES': ['DESV', 'DESO', 'DES / EVENT', 'GATEWAY'], 
             'NA': ['NAV', 'NAO']}
    if prediction[0] == 'OF_VERB':
        return 'VOS'
    else:
        for tag in prediction:
            for key, items in check.items():
                if tag in items:
                    count[key] += 1
    return max(count, key=count.get)

def count_tags(prediction, model):
    d = dict.fromkeys(list(model.Q), 0)
    v = DictVectorizer(sparse=False)
    for tag in prediction:
        d[tag] += 1
    
    X = v.fit_transform(d)[0]
    return X
    
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

def predict_styles(objects, model):
    #Requires samples from a basic tagged dataframe together with the HMM model. 
    actual_style = []
    predicted_style = []
    
    if len(objects) == 1:
        predicted_style.append(get_style(model.viterbi(objects.split)))
        actual_style.append(generalize_style(objects.style))
    else:
        for obj in objects:
            predicted_style.append(get_style(model.viterbi(obj.split)))
            actual_style.append(generalize_style(obj.style))

    X = np.asarray(actual_style)
    Y = np.asarray(predicted_style)

    return X, Y

def evaluate_errors(X, Y, objects, model): 
    #X is the actual style, Y is the predicted style.
    error_indices = []
    R = X == Y
    for i in range(len(R)):
        if R[i] == False:
            error_indices.append(i)

    print('The following objects have errors:')
    print error_indices
    
    print('\nActual: Predicted:\tLabel:\n')
    
    for index in error_indices:
        print('%s\t%s\t\t%s') %(X[index], Y[index], objects[index].label)
        pred = model.viterbi(objects[index].split)
        print('True tags:\t\t%s') %(objects[index].tags)
        print('Predicted tags:\t\t%s\n') %(pred)
        
    print('\nTotal accuracy is %.3f. There were %d errors in %d predictions') %(R.sum()/float(len(R)), (len(R)-R.sum()), len(R))
    
def conf_matrix_styles(X, Y):
    #X is the actual style, Y is the predicted style.
    styles = ['VOS', 'AN', 'DES', 'NA', 'TOTAL']
    conf = pd.DataFrame(columns=styles, index=styles).fillna(0)

    for pred, true in zip(Y, X):
        conf[pred][true] += 1
        conf[pred]['TOTAL'] += 1
        conf['TOTAL'][true] +=1
        conf['TOTAL']['TOTAL'] +=1
    return conf

#Precision, recall and F-score
def precision(label, matrix):
    return matrix.ix[label, label] / float(matrix.ix[label, 'TOTAL'])

def recall(label, matrix):
    return matrix.ix[label, label] / float(matrix.ix['TOTAL', label])

def p_r(x):
    columns = np.delete(x.columns.values, -1)
    p_r = pd.DataFrame(columns=columns, index=['P', 'R', 'F1']).fillna(0).astype(np.float64)
    for item in columns:
        p_r[item]['P']= precision(item ,x)
        p_r[item]['R'] = recall(item ,x)
        p_r[item]['F1'] = (2*((recall(item,x) * precision(item,x))/(recall(item,x) + precision(item,x))))
    
    p_r = p_r.fillna(0)
    p_r = p_r.round(2)
    return p_r

def predict(model):
    print('Enter a label manually, "q" to quit')
    label = None
    while label != 'q':
        label = str(raw_input('Enter label: '))
        pred = model.viterbi(label.split())
        print('Tags predicted: %s') %(pred)
        print('Style predicted: %s\n') %(get_style(pred))
        
def accuracy(true, pred):
    tags = ['VOIO', 'VOIV', 'VOV', 'VOO', 'ANNV', 'ANNO', 'ANINGV', 'ANINGO', 'ANOV', 'ANOO', 'ANIV', 'ANIO', 'NAV', 'NAO', 'DESV', 'DESO', 'OTHER']
    df_tags = ['VOIO', 'VOIV', 'VOV', 'VOO', 'ANNV', 'ANNO', 'ANINGV', 'ANINGO', 'ANOV', 'ANOO', 'ANIV', 'ANIO', 'NAV', 'NAO', 'DESV', 'DESO', 'OTHER', 'TOTAL']
    pred_df = pd.DataFrame(columns=df_tags, index=df_tags).fillna(0).astype(np.float64)
    
    for P, T in zip(pred,true):
        for item_pred, item_true in zip(P,T):
            if item_pred not in tags and item_true in tags:
                pred_df['OTHER'][item_true] += 1
                pred_df['OTHER']['TOTAL'] += 1
                pred_df['TOTAL'][item_true] += 1
            elif item_pred not in tags and item_true not in tags:
                pred_df['OTHER']['OTHER'] += 1
                pred_df['OTHER']['TOTAL'] += 1
                pred_df['TOTAL']['OTHER'] += 1
            elif item_pred in tags and item_true in tags:
                pred_df[item_pred][item_true] += 1
                pred_df[item_pred]['TOTAL'] += 1
                pred_df['TOTAL'][item_true] += 1
            elif item_pred in tags and item_true not in tags:
                pred_df[item_pred]['OTHER'] += 1
                pred_df[item_pred]['TOTAL'] += 1
                pred_df['TOTAL']['OTHER'] += 1   
                
    pred_df['TOTAL']['TOTAL'] = sum(pred_df['TOTAL'])
    
    return pred_df   

 

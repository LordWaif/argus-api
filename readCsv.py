import pandas as pd
from small_text import TextDataset
# Read CSV
dataframe = pd.read_csv("multilabel_balanced.csv")
dataframe = dataframe.sample(frac=1)
LABELS = list(dataframe.columns[1:])
LABEL2INT = lambda x: [LABELS.index(i) for i in x]

def get_label(row):
    rotulos = []
    for c in dataframe.columns[1:]:
        if row[c]==1:
            rotulos.append(c)
    return rotulos

dataframe['rotulos'] = dataframe.apply(get_label, axis=1)

from numpy import arange,tile,asarray,int8,zeros

num_classes = len(LABELS)
target_labels = arange(num_classes)

def get_labels(row):
    bin_labels = []
    for c in LABELS:
        if row[c]==1:
            #bin_labels.append(LABELS.index(c))
            bin_labels.append(1)
        else:
            bin_labels.append(0)
    return bin_labels
dataframe['labels'] = dataframe.apply(get_labels, axis=1)
dataframe = dataframe.drop(columns=LABELS)


from scipy.sparse import csr_matrix
def to_csr_matrix(y):
    _y = list()
    for i in y:
        aux = zeros(7)
        for j in i:
            aux[j] = 1
        _y.extend(aux)
    _y = asarray(_y)
    indices = tile(target_labels,len(y))
    indptr = asarray(range(0,len(y)*num_classes+1,num_classes))
    return csr_matrix((_y,indices,indptr),shape=(len(y),num_classes),dtype=int8)

y = to_csr_matrix(dataframe['labels'])
df_train = dataframe[0:900]
df_train.reset_index(drop=True,inplace=True)
df_test = dataframe[900:]
df_test.reset_index(drop=True,inplace=True)
train = TextDataset.from_arrays(df_train['text'], y[0:900], target_labels=target_labels)
test = TextDataset.from_arrays(df_test['text'], y[900:], target_labels=target_labels)
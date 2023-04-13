# Define listener
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from numpy import where,asarray
# Define some helper variables
ACCURACIES = []
HAMMING_LOSS = []
TRUSTS = []

def trusting(to_log,proba):
    i,j = where((asarray(to_log.y.todense())*proba)>0)
    trust = (asarray(to_log.y.todense())*proba)
    acc_trust = 0
    for _j,_i in list(zip(j,i)):
        acc_trust+=trust[_i][_j]
    avg_trust = acc_trust/len(j)

    i,j = where((1-(asarray(to_log.y.todense()))*proba)>0)
    not_trust = (asarray(to_log.y.todense())*proba)
    acc_not_trust = 0
    for _j,_i in list(zip(j,i)):
        acc_not_trust+=not_trust[_i][_j]
    avg_not_trust = acc_not_trust/len(j)
    avg_trust_delta = avg_trust - avg_not_trust
    return avg_trust_delta

def accuracy(to_log,csr):
    accuracy = accuracy_score(
        asarray(to_log.y.todense()),
        asarray(csr.todense()),
    )
    return accuracy

def hamming(to_log,csr):
    hamming = hamming_loss(
        asarray(to_log.y.todense()),
        asarray(csr.todense()),
    )
    return hamming
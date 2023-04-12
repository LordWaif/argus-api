#%%
import argilla as rg
import pandas as pd
from small_text import TransformersDataset
#%%
#Authentication
rg.init(
    api_url="http://procyon.tce.pi.gov.br:6900",
    api_key="admin.apikey",
    workspace='admin'
)
#%%
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

import numpy as np
from small_text import TextDataset

num_classes = len(LABELS)

target_labels = np.arange(num_classes)

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
#%%
# Adapt to Small Text
from scipy.sparse import csr_matrix

fl_labels = []
for i in dataframe['labels']:
    fl_labels.extend(i)
data = fl_labels
indices = np.tile(target_labels,dataframe['labels'].shape[0])

valores = [0]
for i in range(1, 1121):
    valor_atual = valores[i-1] + 7
    valores.append(valor_atual)
indptr = np.asarray(range(0,7841,7))

y = csr_matrix((data,indices,indptr),shape=(1120,7),dtype=np.int)

df_train = dataframe[0:900]
df_train.reset_index(drop=True,inplace=True)
df_test = dataframe[900:]
df_test.reset_index(drop=True,inplace=True)
train = TextDataset.from_arrays(df_train['text'], y[0:900], target_labels=target_labels)
test = TextDataset.from_arrays(df_test['text'], y[900:], target_labels=target_labels)
#%%
# Instance SetFit
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory


num_classes = 7

sentence_transformer_model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
clf_factory = SetFitClassificationFactory(setfit_model_args, 
                                          num_classes,classification_kwargs={'multi_label':True})

#%%
# Instance ActiveLearning
from small_text import (
    PoolBasedActiveLearner, 
    BreakingTies,
    SubsamplingQueryStrategy
)

# define a query strategy and initialize a pool-based active learner
query_strategy = SubsamplingQueryStrategy(BreakingTies())
# suppress progress bars in jupyter notebook
setfit_train_kwargs = {'show_progress_bar': True}
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train, fit_kwargs={'setfit_train_kwargs': setfit_train_kwargs})
#%%
# Insert Initial Data
from small_text import random_initialization
# Fix seed for reproducibility
np.random.seed(42)


# Number of samples in our queried batches
NUM_SAMPLES = 2

# Randomly draw an initial subset from the data pool
initial_indices = random_initialization(train, NUM_SAMPLES)

# Choose a name for the dataset
DATASET_NAME = "lima"

# Define labeling schema
settings = rg.TextClassificationSettings(label_schema=LABELS)

# Create dataset with a label schema
rg.configure_dataset(name=DATASET_NAME, settings=settings)

# Create records from the initial batch
records = [
    rg.TextClassificationRecord(
        text=df_train.iloc[idx].text,
        metadata={"batch_id": 0},
        id=idx,
        multi_label=True
    )
    for idx in initial_indices
]

# Log initial records to Rubrix
rg.log(records, DATASET_NAME)
#%%
from setfit import SetFitTrainer,SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss
from datasets.arrow_dataset import Dataset

# Load the handlabelled dataset from Rubrix
train_ds = rg.load("lima").prepare_for_training()

# Load the full imdb test dataset
test_ds = Dataset.from_dict({'label':df_test['rotulos'].apply(LABEL2INT),'id':[str(i) for i in df_test.index],'text':df_test['text'],'binarized_label':df_test['labels']})


# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # The number of text pairs to generate
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

#%%
# Define listener
from argilla.listeners import listener
from sklearn.metrics import accuracy_score

# Define some helper variables
ACCURACIES = []
def to_csr_matrix(y):
    _y = list()
    for i in y:
        aux = np.zeros(7)
        for j in i:
            aux[j] = 1
        _y.extend(aux)
    _y = np.asarray(_y)
    indices = np.tile(np.arange(num_classes),len(y))
    indptr = np.asarray(range(0,len(y)*num_classes+1,num_classes))
    return csr_matrix((_y,indices,indptr),shape=(len(y),num_classes),dtype=np.int)

# Set up the active learning loop with the listener decorator
@listener(
    dataset=DATASET_NAME,
    query="status:Validated AND metadata.batch_id:{batch_id}",
    condition=lambda search: search.total==NUM_SAMPLES,
    execution_interval_in_seconds=3,
    batch_id=0
)
def active_learning_loop(records, ctx):

    # 1. Update active learner
    print(f"Updating with batch_id {ctx.query_params['batch_id']} ...")
    y = np.array([LABEL2INT(rec.annotation) for rec in records])
    y = to_csr_matrix(y)
    
    # initial update
    if ctx.query_params["batch_id"] == 0:
        indices = np.array([rec.id for rec in records])
        active_learner.initialize_data(indices, y)
    # update with the prior queried indices
    else:
        active_learner.update(y)
    print("Done!")

    # 2. Query active learner
    print("Querying new data points ...")
    queried_indices = active_learner.query(num_samples=NUM_SAMPLES)
    new_batch = ctx.query_params["batch_id"] + 1
    new_records = [
        rg.TextClassificationRecord(
            text=df_train.iloc[idx].text,
            metadata={"batch_id": new_batch},
            id=idx,
        )
        for idx in queried_indices
    ]

    # 3. Log the batch to Rubrix
    rg.log(new_records, DATASET_NAME)

    # 4. Evaluate current classifier on the test set
    print("Evaluating current classifier ...")
    accuracy = accuracy_score(
        test.y,
        active_learner.classifier.predict(test),
    )

    ACCURACIES.append(accuracy)
    ctx.query_params["batch_id"] = new_batch
    print("Done!")

    print("Waiting for annotations ...")
#%%
# Start active learning
active_learning_loop.start()
#%%
active_learning_loop._LOGGER
#%%
active_learning_loop.stop()
# %%

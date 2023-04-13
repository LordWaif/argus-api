from auth import auth
rg=auth()
# Insert Initial Data
from small_text import random_initialization
from numpy.random import seed
# Fix seed for reproducibility
seed(42)
from readCsv import (
    train,
    LABELS,
    df_train)

# Instance SetFit
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory


num_classes = 7

sentence_transformer_model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
clf_factory = SetFitClassificationFactory(setfit_model_args, 
                                          num_classes,classification_kwargs={'multi_label':True,'mini_batch_size':1})

# Number of samples in our queried batches
NUM_SAMPLES = 5

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
#%%
# Log initial records to Rubrix
rg.log(records, DATASET_NAME)
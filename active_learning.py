#%%
#Authentication
from log import LOGGER
import numpy as np
from readCsv import (
    target_labels,
    LABEL2INT,
    LABELS,
    df_train,
    TextDataset,
    to_csr_matrix)

#%%
from setfit_conf import(
    DATASET_NAME,
    NUM_SAMPLES,
    active_learner,
    rg
)
#%%
from metrics import(
    ACCURACIES,
    TRUSTS,
    HAMMING_LOSS,
    trusting,
    accuracy,
    hamming
)
#%%
from argilla.listeners import listener
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
    y_to_log = \
    to_csr_matrix(
        np.array(
            [
                LABEL2INT(rotulos) 
                for rotulos in 
                df_train.iloc[queried_indices]['rotulos']
            ]
        )
    )
    x_to_log = df_train.iloc[queried_indices]['text']
    to_log = TextDataset.from_arrays(
                x_to_log,
                y_to_log,
                target_labels=target_labels)
    proba_log = active_learner.classifier.predict(to_log,return_proba=True)[1]
    new_records = [
        rg.TextClassificationRecord(
            text=df_train.iloc[idx].text,
            prediction=list(zip(LABELS,proba_log[ind])),
            metadata={"batch_id": new_batch},
            id=idx,
            multi_label=True
        )
        for ind,idx in enumerate(queried_indices)
    ]

    # 3. Log the batch to Rubrix
    rg.log(new_records, DATASET_NAME)

    # 4. Evaluate current classifier on the test set
    print("Evaluating current classifier ...")
    csr,proba = active_learner.classifier.predict(to_log,return_proba=True)
    ## Têm um detalhe aqui, ele tá usando os y do cs original e não do que foi rotulado

    LOGGER.info(f'TRUSTING {trusting(to_log,proba)}')
    #TRUSTS.append(trusting(to_log,proba))
    LOGGER.info(f'ACCURACIES {accuracy(to_log,csr)}')
    #ACCURACIES.append(accuracy(to_log,csr))
    LOGGER.info(f'HAMMING LOSS {hamming(to_log,csr)}')
    #HAMMING_LOSS.append(hamming_loss(to_log,csr))
    ctx.query_params["batch_id"] = new_batch
    print("Done!")

    print("Waiting for annotations ...")
#%%
# Start active learning
active_learning_loop.start()
#%%
#active_learning_loop._LOGGER
#%%
#active_learning_loop.stop()
# %%

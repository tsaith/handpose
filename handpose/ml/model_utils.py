import numpy as np

def get_reduced_proba(pd):
    """
    Return the reduced probability.

    pd: nd-array
        Probability distribution with shape of (num_preds, num_classes).

    """
    index_pred, index_class = np.unravel_index(pd.argmax(), pd.shape)
    reduced_proba = pd[index_pred]

    return reduced_proba


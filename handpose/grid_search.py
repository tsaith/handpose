import numpy as np
from operator import itemgetter



def score_report(scores, num_top=3):
    """
    Report the best scores from grid search.

    Parameters:
    scores: list
        Grid scores.
    num_top: integer
        Number of top scores. 
    """

    top_scores = sorted(scores, key=itemgetter(1), reverse=True)[:num_top]
    for i, score in enumerate(top_scores):
        print("Mean score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

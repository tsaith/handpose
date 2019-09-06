import numpy as np
from operator import itemgetter

def search_result_to_file(result, filename='search_result.txt'):
    """
    Save the searching result to file.
    
    Parameters:
    
    result: dict
        Search result.
    filename: string
        File name.
    """

    print("Write the result to file: {}".format(filename))

    f = open(filename, 'w')

    f.write("Best score: {0:.5f} with {1:s} \n".format(result.best_score_, 
        result.best_params_))
    for params, mean_score, scores in result.grid_scores_:
        f.write("{:.5f} (std:{:.5f}) with: {} \n".format(scores.mean(), 
            scores.std(), params))

    f.close()


def report_best_scores(scores, num_top=3):
    """
    Report the best scores obtained from grid search.

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

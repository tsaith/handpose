import numpy as np
import pandas as pd
import os

def list_files(dir_path, keyword="_rec_"):
    """
    List files with the specified keyword.
    """

    files = []
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)) and f.find(keyword) >= 0:
            files.extend([f])

    return files

"""
File factory of vibrational data.
"""

from handpose.vib import vib_file_factory
import os
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--keyword", default="_rec_", help="Keyword")
args = vars(ap.parse_args())

keyword = args['keyword']

# Basename of the current working directory
cwd_basename = os.path.basename(os.getcwd())

dir_path = os.getcwd()

vib_file_factory(".", keyword=keyword,
                 seg_size=700, noise_ratio=1.5, back=100)


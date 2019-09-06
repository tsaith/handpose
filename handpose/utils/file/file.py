import os

def to_str_digits(n, num_digits=4):
    # Convert a number into a string with specified number of digits.

    def add_zero(target):
        return '0' + target

    s = str(n)
    num = len(s)
    num_diff = num_digits - num

    for i in range(num_diff):
        s = add_zero(s)
    return s


def get_file_list(dir_path, ext='jpeg'):
    """
    Get the file list from a directory.
    """
    keyword = '.' + ext

    return list_files(dir_path, keyword=keyword)

def list_files(dir_path, keyword="_rec_"):
    """
    List files with the specified keyword.
    """

    files = []
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)) and f.find(keyword) >= 0:
            files.extend([f])

    return files

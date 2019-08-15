import os


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

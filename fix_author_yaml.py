import os
import re

from config import CORPUS_PATH
from utils import get_ext

if __name__ == "__main__":
    is_corpus_path_exists = os.path.exists(CORPUS_PATH)
    if not is_corpus_path_exists:
        print('path is not found.')

    list_files = os.listdir(CORPUS_PATH)
    list_files = [a for a in list_files if get_ext(a) == 'yaml']

    for file in list_files:
        file_path = '{}/{}'.format(CORPUS_PATH, file)
        is_file_exists = os.path.exists(file_path)
        if not is_file_exists:
            print('file {} not found'.format(file_path))

        f = open(file_path, "r")
        replaced_content = ""

        for line in f:
            line = line.strip()

            # fix next line value
            is_semicolon_exists = re.search(':', line)
            if is_semicolon_exists is None:
                replaced_content = replaced_content[:len(replaced_content)-1] + ' '

            # fix unquoted author
            is_line_author = re.search('^author: ', line)
            is_line_author_w_str = re.search('^author: \'', line)
            if is_line_author is not None and is_line_author_w_str is None:
                new_line = line.replace('\'', '')
                new_line = new_line.replace("author: ", "author: '")
                new_line += "'"
                line = new_line

            # concatenate the new string and add an end-line break
            replaced_content = replaced_content + line + "\n"

        # close the file
        f.close()
        # Open file in write mode
        write_file = open(file_path, "w")
        # overwriting the old file contents with the new/replaced content
        write_file.write(replaced_content)
        # close the file
        write_file.close()

import os
import re

import yaml

from config import CORPUS_PATH
from main import get_ext

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

        os_identity = open(file_path, 'r')
        identity = yaml.safe_load(re.sub('[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]', '', os_identity.read()))
        if identity is None:
            continue

        title = '?'
        author = '?'

        if 'title' in identity:
            title = identity['title']

        if 'author' in identity:
            author = identity['author']

    print('checked complete')

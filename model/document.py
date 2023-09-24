import re
from mpstemmer import MPStemmer

stemmer = MPStemmer()


class Document:
    def __init__(self, doc_id):
        self.id = doc_id
        self.title: str = ''
        self.body: str = ''
        self.author: str | None = None
        self.preprocessed: [str] = []
        self.tf: {[str]: int} = {}

    def set_body(self, body):
        body_cleaned = re.sub(r'[^\w\s]|\d|http\S+', ' ', body.replace('\n', ' '))
        body_lowered = body_cleaned.lower()
        tokenized = [re.sub("[^A-Za-z]", "", a) for a in body_lowered.split(' ') if a != '']
        stopword_file = open('./stopwords.txt')
        stopwords = [a for a in stopword_file.read().replace('\n', ' ').split(' ') if a != '']
        stopwords_removed = [a for a in tokenized if a not in stopwords]
        stemmed = [stemmer.stem(a) for a in stopwords_removed]

        for word in stemmed:
            if word not in self.tf:
                self.tf[word] = 1
                continue

            self.tf[word] += 1

        self.preprocessed = stemmed
        self.body = body

    def set_title(self, title):
        self.title = title

    def set_author(self, author):
        self.author = author


def document_to_json(obj: Document):
    return obj.__dict__
import re

from sklearn.base import TransformerMixin
from nltk.stem.snowball import SnowballStemmer
from natasha import (
    Segmenter,
    MorphVocab, 
    NewsMorphTagger,  
    NewsEmbedding,  
    Doc
)
from tqdm import tqdm

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

snow_stemmer = SnowballStemmer(language='russian')

tqdm.pandas()


class TextTransformer(TransformerMixin):
     
    def __init__(self, stop_words, stem=False):
        self.stop_words = stop_words
        self.ENG_SYMBOLS_OR_DIGITS_REGEX = r'([a-zA-Z0-9]+_?-?,?)'
        self.stem = stem

    def _preprocess(self, text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)  

        for token in doc.tokens:
            token.lemmatize(morph_vocab)

        return doc

    def _clean(self, doc):
        if self.stem:
            text = [
                snow_stemmer.stem(token.lemma.lower()) for token in doc.tokens 
                if snow_stemmer.stem(token.lemma.lower()) not in self.stop_words
            ]
        else:
            text = [
                token.lemma.lower() for token in doc.tokens if token.lemma.lower() not in self.stop_words
            ]

        return ' '.join(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['doc'] = X['text'].progress_apply(self._preprocess)
        X['num_tokens'] = X['doc'].apply(lambda doc: len(doc.tokens))
        X['num_sentences'] = X['doc'].apply(lambda doc: len(doc.sents))
        X['cleaned'] = X['doc'].apply(
            self._clean
        ).apply(
            lambda text: re.sub(self.ENG_SYMBOLS_OR_DIGITS_REGEX, '', text)
        )
        return X
import re
import string
from functools import partial

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

#  for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
#  bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#  for word embedding
import gensim
from gensim.models import Word2Vec
import pickle


#  convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[\d*]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)


def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


# building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def full_preprocessing(modelw, series):
    series = series.apply(finalpreprocess)
    series = [nltk.word_tokenize(i) for i in series]
    return modelw.transform(series)


def generate_models():
    df_train = pd.read_csv('NLP/train.csv')
    df_test = pd.read_csv('NLP/test.csv')

    df_train['clean_text'] = df_train['Text'].apply(finalpreprocess)

    print(len(df_train))

    unlabeled_sentences = df_train[df_train['Labels'] != 4]['clean_text']
    df_train = df_train[df_train['Labels'] != 4]
    print(len(df_train))

    #  limiting the number of classes
    #  1 is real, 0 is fake
    df_train['Labels'] = df_train['Labels'].apply(lambda x: 1 if x in [3, 5] else 0)
    df_train.head()

    #  SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"], df_train["Labels"], test_size=0.2,
                                                        shuffle=True)

    # Word2Vec
    # Word2Vec runs on tokenized sentences
    X_train_tok = [nltk.word_tokenize(i) for i in X_train]
    X_test_tok = [nltk.word_tokenize(i) for i in X_test]

    unlabeled_sentences_tok = [nltk.word_tokenize(i) for i in unlabeled_sentences]

    # Tf-Idf
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

    model = gensim.models.Word2Vec(sentences=X_train_tok + unlabeled_sentences_tok, window=5, min_count=5, workers=4)
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
    df_train['clean_text_tok'] = [nltk.word_tokenize(i) for i in df_train['clean_text']]
    model = Word2Vec(df_train['clean_text_tok'], min_count=1)
    modelw = MeanEmbeddingVectorizer(w2v)


    with open('NLP/word2vec-model.pkl', 'wb') as f:
        pickle.dump(modelw, f)



    # converting text to numerical data using Word2Vec
    X_train_vectors_w2v = modelw.transform(X_train_tok)
    X_test_vectors_w2v = modelw.transform(X_test_tok)





    # FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
    lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    lr_tfidf.fit(X_train_vectors_tfidf, y_train)
    # Predict y value for test dataset
    y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
    y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]
    print(classification_report(y_test, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_test, y_predict))

    with open('NLP/lr_tfidf', 'wb') as f:
        pickle.dump(lr_tfidf, f)


    # FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
    lr_w2v = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    lr_w2v.fit(X_train_vectors_w2v, y_train)  # model
    # Predict y value for test dataset
    y_predict = lr_w2v.predict(X_test_vectors_w2v)
    y_prob = lr_w2v.predict_proba(X_test_vectors_w2v)[:, 1]
    print(classification_report(y_test, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_test, y_predict))

    with open('NLP/lr_w2v.pkl', 'wb') as f:
        pickle.dump(lr_w2v, f)


    # FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_vectors_tfidf, y_train)
    # Predict y value for test dataset
    y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
    y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]
    print(classification_report(y_test, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_test, y_predict))

    with open('NLP/nb_tfidf.pkl', 'wb') as f:
        pickle.dump(nb_tfidf, f)



    # Pre-processing the new dataset
    df_test['clean_text'] = df_test['Text'].apply(lambda x: finalpreprocess(x))  # preprocess the data
    X_test = df_test['clean_text']
    # converting words to numerical data using tf-idf
    X_vector = tfidf_vectorizer.transform(X_test)
    # use the best model to predict 'target' value for the new dataset
    y_predict = lr_tfidf.predict(X_vector)
    y_prob = lr_tfidf.predict_proba(X_vector)[:, 1]
    df_test['predict_prob'] = y_prob
    df_test['Labels'] = y_predict
    final = df_test[['clean_text', 'Labels']].reset_index(drop=True)
    print(final.head())


if __name__ == '__main__':
    generate_models()

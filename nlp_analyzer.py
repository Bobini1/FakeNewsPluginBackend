import pickle
import numpy as np
import pandas as pd
from NLP import full_preprocessing
import NLP

# NLP.generate_models()

model = pickle.load(open('NLP/lr_w2v.pkl', 'rb'))
word2vec_model = pickle.load(open('NLP/word2vec-model.pkl', 'rb'))


def is_real(text):
    try:
        text = full_preprocessing(word2vec_model, pd.Series(text))
    except:
        print(text)
        raise
    print(model.predict_proba(text))
    x = model.predict(text)[0]
    return x


if __name__ == '__main__':
    print(is_real('I am a real news'))
    print(is_real('I am a fake news'))

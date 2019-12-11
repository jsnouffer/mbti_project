from imported_MBTI import Classifier, PreProcessor
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import pickle

print("loading E")
with open("/mnt/volume_nyc3_01/models/E.p", "rb") as file:
    E = pickle.load(file)
print("loading S")
with open("/mnt/volume_nyc3_01/models/S.p", "rb") as file:
    S = pickle.load(file)
print("loading T")
with open("/mnt/volume_nyc3_01/models/T.p", "rb") as file:
    T = pickle.load(file)
print("loading J")
with open("/mnt/volume_nyc3_01/models/J.p", "rb") as file:
    J = pickle.load(file)

def more_magic(classifier):
    type_func = lambda x, y: y[1] if x[0]==[1] else y[0]
    mpred = []
    mpred_prob = []
    mpred.append(type_func(classifier.perform_magic(E), ['Introvert','Extrovert']))
    mpred.append(type_func(classifier.perform_magic(S), ['Intuitive','Sensing']))
    mpred.append(type_func(classifier.perform_magic(T), ['Feeling','Thinking']))
    mpred.append(type_func(classifier.perform_magic(J), ['Perceiving','Judging']))
    for m in [E, S, T, J]:
        mpred_prob.append(classifier.perform_more_magic(m).item())
    return dict(zip(mpred, mpred_prob))


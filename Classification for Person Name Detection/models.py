# models.py

import time
import numpy as np
from math import *
from utils import *
from collections import Counter
from nerdata import *
from optimizers import *
from typing import List
from itertools import *
import re


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts.get_count(tokens[idx]) > self.neg_counts.get_count(tokens[idx]):
            return 1
        else:
            return 0

def train_count_based_binary_classifier(ner_exs: List[PersonExample]) -> CountBasedPersonClassifier:
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for example in ner_exs:
        for idx in range(0, len(example)):
            if example.labels[idx] == 1:
                pos_counts.increment_count(example.tokens[idx], 1.0)
            else:
               neg_counts.increment_count(example.tokens[idx], 1.0)
    print("All counts: " + repr(pos_counts))
    #print("Count of Peter: " + repr(pos_counts["Peter"]))
    return CountBasedPersonClassifier(pos_counts, neg_counts)

# assign some pattern to word 
def get_pattern(word):
    pattern = ""
    for c in word:
        if c.isupper():
            pattern += "A"
        elif c.islower():
            pattern += "a"
        elif c.isdigit():
            pattern += "0"
        else:
            pattern += c
    summarized_pattern = "".join([i for i, g in groupby(pattern)])
    return pattern, summarized_pattern

class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """
    def __init__(self, weights, indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens, idx):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        is_quotes_present = False
        counts = {}
        for w in tokens:
            if w == '"':
                is_quotes_present = True
            if w in counts:
                counts[w]+=1
            else:
                counts[w]=1

        feature_positions = []
        word = tokens[idx]
        word_length = len(word)
        pattern, sum_pat = get_pattern(word)
        # pattern features
        if self.indexer.contains("sum_pat=" + sum_pat):
            feature_positions.append(self.indexer.index_of("sum_pat="+ sum_pat))
        # first word and last word features
        if idx == 0:
            feature_positions.append(self.indexer.index_of("first word"))
        if idx == len(tokens) - 1:
            feature_positions.append(self.indexer.index_of("last word"))
        # prev and next features
        if idx != 0:
            prev_pat, prev_sum_pat = get_pattern(tokens[idx-1])
            if self.indexer.contains("prev_sum_pat=" + prev_sum_pat):
                feature_positions.append(self.indexer.index_of("prev_sum_pat="+ prev_sum_pat))
            if self.indexer.contains("prev_word="+ tokens[idx-1]):
                feature_positions.append(self.indexer.index_of("prev_word="+ tokens[idx-1]))
            else:
                feature_positions.append(self.indexer.index_of("new prev_word"))
        if idx != len(tokens) - 1:
            next_pat, next_sum_pat = get_pattern(tokens[idx+1])
            if self.indexer.contains("next_sum_pat=" + next_sum_pat):
                feature_positions.append(self.indexer.index_of("next_sum_pat=" + next_sum_pat))
            if self.indexer.contains("next_word="+ tokens[idx+1]):
                feature_positions.append(self.indexer.index_of("next_word="+ tokens[idx+1]))
            else:
                feature_positions.append(self.indexer.index_of("new next_word"))
        if idx != 0 and idx != len(tokens) - 1: 
            if self.indexer.contains("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat):
                feature_positions.append(self.indexer.index_of("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat))
            if self.indexer.contains(self.indexer.index_of("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat)):
                feature_positions.append(self.indexer.index_of("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat))
        # Quotes features
        if is_quotes_present:
            feature_positions.append(self.indexer.index_of("quotes"))
        # length features
        if word_length <= 5:
            feature_positions.append(self.indexer.index_of("len<=5"))
        if word_length >= 6 and word_length <=10:
            feature_positions.append(self.indexer.index_of("len [6 to 10]"))
        if word_length >= 11 and word_length <=15:
            feature_positions.append(self.indexer.index_of("len [11 to 15]"))
        if word_length >= 16 and word_length <=20:
            feature_positions.append(self.indexer.index_of("len [16 to 20]"))
        if word_length > 20:
            feature_positions.append(self.indexer.index_of("len>20"))
        # current word and new word features
        if self.indexer.contains("len="+str(word_length)):
            feature_positions.append(self.indexer.index_of("len="+str(word_length)))
        if self.indexer.contains("current word=" + word):
            feature_positions.append(self.indexer.index_of("current word=" + word))
        else:
            feature_positions.append(self.indexer.index_of("new word"))
        # counts features
        if counts[word] == 1:
            feature_positions.append(self.indexer.index_of("word appeared once"))
        if counts[word] == 2:
            feature_positions.append(self.indexer.index_of("word appeared twice"))
        if counts[word] > 2:
            feature_positions.append(self.indexer.index_of("word appeared more than twice"))
        for f in feature_positions:
            if f == -1:
                raise Exception("Feature does not exist!")
        score = score_indexed_features(feature_positions, self.weights)
        e_score = exp(score)
        prob_name = e_score/(1+e_score)

        # print('feature_positions: ',feature_positions)
        # print('score: ',score)
        if prob_name >= 0.5:
            return 1
        else:
            return 0

    def train_classifier(ner_exs):
        features = Indexer()
        words = []
        labels = []
        total_exs= sum([len(example) for example in ner_exs])
        w = -1 
        feature_positions = {}
        for example in ner_exs:
            is_quotes_present = False
            counts = {}
            for tok in example.tokens:
                if tok == '"':
                    is_quotes_present = True
                if tok in counts:
                    counts[tok]+=1
                else:
                    counts[tok]=1
            for i in range(0,len(example)):
                w += 1
                word = example.tokens[i]
                words.append(word)
                labels.append(example.labels[i])
                word_length = len(word)
                features.get_index("new word")
                features.get_index("new prev_word")
                features.get_index("new next_word")
                feature_positions[w] = []
                pattern, sum_pat = get_pattern(word) 
                feature_positions[w].append(features.get_index("sum_pat="+sum_pat))
                if i != 0:
                    prev_pat, prev_sum_pat = get_pattern(example.tokens[i-1])
                    feature_positions[w].append(features.get_index("prev_sum_pat="+ prev_sum_pat))
                    feature_positions[w].append(features.get_index("prev_word="+ example.tokens[i-1]))
                else:
                    feature_positions[w].append(features.get_index("first word"))
                if i != len(example) - 1:
                    next_pat, next_sum_pat = get_pattern(example.tokens[i+1])
                    feature_positions[w].append(features.get_index("next_sum_pat=" + next_sum_pat))
                    feature_positions[w].append(features.get_index("next_word="+ example.tokens[i+1]))
                else:
                    feature_positions[w].append(features.get_index("last word"))
                if i != 0 and i != len(example) - 1: 
                    feature_positions[w].append(features.get_index("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat))
                    feature_positions[w].append(features.get_index("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat))
                if is_quotes_present:
                    feature_positions[w].append(features.get_index("quotes"))
                if word_length <= 5:
                    feature_positions[w].append(features.get_index("len<=5"))
                if word_length >=6 and word_length <= 10:
                    feature_positions[w].append(features.get_index("len [6 to 10]"))
                if word_length >= 11 and word_length <=15:
                    feature_positions[w].append(features.get_index("len [11 to 15]"))
                if word_length >= 16 and word_length <=20:
                    feature_positions[w].append(features.get_index("len [16 to 20]"))
                if word_length > 20:
                    feature_positions[w].append(features.get_index("len>20"))
                feature_positions[w].append(features.get_index("len="+str(word_length)))
                feature_positions[w].append(features.get_index("current word=" + word))
                if counts[word] == 1:
                    feature_positions[w].append(features.get_index("word appeared once"))
                if counts[word] == 2:
                    feature_positions[w].append(features.get_index("word appeared twice"))
                if counts[word] > 2:
                    feature_positions[w].append(features.get_index("word appeared more than twice"))
                
        num_features = len(features)
        weights = np.random.randn(num_features) * 0.001
        f1_train = []   
        f1_dev = []   
        alpha = 0.1
        grad_ascent = SGDOptimizer(weights,alpha)
        # train by 60 epochs 
        num_epochs = 60
        for epoch_num in range(1, num_epochs+1):
            indices = np.random.choice(total_exs, total_exs)
            indices = np.arange(total_exs)
            for w in indices:
                score = grad_ascent.score(feature_positions[w])
                e_score = exp(score)
                sigmoid = e_score/(1+e_score)
                label_minus_prob = labels[w] - sigmoid # should be close to 0   
                gradient = Counter()
                for feat_pos in feature_positions[w]:
                    gradient.set_count(feat_pos, label_minus_prob)
                grad_ascent.apply_gradient_update(gradient,1)
            print(epoch_num)
        return PersonClassifier(grad_ascent.weights, features)     

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.learning_curve import learning_curve
import numpy as np
import matplotlib.pyplot as plt



class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()
        self.word_freq = {}
        self.bi_freq = {}
        self.tri_freq = {}

    def getWordFrequency(self, word):
        if word in self.word_freq:
            self.word_freq[word] += 1
        else:
            self.word_freq[word] = 1

            

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        no_vowels = sum(map(word.count, "aeiou"))
        no_const = len(word) - no_vowels
        proportion = no_const / (no_vowels+1)
        if (no_vowels > 2):
            multiple_vowels = 1
        else:
            multiple_vowels = 0
        return [self.word_freq[word], len_chars, multiple_vowels]

    def train(self, trainset):
        X = []
        y = []
        ccount = 0
        dcount = 0
        for sent in trainset:
            self.getWordFrequency(sent['target_word'])
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, n_jobs=-1, cv=None, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure()
        plt.title("Logistic Regression Classifier")
        plt.legend(loc="best")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()

        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.ylim(-.1,1.1)
        plt.show()

    def test(self, testset):
        X = []
        for sent in testset:
            self.getWordFrequency(sent['target_word'])
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)

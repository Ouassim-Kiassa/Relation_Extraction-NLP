import csv
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk.metrics import precision,recall,ConfusionMatrix
import collections
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd


def read_docs_from_csv(filename):
    docs = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for text, label in tqdm(reader):
            words = nltk.word_tokenize(text)
            docs.append((words, label))

    return docs


def split_train_dev_test(docs, train_ratio=0.8, dev_ratio=0.1):
    np.random.seed(2022)
    np.random.shuffle(docs)
    train_size = int(len(docs) * train_ratio)
    dev_size = int(len(docs) * dev_ratio)
    return (
        docs[:train_size],
        docs[train_size : train_size + dev_size],
        docs[train_size + dev_size :],
    )

def split_train_test_sklearn(df):
    return train_test_split(df['sentence'],df['target'],test_size=0.2,random_state=1024)


def calculate_tp_fp_fn(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp += 1
        else:
            if true == "positive":
                fn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, fscore


def lemmatize_dataset(df):
    stop_words = list(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x, language='english') if re.match('\w', word)]))
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))
        
    return df

def prepare_dataset(in_path):
    df = pd.read_csv(in_path, delimiter=',')
    #drop columns we do not need and
    #Transform boolean is_cause and is_treat to label variable with 3 labels ("cause","treat" and "neutral")
    df['target']= 'neutral'
    df.loc[df[df.is_cause == 1.0].index.values, 'target'] = 'cause'
    df.loc[df[df.is_treat == 1.0].index.values, 'target'] = 'treat'
    
    df = df.loc[:, ['sentence', 'target']]

    return df


def transform_to_dictionary(df):
    docs = []
    for index,row in df.iterrows():
        words = row['sentence'] #tokenizing is done in the next step
        docs.append((words, row['target']))

    all_words = set(word.lower() for sentence in docs for word in word_tokenize(sentence[0]))
    t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in docs]
    return t


def train_NaiveBayes(df):
    return NaiveBayesClassifier.train(df)

def show_features(model):
    return model.show_most_informative_features(30)

def evaluate_nltk_nb(model,testdata):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    labels = []
    tests = []
    for i, (text, label) in enumerate(testdata):
        refsets[label].add(i)
        observed = model.classify(text)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)

    print('Overall Accuracy:', nltk.classify.accuracy(model, testdata))
    print('Cause precision:', nltk.precision(refsets['cause'], testsets['cause']))
    print('Cause recall:', nltk.recall(refsets['cause'], testsets['cause']))
    print('Cause F-measure:', nltk.f_measure(refsets['cause'], testsets['cause']))
    print('Treat precision:', nltk.precision(refsets['treat'], testsets['treat']))
    print('Treat recall:', nltk.recall(refsets['treat'], testsets['treat']))
    print('Treat F-measure:', nltk.f_measure(refsets['treat'], testsets['treat']))
    print('Neutral precision:', nltk.precision(refsets['neutral'], testsets['neutral']))
    print('Neutral recall:', nltk.recall(refsets['neutral'], testsets['neutral']))
    print('Neutral F-measure:', nltk.f_measure(refsets['neutral'], testsets['neutral']))
    print(nltk.ConfusionMatrix(labels,tests))

def test_prediction_for_new_sentence(df,model,text):
    docs = []
    for index,row in df.iterrows():
        words = row['sentence'] #tokenizing is done in the next step
        docs.append((words, row['target']))

    all_words = set(word.lower() for sentence in docs for word in word_tokenize(sentence[0]))
    doc1={word: (word in nltk.word_tokenize(text.lower())) for word in all_words}
    return model.classify(doc1)


def train_multi_nb_classifier(X_train,y_train):
    multi_nb_classifier = Pipeline([('v', CountVectorizer()), ('tfidf', TfidfTransformer()), ('mnb', MultinomialNB())])
    return(multi_nb_classifier.fit(X_train, y_train))

def train_sgd_classifier(X_train, y_train):
    sgd_classifier = Pipeline([('v', CountVectorizer()), ('svm', SGDClassifier(random_state=1024))])
    sgd_classifier.fit(X_train, y_train)
    return sgd_classifier

def train_sgd_classifier_with_grams(X_train, y_train, n_gram_range=None):
    t = TfidfVectorizer(ngram_range=n_gram_range)
    X_train = t.fit_transform(X_train)

    sgd_classifier = SGDClassifier(random_state=1024)

    return t, (sgd_classifier.fit(X_train, y_train))

def undersampling_dataset(df):
    n = 150
    msk = df.groupby('target')['target'].transform('size') >= n
    df = pd.concat((df[msk].groupby('target').sample(n=n), df[~msk]), ignore_index=True)
    return df


def add_relative_counts(df_main, y_train, y_test, targets, sets):
    df_main['counts_rel'] = 0

    total_counts = {}
    for s in sets:
        total_counts[s] = {}

    for target, value in zip(y_train.value_counts().keys(), y_train.value_counts().values):
        total_counts['train'][target] = value

    for target, value in zip(y_test.value_counts().keys(), y_test.value_counts().values):
        total_counts['missclassified'][target] = value

    for target, value in zip(y_test.value_counts().keys(), y_test.value_counts().values):
        total_counts['classified correct'][target] = value

    for s in sets:
        for target in targets:
            df_main.loc[(df_main.target == target) & (df_main.set == s), 'counts_rel'] = df_main.loc[(df_main.target == target) & (df_main.set == s), 'counts'] / total_counts[s][target]

    return df_main.copy()


def get_word_counts(X_train, y_train, X_test, y_test, y_pred, targets, sets):
    words_set = {}
    sets = ['classified correct', 'missclassified', 'train']
    targets = ['treat', 'cause', 'neutral']

    for s in sets:
        words_set[s] = {}
        for target in targets:
            words_set[s][target] = {}

    def add_word_counts_from_sentence(sentence, words_set, prediction_type, true):
        for word in sentence.split(' '):
            word_count = words_set[prediction_type][true].get(word, 0)
            words_set[prediction_type][true][word] = word_count + 1

    for sentence, true, predict in zip(X_test, y_test, y_pred):
        prediction_type = 'classified correct' if true == predict else 'missclassified'
        add_word_counts_from_sentence(sentence, words_set, prediction_type, true)

    for sentence, true in zip(X_train, y_train):
        prediction_type = 'train'
        add_word_counts_from_sentence(sentence, words_set, prediction_type, true)

    dfs = []
    for set_type in sets:
        for class_type in targets:
            df = pd.DataFrame({'words': words_set[set_type][class_type].keys(), 'counts': words_set[set_type][class_type].values()})
            df['target'] = class_type
            df['set'] = set_type
            dfs.append(df)

    df_main = pd.concat(dfs)
    df_main = add_relative_counts(df_main, y_train, y_test, targets, sets)
    
    return df_main



def plot_word_counts_by_target(df_main, current_target, n=15):
    _, axes = plt.subplots(1, 3, figsize=(25, 6))
    sns.despine()

    axes = axes.flatten()

    plot_types = ['largest-set', 'largest-target', 'missclassified']

    for ax, plot_type in zip(axes, plot_types):
        print(plot_type)
        current_set = 'train'
        df_current = df_main[(df_main.set == current_set) & (df_main.target == current_target)]

        if plot_type == 'largest-set':
            df_most_popular = df_current.nlargest(n, 'counts', keep='first')
            sns.barplot(data=df_main[(df_main.target == current_target) & (df_main.words.isin(df_most_popular.words))], y='words', x='counts_rel', ax=ax, hue='set')
            
            ax.set_title(f'Top {n} words in {current_target} train dataset for {current_target} class')
        elif plot_type == 'largest-target':
            df_most_popular = df_current.nlargest(n, 'counts', keep='first')
            sns.barplot(data=df_main[(df_main.set == current_set) & (df_main.words.isin(df_most_popular.words))], y='words', x='counts_rel', ax=ax, hue='target')
            
            ax.set_title(f'Top {n} words in {current_set} dataset by {current_target} class')
        else:
            current_set = 'missclassified'
            df_current = df_main[(df_main.set == current_set) & (df_main.target == current_target)]
            df_most_popular = df_current.nlargest(n, 'counts', keep='first')

            sns.barplot(data=df_main[(df_main.set == 'train') & (df_main.words.isin(df_most_popular.words))], y='words', x='counts_rel', ax=ax, hue='target')

            ax.set_title(f'Top {n} words in misclassified part of dataset for {current_target} class in train dataset')


def plot_misclassified_sentences(df_main, sentence_mscl, sentence_normal, current_target, n=15):
    _, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.despine()

    axes = axes.flatten()

    plot_types = ['missclassified', 'classified correct']
    sentences = [sentence_mscl, sentence_normal]

    for ax, plot_type, sentence in zip(axes, plot_types, sentences):
        print(sentence)
        df_current = df_main[(df_main.words.isin(sentence.split(' '))) & (df_main.set == 'train')]
        sns.barplot(data=df_current, y='words', x='counts_rel', hue='target', ax=ax)
        ax.set_title(f'{plot_type} sentence, {current_target} class, probabilities of word occurence across classes')

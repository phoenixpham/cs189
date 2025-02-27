'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb
import string
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def capitalization_ratio_feature(text, freq):
    return sum(1 for char in text if char.isupper()) / (len(text) + 1e-6)

def special_char_count_feature(text, freq):
    return sum(1 for char in text if char in string.punctuation)

def price_feature(text, freq):
    return len(re.findall(r'\$\d+(?:\.\d{2})?', text))

def spam_word_count_feature(text, freq):
    spammy_words = ['free', 'win', 'prize', 'click', 'offer', 'limited', 'guaranteed']
    return sum(text.lower().count(word) for word in spammy_words)

# Generates a feature vector
def generate_feature_vector(text, freq, vocabulary):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    
    feature.append(capitalization_ratio_feature(text, freq))
    feature.append(special_char_count_feature(text, freq))
    feature.append(price_feature(text, freq))
    feature.append(spam_word_count_feature(text, freq))
    
    # Bag-Of-Words
    bow_vector = np.zeros(len(vocabulary))
    for word in freq:
        if word in vocabulary:
            bow_vector[vocabulary[word]] += freq[word]
    feature.extend(bow_vector)

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames, vocabulary):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq, vocabulary)
            design_matrix.append(feature_vector)
    return design_matrix

def build_vocabulary(filenames, vocab_size=1000):
    word_freq = defaultdict(int)
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_freq[word] += 1

    sorted_words = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
    vocabulary = {word: idx for idx, word in enumerate(sorted_words[:vocab_size])}
    return vocabulary

# Extract TF-IDF features from text files
def extract_tfidf_features(filenames):
    raw_texts = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read()
                text = text.replace('\r\n', ' ')
                raw_texts.append(text)
            except Exception as e:
                # skip files we have trouble reading.
                continue
    return raw_texts 

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]

vocabulary = build_vocabulary(spam_filenames + ham_filenames)

spam_texts = extract_tfidf_features(spam_filenames)
ham_texts = extract_tfidf_features(ham_filenames)
test_texts = extract_tfidf_features(test_filenames)

tfidf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.92, ngram_range=(1, 2), stop_words='english', vocabulary=vocabulary.keys())

all_train_texts = spam_texts + ham_texts

tfidf_features_train = tfidf.fit_transform(all_train_texts) # fit and transform training data
tfidf_features_test = tfidf.transform(test_texts) # only transform test data

spam_design_matrix = generate_design_matrix(spam_filenames, vocabulary)
ham_design_matrix = generate_design_matrix(ham_filenames, vocabulary)
test_design_matrix = generate_design_matrix(test_filenames, vocabulary)

original_features_train = np.array(spam_design_matrix + ham_design_matrix)
original_features_test = np.array(test_design_matrix)

X = np.hstack((original_features_train, tfidf_features_train.toarray()))
test_data = np.hstack((original_features_test, tfidf_features_test.toarray()))

Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data-hw3.npz', training_data=X, training_labels=Y, test_data=test_data)
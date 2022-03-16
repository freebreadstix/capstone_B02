import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# nltk.download('stopwords')
# nltk.download('punkt')


def load_data(filename):
    df = pd.read_csv(filename)

    df['verdict'] = df['label'].apply(lambda verdict: 0 if verdict == 'fake' else 1)

    print("Shape of the dataset: {}".format(df.shape))
    print("Number of the 'REAL' label in the dataset: {}".format(len(df[df['verdict'] == 1])))
    print("Number of the 'FAKE' label in the dataset: {}".format(len(df[df['verdict'] == 0])))

    return df
"""
Generates a word cloud from the words in the json file given by the variable (words) which is a path to a json
the path variable is where we want to save the word cloud to and what name we want
"""
def important_cloud(words,path):
    f = open(words)
    most_important_words = json.load(f)
    wc = WordCloud()
    wc.generate_from_frequencies(most_important_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(path+".png")
    plt.show()
#makes intersection of sets, select two topics, what preprocessing, and what model
def topicintersection (topic1, topic2, preproc, model, model2):
    f = open("results/"+ topic1 + "_" + model + "_"  + preproc + ".json")
    set1 = set(json.load(f).keys())
    f2 = open("results/"+ topic2 + "_" + model2 + "_"  + preproc + ".json")
    set2 = set(json.load(f2).keys()) 
    pct_intersect = len(set1.intersection(set2))/len(set2)
    return pct_intersect

"""
Runs intersections for every single topic and model combination with every other model and topic combination and puts
them into a two layer dictionary
"""
def intersect_all():
    intersections = {}
    for topic in ["general","politics", "economics", "science"]:
        intersections[topic] = {}
        for topic2 in ["general","politics", "economics", "science"]:
            if topic == topic2:
                pass
            else:
                for model in ["logistic-regression", "svm", "decision-tree", "random-forest"]:
                    for model2 in ["logistic-regression", "svm", "decision-tree", "random-forest"]:
                        pct_intersect = topicintersection(topic, topic2, 'tfidf', model, model2)
                        intersections[topic][topic2 + ", tfidf, " + model + ", "+ model2] = pct_intersect
    return intersections


def get_vocab_size(X_train_preproc, X_test_preproc, X_val_preproc):
    text_data = []
    vocab = []

    for i in X_train_preproc:
        text_data.append(i)
    for i in X_test_preproc:
        text_data.append(i)
    for i in X_val_preproc:
        text_data.append(i)

    for text in text_data:
        text_list = text.split(' ')
        for word in text_list:
            vocab.append(word)
    vocab_size = len(set(vocab))

    num_token = [len(tokens.split(' ')) for tokens in X_train_preproc + X_test_preproc + X_val_preproc]
    num_token = np.array(num_token)
    max_token = np.mean(num_token) + 2 * np.std(num_token)
    max_token = int(max_token)

    return vocab_size, max_token
"""
Plots a decision tree and saves it to the results/Figures Folder
PKL is the path to the pkl file with the dtree
path is the path we want to save to and under what name
width is how wide we want the figure to be
height is how tall we want the figure to be
"""
def plot_dtree(pkl, path, width, height ):
    dtree = pickle.load(open(pkl, "rb"))
    fig = plt.figure(figsize = (width,height))
    _ = tree.plot_tree(clf, filled = True)
    fig1= plt.gcf()
    fig1.savefig(path+ ".png")
def create_token2id(X_train_preproc, X_test_preproc, X_val_preproc):
    tokenized_docs = [word_tokenize(text) for text in X_train_preproc + X_test_preproc + X_val_preproc]
    dictionary = Dictionary(tokenized_docs)
    return dictionary.token2id


def one_hot_text(token2id, input_text):
    return [token2id[word] for word in word_tokenize(input_text)]


def preprocessing(text):
	negate_dict = {
    "couldn't": "could not",
	"can't": "can not",
    "didn't": "did not",
    "won't": "will not",
    "don't": "do not",
    "aren't": "are not",
    "doesn't": "does not",
    "hadn't": "had not", 
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not"
    }
	stemmer = PorterStemmer()
	stopwords_english = stopwords.words('english')
	stopwords_english.remove('not')
	stopwords_english.remove('no')
	text = re.sub(r'$', '', str(text))
	text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
	text = re.sub(r'^RT[\s]+', '', str(text))
	text = re.sub(r'#', '', str(text))
	text = re.sub(r'\@\w*', '', str(text))
	text = re.sub(r'WHO', 'world health organization', str(text))
	for negate_word in negate_dict.keys():
		text = re.sub(negate_word, negate_dict[negate_word], str(text))
	text = re.sub(r"&", ' and ', str(text))
	text = text.replace('&amp', ' ')
	text = re.sub(r"[^0-9a-zA-Z]+", ' ', str(text))
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
	text_tokens = tokenizer.tokenize(text)
	clean_text = []
	for word in text_tokens:
		if (word not in stopwords_english and word not in string.punctuation):
			stem_word = stemmer.stem(word)
			clean_text.append(stem_word)
	return ' '.join(clean_text)


def confusion_matrix_plot(matrix):
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]
    labels = [f'{a}\n{b}' for a, b in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(matrix, annot=labels, fmt='', cmap=sns.light_palette("seagreen", as_cmap=True))


def evaluate(y_test, prediction):
    print(classification_report(y_test, prediction))

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print('Accuracy score: {}'.format(accuracy))
    print('Precision score: {}'.format(precision))
    print('Recall score: {}'.format(recall))

    return accuracy, precision, recall


def get_verdict(model, input_text, vocab_size, maxlen):
    input_text_preproc = preprocessing(input_text)
    input_text_onehot = one_hot(input_text_preproc, vocab_size)
    input_text_embedded_docs = pad_sequences([input_text_onehot], padding='pre', maxlen=maxlen)
    output_verdict = model.predict(input_text_embedded_docs)
    return output_verdict[0][0]


def get_verdict_with_token2id(model, token2id, input_text, maxlen):
    input_text_preproc = preprocessing(input_text)
    input_text_onehot = one_hot_text(token2id, input_text_preproc)
    input_text_embedded_docs = pad_sequences([input_text_onehot], padding='pre', maxlen=maxlen)
    output_verdict = model.predict(input_text_embedded_docs)
    return output_verdict[0][0]
	

	
	
	


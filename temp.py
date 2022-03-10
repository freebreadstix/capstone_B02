import json
import os
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.old_utils import evaluate
from src.preprocessing.preprocessing import get_Xy_from_sheet, spacy_preprocess_texts

model_save_path = './models/'
result_save_path = './results/'

preproc = 'bow'

data_path = "data/processed/joined_data.csv"

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create DataFrame and new column for analysis
X, y = get_Xy_from_sheet(data_path, X_col='Original article text', y_col='Verdict')

# x_transform, nlp = spacy_preprocess_texts(X, 'test1', 'md')

# vectors = [nlp.vocab.vectors.key2row[word.norm] if word.has_vector and word.norm in nlp.vocab.vectors.key2row else None for word in x_transform[0]]
# print(vectors)
# print(nlp.vocab.vectors.key2row)

# Preprocessing TODO: 
# n-gram

# spacy preprocessing: lemma, stem, etc
# Remove punct, nums

# Find way to incorporate sentence structure
# Include parts of speech


if config['preprocessing']['type'] == 'tfidf':
    tfidf = TfidfVectorizer(ngram_range=(config['preprocessing']['ngram_min'], config['preprocessing']['ngram_max']))
    X_transform = tfidf.fit_transform(X)

    idx2word = {idx: word for word, idx in tfidf.vocabulary_.items()}

else:
    cv = CountVectorizer(ngram_range=(config['preprocessing']['ngram_min'], config['preprocessing']['ngram_max']))
    X_transform = cv.fit_transform(X)

    idx2word = {idx: word for word, idx in cv.vocabulary_.items()}

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, shuffle=True, random_state=42)


models = {
    'logistic-regression': LogisticRegression(),
    'svm': SVC(kernel='linear'),
    'decision-tree': DecisionTreeClassifier(),
    'random-forest': RandomForestClassifier(),
}

# LogReg, Decision Trees, Random Forests, SVMs

dataset_name = data_path.split('.')[0].split('/')[-1]

for model_name, model in models.items():
    fname = dataset_name + '_' + model_name
    pklname = model_save_path + fname + '.pkl'

    if pklname not in os.listdir(model_save_path):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        pickle.dump(model, open(pklname, 'wb'))

    else:
        model = pickle.load(open(pklname, 'rb'))


    acc, prec, recall = evaluate(y_test, preds)

    if model_name in ['logistic-regression', 'svm']:
        coefs = model.coef_[0]
        if model_name == 'svm':
            coefs = coefs.toarray().reshape((-1,))
        

    elif model_name in ['decision-tree', 'random-forest']:
        coefs = model.feature_importances_

    coefs = [(coef, i) for i, coef in enumerate(coefs)]
    coefs.sort(key=lambda x: x[0] ** 2, reverse=True)

    num_words = config['output']['num_words']
    most_important_words = {idx2word[idx]: coef for coef, idx in coefs[:num_words]}

    if config['output']['print']: print(f'Most important words for {model_name}: {most_important_words}')

    resultname = result_save_path + fname + '_' + preproc + '.json'
    with open(resultname, 'w') as f:
        json.dump(most_important_words, f)


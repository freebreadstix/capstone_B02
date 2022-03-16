import json
import os
import pickle
import sys
import yaml
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.old_utils import evaluate, important_cloud, topicintersection, intersect_all, plot_dtree
from src.preprocessing.preprocessing import get_Xy_from_sheet, spacy_preprocess_texts


def main(targets):
    DT = DecisionTreeClassifier()
    with open(targets[0], 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data_path']

    preproc = config['preprocessing']['type']

    model_save_path = config['model_save_path']
    result_save_path = config['result_save_path']


    # Load, preprocess data
    X, y = get_Xy_from_sheet(data_path, X_col='Original article text', y_col='Verdict')

    if preproc == 'tfidf':
        tfidf = TfidfVectorizer(ngram_range=(config['preprocessing']['ngram_min'], config['preprocessing']['ngram_max']))
        X_transform = tfidf.fit_transform(X)

        idx2word = {idx: word for word, idx in tfidf.vocabulary_.items()}

    else:
        cv = CountVectorizer(ngram_range=(config['preprocessing']['ngram_min'], config['preprocessing']['ngram_max']))
        X_transform = cv.fit_transform(X)

        idx2word = {idx: word for word, idx in cv.vocabulary_.items()}

    X_train, X_test, y_train, y_test = train_test_split(X_transform, y, shuffle=True, random_state=42)

    # Run models
    models = {
        'logistic-regression': LogisticRegression(),
        'svm': SVC(kernel='linear'),
        'decision-tree': DecisionTreeClassifier(),
        'random-forest': RandomForestClassifier(),
    }

    dataset_name = data_path.split('.')[-2].split('/')[-1]
    results_dict = {}

    for model_name, model in models.items():
        fname = dataset_name + '_' + model_name
        pklname = model_save_path + fname + '.pkl'
        if pklname not in os.listdir(model_save_path):
            model.fit(X_train, y_train)
            
            pickle.dump(model, open(pklname, 'wb'))

        else:
            model = pickle.load(open(pklname, 'rb'))

        preds = model.predict(X_test)

        if config['output']['save_predictions']:
            result_path = result_save_path + f'{fname}_preds.csv'
            np.savetxt(result_path, preds, delimiter=",")

        acc, prec, recall = evaluate(y_test, preds)

        results_dict[model_name] = {}
        results_dict[model_name]['acc'] = str(acc)
        results_dict[model_name]['prec'] = str(prec)
        results_dict[model_name]['recall'] = str(recall)

        results_dict[model_name]['confusion_matrix'] = str(confusion_matrix(y_test, preds))
        

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

        if config['output']['print_words']: print(f'Most important words for {model_name}: {most_important_words}')

        resultname = result_save_path + fname + '_' + preproc + '.json'
        with open(resultname, 'w') as f:
            json.dump(most_important_words, f)
        if config["output"]["wordcloud"]:
            important_cloud(resultname, result_save_path + "Figures/"+ fname+"cloud")
        if config["output"]["decision_tree_model"] and type(model) == type(DT):
            plot_dtree(pklname, result_save_path + "Figures/" +fname+"_plotted", 40, 40)
            

    with open(result_save_path + dataset_name + '_classification_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    if config['output']['print_results']:
        print(results_dict)
    if config["output"]["wordcloud"]:
        important_cloud(resultname, result_save_path + "Figures/"+ fname+"_cloud")
        #Requires important words for eachmodel
    if config["output"]["intersections"]:
        intersect_dict = intersect_all()
        with open(result_save_path + "_intersections.json", "w") as f:
            json.dump(intersect_dict,f)
        


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
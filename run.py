import json
import numpy as np
import pickle
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from src.preprocessing.preprocessing import get_Xy_from_sheet
from src.preprocessing.old_utils import evaluate

def main(targets):
    if targets[0] == "test":
        data_path = "./test/testdata/cleaned data - Sheet1.csv"
        results_prefix = './test/results/'
    else:
        data_path = targets[0]
        results_prefix = './results/'

    # Create DataFrame and new column for analysis
    X, y = get_Xy_from_sheet(data_path)

    # TF-IDF Count vectorizer trained on original study data
    # cv = pickle.load(open('models/vector.pickle', 'rb'))
    # x_transform = cv.transform(x_preproc)
    if targets[1] == "spacy":
        x_transform = spacy_preprocess_texts(X, 'test1')
        

    else:
        cv = CountVectorizer()
        x_transform = cv.fit_transform(X)

    results_dict = {}

    models = ['naive-bayes', 'logistic-regression', 'svm', 'decision-tree', 'random-forest', 'stacking']
    model_path_prefixes = [
        'models/naive_bayes_model', 
        'models/logistic_regression_model', 
        'models/svm_model', 
        'models/decision_tree_model', 
        'models/random_forest_model', 
        'models/stacking_model']

    extension = '.pkl'

    import os

    arr = os.listdir('./models/')
    print(arr)

    for name, prefix in zip(models, model_path_prefixes):

        for model_path_prefix in [prefix, prefix + '_general']:
            model_path = f'./{model_path_prefix}{extension}'
            try:
                model = pickle.load(open(model_path, 'rb'))
            except:
                print(f"Can't locate model {name} at {model_path}")
                continue

            print('h')
            pred = model.predict(x_transform)
            result_path = results_prefix + f'pred/{model_path_prefix}.csv'
            np.savetxt(result_path, pred, delimiter=",")

            results_dict[model_path] = {}
            
            acc, prec, recall = evaluate(y, pred)  
            results_dict[model_path]['acc'] = str(acc)
            results_dict[model_path]['prec'] = str(prec)
            results_dict[model_path]['recall'] = str(recall)

            results_dict[model_path]['confusion_matrix'] = str(confusion_matrix(y, pred))
            


    with open(results_prefix + 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    print(results_dict)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
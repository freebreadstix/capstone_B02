import pandas as pd
import spacy

def get_Xy_from_sheet(file, X_col="Original article text", y_col="Verdict", true_val="TRUE", false_val="FALSE"):
    df = pd.read_csv(file)
    simple_df = df[[X_col, y_col]]
    X = df[X_col].values
    y = df[y_col].replace(true_val,1).replace(false_val,0).values
    return X, y


def spacy_preprocess_texts(texts, size='sm', serialize_path='', disable=[]):
    '''
    serialize_path (str): path to serialize spacy nlp object
    size (str): suffix for spacy model size, options are "sm", "md", "lg"
    '''
    nlp = spacy.load(f'en_core_web_{size}')

    preprocessed = []
    for doc in nlp.pipe(texts, disable=disable):
        preprocessed.append(doc)

    if serialize_path:
        nlp.to_disk(serialize_path)

    return preprocessed, nlp

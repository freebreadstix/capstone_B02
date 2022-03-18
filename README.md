# Group B02 Capstone project repo

To run:


If running from DSMLP cluster:
```
ssh user@dsmlp-login.ucsd.edu
launch-scipy-ml.sh -i freebreadstix/q1-replication
```
Else be sure to run in container: https://hub.docker.com/repository/docker/freebreadstix/q1-replication

Then:
```
git clone https://github.com/freebreadstix/capstone_B02.git
cd capstone_B02
```
If not merged to main, make sure to switch to branch with run.py
```
git checkout lucas-runpy
```

Configure config yaml with appropriate parameters. You can make your own .yml using config.yml as reference, just pass it as the argument on CLI

Run run.py w/ config yaml corresponding to configuration you are running. For testing this is test_config.yml
```
python3 run.py test_config.yml
```
Link to Presentation Website
```
https://micmiccitymax.github.io/dsc180b02-site/
```

Explainations of Config.yml output options
```
num_words: how many words are in the "important words" for the models
save_predictions: saves output of predictions to a file
print_results: prints results of evaluations to terminal
print words: Prints important words of each model in terminal
intersections: computes the important words similarity of all combinations of model and topics, USE ONLY WHEN YOU HAVE ALL MODELS MADE
decision_tree_model: outputs a plotting of decision tree to a figure
wordcloud: outputs an important word wordcloud to a figure in the figures folder
```

**Note**: if you are using intersections, decision_tree_model, or wordcloud options, make sure data is saved as 'data/processed/general.csv' and has columns 'Original Article Text' as the document text, 'Verdict' as 'TRUE' or 'FALSE', 'Category' as category, or change code within old_utils.py

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

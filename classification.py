import sklearn
import sklearn.datasets
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics


def manual_bayes(train):
    # vectorisation
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train.data)

    # tfidf: Term Frequency times Inverse Document Frequency
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # training
    clf = MultinomialNB().fit(X_train_tfidf, train.target)

    # testing
    docs_new = ['Super ce film !', "J'ai perdu mon temps...", "waou génial !", "bof bof"]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, train.target_names[category]))


def pipeline_bayes(train):
    """
    Classificateur naïf de Bayes.
    """
    return Pipeline([
        ('vect', CountVectorizer()), # vectorisation
        ('tfidf', TfidfTransformer()), # indexation
        ('clf', MultinomialNB()) # classification
    ]).fit(train.data, train.target) # entrainement


def pipeline_svm(train):
    """
    Classificateur SGD (régression linéaire) avec paramètres par défauts.
    """
    return Pipeline([
        ('vect', CountVectorizer()), # vectorisation
        ('tfidf', TfidfTransformer()), # indexation
        ('clf', SGDClassifier()) # classification
    ]).fit(train.data, train.target) # entrainement


def grid_search_svm(train):
    """
    Customisation des paramètres d'entrées du classificateur SGD (stochastic gradient descent)
    Plusieurs combinaisons de paramètres sont testées et la meilleure est retenue
    """
    text_clf = pipeline_svm(train)

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }

    grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1)
    grid_search = grid_search.fit(train.data, train.target)
    # Note: ces paramètres sont moins bons que les paramètres par défaut
    print("best parameters: ", grid_search.best_params_)
    return grid_search


def calssify(folder):
    categories = ['pos', 'neg']

    config = {
        'description':None,
        'categories':categories,
        'load_content':True,
        'shuffle':True,
        'encoding':'utf-8',
        'decode_error':'strict',
        'random_state':42
    }

    # chargement de données d'entrainement
    train = sklearn.datasets.load_files(folder + "/train", **config)

    # chargement de données de test
    test = sklearn.datasets.load_files(folder + "/test", **config)

    print("******\nVectorisation, indexation et test avec des données simples")
    manual_bayes(train)

    # Classifie les données de test selon 3 classificateurs
    for method in (pipeline_bayes, pipeline_svm, grid_search_svm):
        print("******\n" + method.__doc__)
        text_clf = method(train)
        predicted = text_clf.predict(test.data)
        print("Precision: {} %".format(np.mean(predicted == test.target)*100))
        print(metrics.classification_report(
            test.target, predicted,
            target_names = test.target_names))



if __name__ == '__main__':
    try:
        folder = sys.argv[1]
    except:
        folder = 'tagged_prepared'

    calssify(folder)

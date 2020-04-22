import pandas as pd
import spacy
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pickle

nlp = spacy.load('en_core_web_sm')
model_dict = {'svm_linear': SVC(gamma='auto', C=10, kernel='linear'),
              'logistic_regression': LogisticRegression(C=10, solver='lbfgs', max_iter=1000, multi_class="multinomial"),
               'gaussian_nb': GaussianNB(),
               'random_forest': RandomForestClassifier(n_estimators=100, max_depth = None, max_features='sqrt'),
               'decision_tree': DecisionTreeClassifier(random_state=0)}


def evaluate_model(classifier, testing_data, testing_labels):
    predictions = classifier.predict(testing_data)
    f1 = f1_score(testing_labels, predictions, average='weighted')
    print('F1 score: ', f1)
    # Accuracy
    score = classifier.score(testing_data, testing_labels)
    print('Accuracy: ', score)


def get_model(model_name: str):
    return model_dict[model_name]


def train_model(model_name: str, vector: str, features: list, training_data: pd.Series, training_labels: pd.Series):
    print('\nModel name: ', model_name)

    if os.path.isfile('Data/trained_models/' + model_name + '_with_' + '_'.join([vector] + features) + '.pckl'):
        return pd.read_pickle('Data/trained_models/' + model_name + '_with_' + '_'.join([vector] + features) + '.pckl')

    model = get_model(model_name)
    model.fit(training_data, training_labels)

    with open('Data/trained_models/' + model_name + '_with_' + '_'.join([vector] + features) + '.pckl', 'wb') as f:
        pickle.dump(model, f)

    return model


def train_and_evaluate(data_tuple: tuple):
    training_data, testing_data, training_labels, testing_labels = data_tuple
    for classifier_name in model_dict.keys():
        for vector_name in ['bag_of_words', 'tf_idf']:
            for feature_names in [[], ['topic']]:
                classifier = train_model(classifier_name, vector_name, feature_names, training_data, training_labels)
                evaluate_model(classifier, testing_data, testing_labels)


if __name__ == '__main__':

    # ------ MODEL SETTINGS ------
    vector_type = 'bag_of_words'
    model_name = 'logistic_regression'
    features = []
    split = 0.2

    # ---- CONFIGURE PATHS -------
    path_to_vectors = 'Data/vectors/'
    path_to_raw_data = 'Data/processed_data/RawData.csv'
    path_to_clean_data = 'Data/processed_data/CleanData.csv'

    # -------- READ DATASET --------
    data = pd.read_csv(path_to_raw_data, encoding="ISO-8859-1")
    data['clean_data'] = pd.read_csv(path_to_clean_data, encoding="ISO-8859-1")
    features.insert(0, vector_type)
    vector_list = [pd.DataFrame(pd.read_pickle(path_to_vectors + f_type + '_vectors.pckl')) for f_type in features]

    vectors = pd.concat(vector_list)

    train_data, test_data, train_labels, test_labels = train_test_split(vectors, data['label'], test_size=split)

    # train_and_evaluate((train_data, test_data, train_labels, test_labels))

    model = train_model(model_name, vector_type, features, train_data, train_labels)
    evaluate_model(model, test_data, test_labels)

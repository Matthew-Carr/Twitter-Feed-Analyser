import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from Console import get_tweets
from sklearn.decomposition import LatentDirichletAllocation

file_name = open("../vectors/bow_vectors.pckl", 'wb')
data = pd.read_csv("../processed_data/RawData.csv", encoding="ISO-8859-1")
data['clean_data'] = pd.read_csv("../processed_data/CleanData.csv", encoding="ISO-8859-1")
nlp = spacy.load('en_core_web_md')


num_topics = 10
max_iterations = 150


def lemmatisation(text: str, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                     token.pos_ in allowed_postags and token.text not in {'url', 'subject', 'user'}])


def annotate_string(string: str, bow_vectoriser, trained_lda) -> list:
    vector = bow_vectoriser.transform(string)
    topic_distribution = trained_lda.transform(vector)
    return topic_distribution.tolist()


lemmatised_data = data['clean_data'].apply(lambda x: lemmatisation(x))
bow_vectoriser = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
vectorised_data = bow_vectoriser.fit_transform(lemmatised_data)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iterations, learning_method='online',
                                verbose=True)
topic_distribution = lda.fit_transform(vectorised_data)



print("LDA Model:")
top_n = 10

words = bow_vectoriser.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("\nTopic #%d:" % topic_idx)
    print(" ".join([words[i]
                    for i in topic.argsort()[:-top_n - 1:-1]]))

print(topic_distribution.tolist())



training_data, testing_data, training_labels, testing_labels = train_test_split(topic_distribution, data['label'], test_size=0.2)

print('----------- RESULTS ------------')
print('\nLogistic Regression')
lreg = LogisticRegression(C=10, solver='lbfgs', max_iter=1000, multi_class="multinomial")
lreg.fit(training_data, training_labels)
predictions = lreg.predict(testing_data)
f1 = f1_score(testing_labels, predictions, average='weighted')
print('F1 score: ', f1)
# Accuracy
score = lreg.score(testing_data, testing_labels)
print('Accuracy: ', score)
#
# print('\nSVM-Linear')
# svml = SVC(gamma='auto', C=10, kernel='linear')
# svml.fit(training_data, training_labels)
# predictions = svml.predict(testing_data)
# f1 = f1_score(testing_labels, predictions, average='weighted')
# print('F1 score: ', f1)
# # Accuracy
# score = svml.score(testing_data, testing_labels)
# print('Accuracy: ', score)
#
# print('\nSVM-Radial')
# svmr = SVC(gamma='auto', C=10, kernel='rbf')
# svmr.fit(training_data, training_labels)
# predictions = svmr.predict(testing_data)
# f1 = f1_score(testing_labels, predictions, average='weighted')
# print('F1 score: ', f1)
# # Accuracy
# score = svmr.score(testing_data, testing_labels)
# print('Accuracy: ', score)
#
# print('\nGaussian Naive Bayes')
# gaussianNB = GaussianNB()
# gaussianNB.fit(training_data, training_labels)
# predictions = gaussianNB.predict(testing_data)
# f1 = f1_score(testing_labels, predictions, average='weighted')
# print('F1 score: ', f1)
# # Accuracy
# score = gaussianNB.score(testing_data, testing_labels)
# print('Accuracy: ', score)
#
# print('\nRandom Forest Classifier')
# rfc = RandomForestClassifier(n_estimators = 100, max_depth = None, max_features = 'sqrt')
# rfc.fit(training_data, training_labels)
# predictions = rfc.predict(testing_data)
# f1 = f1_score(testing_labels, predictions, average='weighted')
# print('F1 score: ', f1)
# # Accuracy
# score = rfc.score(testing_data, testing_labels)
# print('Accuracy: ', score)


#clf = DecisionTreeClassifier(random_state=0)
#evaluate(clf)
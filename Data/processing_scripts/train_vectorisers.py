from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import pickle
import spacy
from sklearn.decomposition import LatentDirichletAllocation
nlp = spacy.load('en_core_web_sm')


class Vectoriser:
    def __init__(self, trained_vectorisers: list):
        self.trained_vectorisers = trained_vectorisers
        # all vectorisers should have the transform method
        # they should be TRAINED

    def transform(self, list_of_strings: list):
        list_of_vectors = [[] for _ in range(len(list_of_strings))]

        for vectoriser in self.trained_vectorisers:
            feature_list = vectoriser.transform(list_of_strings)

            try:
                feature_list = feature_list.toarray()
            except AttributeError:
                pass

            try:
                feature_list = feature_list.tolist()
            except AttributeError:
                pass

            for idx, feature in enumerate(feature_list):
                list_of_vectors[idx] += feature

        return list_of_vectors


class TopicModelAnnotator:
    def __init__(self):
        self.topics = 10
        self.max_iterations = 30
        self.vectoriser = None
        self.lda = None

    def transform(self, string: str) -> list:
        vector = self.vectoriser.transform(string)
        topic_distribution = self.lda.transform(vector)
        return topic_distribution.tolist()

    def batch_annotate(self, data: pd.Series):
        self.vectoriser = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        vectorised_data = self.vectoriser.fit_transform(data.apply(lambda x: " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in nlp(x) if
                         token.pos_ in ('NOUN', 'ADJ', 'VERB', 'ADV')])))
        self.lda = LatentDirichletAllocation(n_components=self.topics, max_iter=self.max_iterations, learning_method='online',
                                             verbose=True)
        distribution = self.lda.fit_transform(vectorised_data)

        top_n = 10

        words = self.vectoriser.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-top_n - 1:-1]]))
        return distribution


def custom_vectoriser(vector_types: list):
    list_of_vectorisers = [pd.read_pickle("Data/vectorisers/" + v_type + "_vectoriser.pckl") for v_type in vector_types]
    return Vectoriser(list_of_vectorisers)


def create_vectors(dataset: pd.DataFrame, vector: str, vectoriser) -> np.ndarray:
    if vector not in {'tf_idf', 'bag_of_words', 'topic'}:
        raise ValueError('vector parameter must have value "tf_idf" or "bag_of_words" or "topic"')

    file = open("../vectors/" + vector + "_vectors.pckl", 'wb')

    if vector == 'topic':
        array = vectoriser.batch_annotate(dataset['clean_data'])
    else:
        matrix = vectoriser.fit_transform(dataset['clean_data'])
        array = matrix.toarray()


    if vector in {'tf_idf', 'bag_of_words', 'topic'}:
        vectoriser_file = open("../vectorisers/" + vector + "_vectoriser.pckl", 'wb')
        pickle.dump(vectoriser, file=vectoriser_file)
        vectoriser_file.close()

    pickle.dump(array, file=file)
    file.close()
    return array


if __name__ == "__main__":
    data = pd.read_csv("../processed_data/CleanData.csv", encoding="ISO-8859-1")
    # create_vectors(data, "tf_idf", TfidfVectorizer(max_features=5000))
    # create_vectors(data, "bag_of_words", CountVectorizer(max_features=5000))
    create_vectors(data, "topic", TopicModelAnnotator())



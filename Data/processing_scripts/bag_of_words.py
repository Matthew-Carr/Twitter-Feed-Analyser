from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle


file_name = open("../vectors/bow_vectors.pckl", 'wb')

data = pd.read_csv("../processed_data/CleanData.csv", encoding="ISO-8859-1")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_data'])
X = X.toarray()
pickle.dump(X, file=file_name)
file_name.close()
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle


# file_name = open("../vectors/tf_idf_5000_vectors.pckl", 'wb')

data = pd.read_csv("../processed_data/CleanData.csv", encoding="ISO-8859-1")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_data'])
X = X.toarray()

#
# pickle.dump(X, file=file_name)
# file_name.close()
#

y = vectorizer.transform(['hello mattat'])
print(vectorizer.get_feature_names())
print(y.toarray().shape)
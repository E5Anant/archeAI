from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stop_words = ENGLISH_STOP_WORDS.union({"your", "custom", "stopwords", "here"})
self.tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
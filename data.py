import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./letterboxd_250movie_reviews.csv")
print(df.shape)
print(df.dtypes)
print(df.head())   # confirm the real column names


def stars_to_num(s):
    if pd.isna(s):
        return np.nan
    return s.count("★") + (0.5 if "½" in s else 0.0)


df["rating_num"] = df["Rating"].apply(stars_to_num)
df = df.dropna(subset=["rating_num", "Review"])
print(df["rating_num"].value_counts().sort_index())   # check the distribution

# currently ratings are in terms of stars
# do we need to convert stars to concrete number?
Y = df["rating_num"].to_numpy(dtype=str)
print(Y.shape)

# Features: turn the review text into numbers
# vec = TfidfVectorizer(max_features=1000, stop_words="english")
# X = vec.fit_transform(df["review"])              # sparse (n_reviews, 1000)
# print(X.shape)
# .toarray() only if you need it dense
# print(X[:5].toarray())

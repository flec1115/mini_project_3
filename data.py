from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

# Resolve the CSV relative to this script, not the terminal's working directory.
DATA_PATH = Path(__file__).with_name("letterboxd_250movie_reviews.csv")


def stars_to_num(rating):
    """Convert Letterboxd stars to a numeric target; leave missing ratings as NaN."""
    if pd.isna(rating) or not rating.strip():
        return np.nan

    rating = rating.strip()
    value = rating.count("★") + (0.5 if "½" in rating else 0.0)
    # Accept only whole stars followed by an optional half, within the rating scale.
    if not re.fullmatch(r"★{0,5}½?", rating) or not 0.5 <= value <= 5:
        raise ValueError(f"Unexpected star rating: {rating!r}")
    return value


def load_and_clean_data(csv_path=DATA_PATH):
    """Keep labeled, nonblank reviews and remove repeated review/rating pairs."""
    # Other columns, including Status, are not needed for text-based prediction.
    df = pd.read_csv(csv_path, usecols=["Review", "Rating"], dtype="string")
    print("Missing values:")
    print(df.isna().sum())
    original_rows = len(df)

    df["rating_num"] = df["Rating"].apply(stars_to_num)
    # A supervised example needs both review text and a known target rating.
    df = df.dropna(subset=["Review", "rating_num"]).copy()
    # Collapse spaces, tabs, and newlines without stripping sentiment-bearing words.
    df["Review"] = df["Review"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Whitespace-only reviews become empty strings, which dropna does not remove.
    df = df.loc[df["Review"].ne("")]
    # Deduplicate after normalization; retain differing ratings for the same text.
    df = df.drop_duplicates(subset=["Review", "rating_num"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable labeled reviews remain after cleaning.")

    print(f"\nRemoved rows: {original_rows - len(df)}")
    print(f"Usable labeled reviews: {len(df)}")
    print("\nRating distribution:")
    # Uneven rating counts can reveal a bias toward high or low scores.
    print(df["rating_num"].value_counts().sort_index())
    return df


def split_reviews(df, test_size=0.2, random_state=42):
    """Split raw reviews before learning features, keeping duplicate text together."""
    if df["Review"].nunique() < 2:
        raise ValueError("At least two distinct reviews are needed for a train/test split.")

    # Keep identical reviews together, even when they have different ratings.
    # The 20% test fraction applies to review groups, not necessarily 20% of rows.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, test_indices = next(splitter.split(df, groups=df["Review"]))
    # The splitter returns row positions, so select with iloc rather than loc.
    return df.iloc[train_indices].copy(), df.iloc[test_indices].copy()


def preprocess_data(csv_path=DATA_PATH):
    """Return sparse train/test features, numeric targets, and the fitted vectorizer."""
    df = load_and_clean_data(csv_path)
    train, test = split_reviews(df)

    # Keep words such as "not" and "never", which matter for sentiment.
    vectorizer = TfidfVectorizer(max_features=1000)
    # Learn vocabulary and inverse-document-frequency weights from training data only.
    X_train = vectorizer.fit_transform(train["Review"])
    # Reuse training weights; learning from test reviews would leak evaluation data.
    X_test = vectorizer.transform(test["Review"])
    # Keep targets numeric and in the same row order as their feature matrices.
    y_train = train["rating_num"].to_numpy(dtype=float)
    y_test = test["rating_num"].to_numpy(dtype=float)

    # Preserve sparse matrices to save memory and the vectorizer for future reviews.
    return X_train, X_test, y_train, y_test, vectorizer


# Importing these helpers should not automatically read and preprocess the dataset.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data()
    # Feature shapes are (reviews, vocabulary size); target shapes are (reviews,).
    print("\nTraining features:", X_train.shape)
    print("Test features:", X_test.shape)
    print("Training ratings:", y_train.shape)
    print("Test ratings:", y_test.shape)

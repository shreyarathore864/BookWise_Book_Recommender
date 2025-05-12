import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = r"E:\\github\\BOOK-WISE\\data"

# Function to safely load CSV
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, engine="python", quoting=3, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
        return pd.DataFrame()
    
    print(f"Loaded {filename} with columns: {list(df.columns)}")  # Debug line
    return df

# Preprocess and align columns
def preprocess(df, source):
    df = df.copy()
    df["source"] = source

    df["title"] = df.get("title", df.get("Title", pd.Series([""] * len(df)))).astype(str)
    
    if source == "Goodreads":
        authors = df.get("authors", pd.Series(["Unknown Author"] * len(df))).astype(str)
        df["description"] = df["title"] + " by " + authors
        df["rating"] = df.get("average_rating", pd.Series(["N/A"] * len(df)))
        df["image_url"] = pd.Series([""] * len(df))  # Goodreads data has no image URL
    else:  # Kindle dataset
        author = df.get("author", pd.Series(["Unknown Author"] * len(df))).astype(str)
        category = df.get("category_name", pd.Series([""] * len(df))).astype(str)
        df["description"] = df["title"] + " by " + author + " - Category: " + category
        df["rating"] = df.get("stars", pd.Series(["N/A"] * len(df)))
        df["image_url"] = df.get("imgUrl", pd.Series([""] * len(df)))

    return df[["title", "description", "source", "rating", "image_url"]]


# Load and preprocess datasets
goodreads_df = preprocess(load_csv("books.csv"), "Goodreads")
kindle_df = preprocess(load_csv("kindle_data-v2.csv"), "Kindle")

# Merge both
books_df = pd.concat([goodreads_df, kindle_df], ignore_index=True)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books_df["description"])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, num_recommendations=5):
    indices = pd.Series(books_df.index, index=books_df["title"].str.lower())
    title = title.lower()

    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    book_indices = [i[0] for i in sim_scores]

    return books_df.iloc[book_indices][["title", "source", "rating", "image_url"]].to_dict(orient="records")

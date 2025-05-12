import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(
    page_title="BookWise | Find Your Next Read",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4B5563;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .book-card {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    .book-card:hover {
        background-color: #F3F4F6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .book-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .book-metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }
    .book-badge {
        background-color: #E5E7EB;
        color: #4B5563;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
    }
    .badge-rating {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .badge-source {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .badge-genre {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .search-container {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .view-more-button button {
        background-color: transparent;
        color: #1E3A8A;
        border: 1px solid #1E3A8A;
    }
    .view-more-button button:hover {
        background-color: #EFF6FF;
        color: #1E40AF;
    }
    .stSelectbox div [data-baseweb="select"] {
        border-radius: 8px;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
        text-align: center;
    }
    hr {
        margin: 2rem 0;
        border-color: #E5E7EB;
    }
    .empty-image-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #E5E7EB;
        color: #6B7280;
        border-radius: 8px;
        height: 150px;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ FILE PATH SETUP ------------------
data_dir = Path("data")
goodreads_path = data_dir / "books.csv"
kindle_path = data_dir / "kindle_data-v2.csv"

# ------------------ UTILITY FUNCTIONS ------------------
@st.cache_data
def preprocess(df, source):
    """Preprocess the dataframe and standardize columns"""
    df = df.copy()
    df["source"] = source
    df["title"] = df.get("title", df.get("Title", pd.Series(["Untitled"] * len(df)))).astype(str)

    if source == "Goodreads":
        authors = df.get("authors", pd.Series(["Unknown Author"] * len(df))).astype(str)
        df["description"] = df["title"] + " by " + authors
        df["rating"] = df.get("average_rating", "N/A")
        df["image_url"] = df.get("image_url", "")
        df["genre"] = df.get("genres", "")
    else:
        author = df.get("author", pd.Series(["Unknown Author"] * len(df))).astype(str)
        category = df.get("category_name", "")
        df["description"] = df["title"] + " by " + author + " - Category: " + category
        df["rating"] = df.get("stars", "N/A")
        df["image_url"] = df.get("imgUrl", "")
        df["genre"] = category

    df = df[["title", "description", "source", "rating", "image_url", "genre"]].fillna("")
    df = df[df["description"].str.strip() != ""]
    df = df[df["description"].str.len() > 10]
    return df

@st.cache_data
def load_and_process_data():
    """Load and process both datasets"""
    goodreads_df = preprocess(pd.read_csv(goodreads_path, on_bad_lines='skip'), "Goodreads")
    kindle_df = preprocess(pd.read_csv(kindle_path), "Kindle")
    return pd.concat([goodreads_df, kindle_df], ignore_index=True)

@st.cache_resource
def build_recommendation_model(books_df):
    """Build the TF-IDF vectorizer and Nearest Neighbors model"""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["description"])
    
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)
    
    index_map = pd.Series(books_df.index, index=books_df["title"].str.lower().str.strip())
    
    return tfidf, tfidf_matrix, nn_model, index_map

def render_book_card(row):
    """Render a book card with all details"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            if pd.notna(row["image_url"]) and row["image_url"] != "":
                st.image(row["image_url"], width=130)
            else:
                st.markdown("""
                    <div class="empty-image-placeholder">
                        <span>📖<br>No Cover Available</span>
                    </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{row['title']}</div>
                    <div class="book-metadata">
                        <span class="book-badge badge-rating">⭐ {row['rating']}</span>
                        <span class="book-badge badge-source">📚 {row['source']}</span>
                        <span class="book-badge badge-genre">🏷️ {row['genre'][:30] + ('...' if len(row['genre']) > 30 else '')}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def get_recommendations(title, nn_model, tfidf_matrix, index_map, books_df, num_recommendations=5):
    """Get book recommendations based on title"""
    title = title.lower().strip()
    if title not in index_map:
        return pd.DataFrame()
    
    idx = index_map[title]
    distances, indices_found = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
    book_indices = indices_found[0][1:]  # skip self
    return books_df.iloc[book_indices]

# ------------------ LOAD DATA & BUILD MODEL ------------------
books_df = load_and_process_data()
tfidf, tfidf_matrix, nn_model, index_map = build_recommendation_model(books_df)

# ------------------ SESSION STATE INITIALIZATION ------------------
if "genre_count" not in st.session_state:
    st.session_state.genre_count = 10

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ------------------ HEADER SECTION ------------------
st.markdown('<p class="main-header">📚 BookWise</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent companion for discovering your next favorite book</p>', unsafe_allow_html=True)

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["📚 Genre-Based Discovery", "🔍 Title Search"])

# ------------------ TAB 1: GENRE-BASED DISCOVERY ------------------
with tab1:
    st.markdown('<p class="section-header">Discover Books by Genre</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="search-container">
                <p>Browse through our collection and find books by genre</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Genre filter
        genres = sorted(books_df["genre"].dropna().unique())
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_genre = st.selectbox("🎨 Select a Genre", ["All"] + genres)
        
        with col2:
            st.write("")
            st.write("")
            get_genre_recs = st.button("📚 Browse Books", key="genre_button")
    
    if get_genre_recs or selected_genre != "All":
        if selected_genre != "All":
            genre_results = books_df[books_df["genre"].str.lower().str.contains(selected_genre.lower())]
        else:
            genre_results = books_df
        
        if not genre_results.empty:
            st.markdown(f"### Found {len(genre_results)} books in *{selected_genre}*")
            shown_books = genre_results.head(st.session_state.genre_count)
            
            for _, row in shown_books.iterrows():
                render_book_card(row)
            
            # View more button
            if st.session_state.genre_count < len(genre_results):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="view-more-button">', unsafe_allow_html=True)
                    if st.button(f"🔄 View More ({st.session_state.genre_count} of {len(genre_results)})", key="view_more"):
                        st.session_state.genre_count += 10
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No books found in this genre.")


# ------------------ TAB 2: TITLE-BASED SEARCH ------------------
with tab2:
    st.markdown('<p class="section-header">Find Similar Books</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="search-container">
                <p style="font-size: 16px;">Search for a book title to get personalized AI-powered recommendations.</p>
            </div>
        """, unsafe_allow_html=True)

        def update_search_query():
            st.session_state.search_query = st.session_state.book_input

        # Text input for title
        book_input = st.text_input(
            "📖 Search for a Book Title",
            value=st.session_state.get("search_query", ""),
            placeholder="Start typing...",
            key="book_input",
            on_change=update_search_query
        )

        selected_title = None
        suggestions = []

        if book_input:
            # Exact match titles
            exact = books_df[books_df["title"].str.lower().str.startswith(book_input.lower())]["title"].unique().tolist()
            # Partial match titles
            partial = books_df[
                ~books_df["title"].str.lower().str.startswith(book_input.lower()) &
                books_df["title"].str.lower().str.contains(book_input.lower())
            ]["title"].unique().tolist()

            suggestions = exact + partial

            if suggestions:
                selected_title = st.selectbox("📚 Select a book from suggestions:", suggestions, key="book_suggestions")

        col1, col2 = st.columns([3, 1])
        with col1:
            filter_genre = st.checkbox("🎨 Filter recommendations by genre", value=False)
            if filter_genre:
                selected_filter_genre = st.selectbox("Filter by genre:", ["All"] + genres, key="genre_selector")

        with col2:
            st.write("")
            st.write("")
            get_recs = st.button("✨ Get Recommendations", key="title_button", disabled=not selected_title)

    # Display recommendations
    if get_recs and selected_title:
        title_lower = selected_title.lower().strip()
        if title_lower in index_map:
            idx = index_map[title_lower]
            # Ensure idx is scalar
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            idx = int(idx)

            distances, indices_found = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=11)
            rec_indices = [i for i in indices_found[0] if int(i) != idx][:10]
            results = books_df.iloc[rec_indices]

            if filter_genre and selected_filter_genre.lower() != "all":
                results = results[results["genre"].str.lower().str.contains(selected_filter_genre.lower())]

            if not results.empty:
                st.markdown("### 📖 You selected:")
                render_book_card(books_df.iloc[idx])

                st.markdown(f"### 🪄 Books similar to *{selected_title}*:")
                for _, row in results.iterrows():
                    render_book_card(row)
            else:
                st.warning("No similar books found for your query.")
        else:
            st.warning("The selected book title was not found.")


# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 1rem 0;">
        BookWise: Your Intelligent Book Recommendation System<br>
        Find your next great read with the power of AI
    </div>
""", unsafe_allow_html=True)
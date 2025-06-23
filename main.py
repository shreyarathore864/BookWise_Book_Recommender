import streamlit as st
import pandas as pd
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import random
import re

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
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .book-card:hover {
        background-color: #F3F4F6;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .book-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .book-description {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 15px;
        line-height: 1.5;
        text-align: justify;
    }
    .book-metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 15px;
    }
    .book-badge {
        background-color: #E5E7EB;
        color: #4B5563;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        font-weight: 500;
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
    .external-link {
        display: inline-flex;
        align-items: center;
        padding: 8px 15px;
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        font-size: 0.85rem;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .external-link:hover {
        background-color: #1E40AF;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        text-decoration: none;
        color: white;
    }
    .search-container {
        background-color: transparent;
        padding: 20px;
        border-radius: 12px;
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
        transform: translateY(-2px);
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
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    .empty-image-placeholder:hover {
        background-color: #D1D5DB;
    }
    .image-container {
        transition: all 0.3s ease;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .result-header {
        background-color: #EFF6FF;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #1E3A8A;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 4px solid #1E3A8A;
    }
    .tabs-container .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F9FAFB;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .tabs-container .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: white;
    }
    .tabs-container .stTabs [aria-selected="true"] {
        background-color: #EFF6FF;
        font-weight: 600;
    }
    .trending-tag {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #EF4444;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        transform: rotate(12deg);
        z-index: 10;
    }
    .popular-tag {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #F59E0B;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        transform: rotate(12deg);
        z-index: 10;
    }
    .alert-container {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-info {
        background-color: #DBEAFE;
        border-left: 4px solid #1E40AF;
        color: #1E40AF;
    }
    .alert-warning {
        background-color: #FEF3C7;
        border-left: 4px solid #92400E;
        color: #92400E;
    }
    .alert-error {
        background-color: #FEE2E2;
        border-left: 4px solid #B91C1C;
        color: #B91C1C;
    }
    .stTabs {
        background-color: transparent;
    }
    .book-content-row {
        display: flex;
        gap: 20px;
        align-items: flex-start;
    }
    .book-image-col {
        flex: 0 0 120px;
    }
    .book-details-col {
        flex: 1;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ UTILITY FUNCTIONS ------------------
def display_alert(message, alert_type="info"):
    """Display a styled alert box with a message"""
    icon = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌"
    }.get(alert_type, "ℹ️")
    
    st.markdown(f"""
        <div class="alert-container alert-{alert_type}">
            <span>{icon}</span>
            <span>{message}</span>
        </div>
    """, unsafe_allow_html=True)

def clean_text(text):
    """Clean and truncate text for display"""
    if pd.isna(text) or text == "":
        return "No description available."
    
    text = str(text)
    # Remove any HTML tags if present
    text = re.sub('<.*?>', '', text)
    # Remove any remaining HTML entities
    text = re.sub('&[a-zA-Z0-9#]+;', '', text)
    
    # Truncate if too long
    if len(text) > 200:
        text = text[:200] + "..."
    
    return text

def safe_str(value):
    """Safely convert value to string, handling NaN and None"""
    if pd.isna(value) or value is None:
        return ""
    return str(value)

# ------------------ FILE PATH SETUP ------------------
# Use relative paths instead of absolute paths for better compatibility
data_dir = Path("data")

# For development purposes, if files aren't found, use sample data
try:
    if os.path.exists(data_dir / "books.csv"):
        goodreads_path = data_dir / "books.csv"
    else:
        # If running from GitHub or specific path is not found
        goodreads_path = Path("E:/github/BOOK-WISE/data/books.csv")
        
    if os.path.exists(data_dir / "kindle_data-v2.csv"):
        kindle_path = data_dir / "kindle_data-v2.csv"
    else:
        # If running from GitHub or specific path is not found
        kindle_path = Path("E:/github/BOOK-WISE/data/kindle_data-v2.csv")
        
    if not os.path.exists(goodreads_path) or not os.path.exists(kindle_path):
        st.warning("📚 Sample data will be used as the original data files couldn't be located.")
        use_sample_data = True
    else:
        use_sample_data = False
except Exception:
    st.warning("📚 Sample data will be used as the original data files couldn't be located.")
    use_sample_data = True

# ------------------ SAMPLE DATA GENERATION ------------------
def generate_sample_data():
    """Generate sample data if the original CSV files are not found"""
    # Sample Goodreads data
    goodreads_data = {
        'title': [
            'To Kill a Mockingbird', 'Pride and Prejudice', '1984', 
            'The Great Gatsby', 'The Catcher in the Rye', 'Brave New World',
            'Animal Farm', 'The Lord of the Rings', 'The Hobbit', 'Harry Potter and the Sorcerer\'s Stone'
        ],
        'authors': [
            'Harper Lee', 'Jane Austen', 'George Orwell', 
            'F. Scott Fitzgerald', 'J.D. Salinger', 'Aldous Huxley',
            'George Orwell', 'J.R.R. Tolkien', 'J.R.R. Tolkien', 'J.K. Rowling'
        ],
        'average_rating': [4.27, 4.25, 4.17, 3.91, 3.80, 4.0, 3.95, 4.35, 4.27, 4.47],
        'description': [
            'A powerful tale of racial injustice and moral growth in the American South during the 1930s.',
            'A classic novel exploring societal expectations, marriage, and class in early 19th-century England.',
            'A dystopian novel set in a totalitarian society ruled by Big Brother and the Party.',
            'A tragic story of wealth, love, and the American Dream during the Roaring Twenties.',
            'A coming-of-age novel following the adventures of Holden Caulfield in New York City.',
            'A dystopian novel depicting a genetically engineered society controlled by technology.',
            'An allegorical novella reflecting on the Russian Revolution and the rise of Stalinism.',
            'An epic fantasy adventure about the quest to destroy the One Ring and defeat the Dark Lord Sauron.',
            'The tale of Bilbo Baggins and his journey to help a group of dwarves reclaim their homeland.',
            'The first book in the Harry Potter series, introducing the magical world of wizardry.'
        ],
        'genres': [
            'Fiction, Classics', 'Fiction, Romance, Classics', 'Fiction, Dystopian, Classics',
            'Fiction, Classics', 'Fiction, Coming of Age', 'Fiction, Dystopian, Classics',
            'Fiction, Allegory, Classics', 'Fiction, Fantasy, Adventure', 'Fiction, Fantasy, Adventure',
            'Fiction, Fantasy, Young Adult'
        ],
        'image_url': [
            'https://images.gr-assets.com/books/1553383690l/2657.jpg',
            'https://images.gr-assets.com/books/1320399351l/1885.jpg',
            'https://images.gr-assets.com/books/1348990566l/5470.jpg',
            'https://images.gr-assets.com/books/1490528560l/4671.jpg',
            'https://images.gr-assets.com/books/1398034300l/5107.jpg',
            'https://images.gr-assets.com/books/1575509280l/5129.jpg',
            'https://images.gr-assets.com/books/1424037542l/7613.jpg',
            'https://images.gr-assets.com/books/1566425108l/33.jpg',
            'https://images.gr-assets.com/books/1546071216l/5907.jpg',
            'https://images.gr-assets.com/books/1474154022l/3.jpg'
        ]
    }
    
    # Sample Kindle data
    kindle_data = {
        'Title': [
            'The Alchemist', 'The Da Vinci Code', 'The Hunger Games', 
            'The Girl with the Dragon Tattoo', 'The Shining', 'Dune',
            'Gone Girl', 'The Silent Patient', 'Where the Crawdads Sing', 'Educated'
        ],
        'author': [
            'Paulo Coelho', 'Dan Brown', 'Suzanne Collins', 
            'Stieg Larsson', 'Stephen King', 'Frank Herbert',
            'Gillian Flynn', 'Alex Michaelides', 'Delia Owens', 'Tara Westover'
        ],
        'stars': [4.6, 4.2, 4.3, 4.1, 4.4, 4.5, 4.0, 4.5, 4.8, 4.7],
        'category_name': [
            'Fiction, Philosophy', 'Fiction, Thriller, Mystery', 'Fiction, Young Adult, Dystopian',
            'Fiction, Mystery, Thriller', 'Fiction, Horror', 'Fiction, Science Fiction',
            'Fiction, Mystery, Thriller', 'Fiction, Psychological Thriller', 'Fiction, Mystery',
            'Non-Fiction, Memoir'
        ],
        'imgUrl': [
            'https://images-na.ssl-images-amazon.com/images/I/41ybG235TcL._SX329_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/51vw2QyiNSL._SX304_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/41V56ye3PrL._SX331_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/51qdH4S-bTL._SX334_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/41PfG6klhTL._SX310_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/41yN1XxHBOL._SX331_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/41hnNzuDHPL._SX327_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/51svtLI3XdL._SX329_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/51j5p8XA84L._SX330_BO1,204,203,200_.jpg',
            'https://images-na.ssl-images-amazon.com/images/I/41-oQwYLdML._SX329_BO1,204,203,200_.jpg'
        ]
    }
    
    return pd.DataFrame(goodreads_data), pd.DataFrame(kindle_data)

# ------------------ DATA LOADING & PROCESSING ------------------
@st.cache_data
def preprocess(df, source):
    """Preprocess the dataframe and standardize columns"""
    df = df.copy()
    df["source"] = source
    
    # Standardize title column
    if "title" in df.columns:
        df["title"] = df["title"].astype(str)
    elif "Title" in df.columns:
        df["title"] = df["Title"].astype(str)
    else:
        df["title"] = "Untitled"
    
    if source == "Goodreads":
        # Handle Goodreads data
        authors = df.get("authors", pd.Series(["Unknown Author"] * len(df))).astype(str)
        df["search_text"] = df["title"] + " by " + authors
        
        # Get the actual description if available
        if "description" in df.columns:
            actual_desc = df["description"].fillna("").astype(str)
            df["summary"] = actual_desc.apply(lambda x: clean_text(x) if x and len(x) > 20 else "")
        else:
            df["summary"] = ""
            
        # Fill empty summaries with a basic description
        mask = (df["summary"] == "") | (df["summary"] == "No description available.")
        df.loc[mask, "summary"] = "A book by " + authors[mask]
            
        df["rating"] = df.get("average_rating", "N/A")
        df["image_url"] = df.get("image_url", "")
        df["genre"] = df.get("genres", "")
        df["external_url"] = "https://www.goodreads.com/search?q=" + df["title"].str.replace(" ", "+")
    else:
        # Handle Kindle data
        author = df.get("author", pd.Series(["Unknown Author"] * len(df))).astype(str)
        category = df.get("category_name", "")
        df["search_text"] = df["title"] + " by " + author + " - Category: " + category.astype(str)
        df["summary"] = "A book by " + author + " in the " + category.astype(str) + " category."
        df["rating"] = df.get("stars", "N/A")
        df["image_url"] = df.get("imgUrl", "")
        df["genre"] = category
        df["external_url"] = "https://www.amazon.com/s?k=" + df["title"].str.replace(" ", "+")

    # Select and clean required columns
    result_df = df[["title", "search_text", "summary", "source", "rating", "image_url", "genre", "external_url"]].copy()
    
    # Safely handle and clean data
    result_df["search_text"] = result_df["search_text"].fillna("").astype(str)
    result_df["summary"] = result_df["summary"].fillna("").astype(str)
    result_df["rating"] = result_df["rating"].fillna("N/A")
    result_df["image_url"] = result_df["image_url"].fillna("")
    result_df["genre"] = result_df["genre"].fillna("")
    result_df["external_url"] = result_df["external_url"].fillna("#")
    
    # Filter out books with empty search text
    result_df = result_df[result_df["search_text"].str.strip() != ""]
    result_df = result_df[result_df["search_text"].str.len() > 5]
    
    return result_df

@st.cache_data
def load_and_process_data():
    """Load and process both datasets"""
    if use_sample_data:
        goodreads_df, kindle_df = generate_sample_data()
    else:
        try:
            goodreads_df = pd.read_csv(goodreads_path, on_bad_lines='skip')
        except Exception:
            st.error("Error loading Goodreads data. Using sample data instead.")
            goodreads_df, _ = generate_sample_data()
            
        try:
            kindle_df = pd.read_csv(kindle_path)
        except Exception:
            st.error("Error loading Kindle data. Using sample data instead.")
            _, kindle_df = generate_sample_data()
    
    # Process both datasets
    goodreads_processed = preprocess(goodreads_df, "Goodreads")
    kindle_processed = preprocess(kindle_df, "Kindle")
    
    # Combine and return
    combined_df = pd.concat([goodreads_processed, kindle_processed], ignore_index=True)
    return combined_df

@st.cache_resource
def build_recommendation_model(books_df):
    """Build the TF-IDF vectorizer and Nearest Neighbors model"""
    try:
        # Ensure search_text column contains strings
        descriptions = books_df["search_text"].fillna("").astype(str).tolist()
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = tfidf.fit_transform(descriptions)
        
        # Build nearest neighbors model
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
        nn_model.fit(tfidf_matrix)
        
        # Create title-to-index mapping
        titles = books_df["title"].str.lower().str.strip()
        index_map = {}
        
        for i, title in enumerate(titles):
            index_map[title] = i
        
        return tfidf, tfidf_matrix, nn_model, index_map
    except Exception as e:
        st.error(f"Error building recommendation model: {str(e)}")
        return None, None, None, None

def render_book_card(row, show_description=True):
    """Render a book card with all details using Streamlit components"""
    try:
        # Randomly assign trending/popular tags to some books for visual interest
        tag_text = ""
        if random.random() < 0.1:  # 10% chance for trending
            tag_text = "🔥 Trending"
        elif random.random() < 0.15:  # 15% chance for popular
            tag_text = "⭐ Popular"
        
        # Format rating for display
        try:
            rating_display = f"{float(row['rating']):.1f}" if pd.notna(row['rating']) and row['rating'] != 'N/A' else "N/A"
        except:
            rating_display = "N/A"
        
        # Format genre for display
        genre_display = safe_str(row['genre'])[:50] + ('...' if len(safe_str(row['genre'])) > 50 else '')
        
        # Clean text for display
        title_text = safe_str(row['title'])
        summary_text = clean_text(row['summary']) if show_description else ""
        
        # Create container for the book card
        with st.container():
            # Create columns for image and content
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Display image or placeholder
                if pd.notna(row["image_url"]) and row["image_url"] != "":
                    try:
                        st.image(row["image_url"], width=120)
                    except:
                        st.markdown("📖")
                        st.caption("Image Error")
                else:
                    st.markdown("📚")
                    st.caption("No Cover")
            
            with col2:
                # Book details
                if tag_text:
                    st.markdown(f"**{tag_text}**")
                
                st.markdown(f"### {title_text}")
                
                if show_description and summary_text:
                    st.write(summary_text)
                
                # Metadata badges
                col_rating, col_source, col_genre = st.columns(3)
                
                with col_rating:
                    st.markdown(f"⭐ **{rating_display}**")
                
                with col_source:
                    st.markdown(f"📚 **{safe_str(row['source'])}**")
                
                with col_genre:
                    if genre_display:
                        st.markdown(f"🏷️ **{genre_display}**")
                
                # External link
                if safe_str(row['external_url']) != "#" and safe_str(row['external_url']) != "":
                    st.markdown(f"[🔍 View on {safe_str(row['source'])} →]({safe_str(row['external_url'])})")
                
            st.divider()
                
    except Exception as e:
        st.error(f"Error rendering book card: {str(e)}")

def get_recommendations(title, nn_model, tfidf_matrix, index_map, books_df, num_recommendations=5):
    """Get book recommendations based on title"""
    try:
        # Clean input title
        title = title.lower().strip()
        
        # Check if the title exists in our index
        if title not in index_map:
            return pd.DataFrame()
        
        # Get the index of the title
        idx = index_map[title]
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(tfidf_matrix[idx:idx+1], n_neighbors=num_recommendations+1)
        
        # Skip the first index (which is the book itself)
        indices = indices.flatten()[1:]
        
        # Return recommendations
        return books_df.iloc[indices]
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

# ------------------ LOAD DATA & BUILD MODEL ------------------
try:
    books_df = load_and_process_data()
    tfidf, tfidf_matrix, nn_model, index_map = build_recommendation_model(books_df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ------------------ SESSION STATE INITIALIZATION ------------------
if "genre_count" not in st.session_state:
    st.session_state.genre_count = 10

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "last_selected_title" not in st.session_state:
    st.session_state.last_selected_title = None

# ------------------ HEADER SECTION ------------------
st.markdown('<p class="main-header">📚 BookWise</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent companion for discovering your next favorite book</p>', unsafe_allow_html=True)

# ------------------ TABS ------------------
st.markdown('<div class="tabs-container">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📚 Genre-Based Discovery", "🔍 Title Search"])
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB 1: GENRE-BASED DISCOVERY ------------------
with tab1:
    st.markdown('<div class="section-header">Discover Books by Genre</div>', unsafe_allow_html=True)
    
    # Get unique genres/categories
    all_genres = []
    for genre in books_df["genre"].dropna():
        genre_str = str(genre)
        # Split by common delimiters
        genres = re.split('[,;/&]', genre_str)
        for g in genres:
            g = g.strip()
            if g and len(g) > 2:
                all_genres.append(g)
    
    # Get unique genres and sort them
    unique_genres = sorted(list(set(all_genres)))
    
    # Filter controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_genre = st.selectbox(
            "🏷️ Select a genre to explore:",
            ["All Genres"] + unique_genres,
            index=0,
            help="Choose a genre to discover books from both Goodreads and Kindle datasets"
        )
    
    with col2:
        source_filter = st.selectbox(
            "📚 Filter by source:",
            ["All Sources", "Goodreads", "Kindle"],
            index=0,
            help="Filter books by data source"
        )
    
    # Filter books based on genre and source
    if selected_genre == "All Genres":
        filtered_books = books_df.copy()
    else:
        # Filter by genre (case insensitive partial match)
        genre_mask = books_df["genre"].str.contains(selected_genre, case=False, na=False)
        filtered_books = books_df[genre_mask]
    
    # Apply source filter
    if source_filter != "All Sources":
        filtered_books = filtered_books[filtered_books["source"] == source_filter]
    
    # Sort by rating (descending) and then randomly shuffle within same ratings
    filtered_books = filtered_books.copy()
    try:
        # Convert ratings to numeric for sorting, keeping non-numeric as is
        filtered_books["rating_numeric"] = pd.to_numeric(filtered_books["rating"], errors='coerce')
        filtered_books = filtered_books.sort_values(['rating_numeric'], ascending=False, na_position='last')
    except:
        # If sorting fails, just shuffle randomly
        filtered_books = filtered_books.sample(frac=1).reset_index(drop=True)
    
    # Show results
    if len(filtered_books) == 0:
        display_alert("No books found for the selected genre and source combination. Try a different selection!", "warning")
    else:
        # Display result summary
        genre_text = selected_genre if selected_genre != "All Genres" else "all genres"
        source_text = f" from {source_filter}" if source_filter != "All Sources" else ""
        
        st.markdown(f"""
            <div class="result-header">
                📖 Found {len(filtered_books)} books in {genre_text}{source_text}
            </div>
        """, unsafe_allow_html=True)
        
        # Display books
        books_to_show = filtered_books.head(st.session_state.genre_count)
        
        for idx, (_, book) in enumerate(books_to_show.iterrows()):
            render_book_card(book, show_description=True)
        
        # Show more button
        if len(filtered_books) > st.session_state.genre_count:
            remaining = len(filtered_books) - st.session_state.genre_count
            st.markdown('<div class="view-more-button">', unsafe_allow_html=True)
            if st.button(f"📚 View {min(10, remaining)} More Books", key="genre_view_more"):
                st.session_state.genre_count += 10
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset view count button
        if st.session_state.genre_count > 10:
            if st.button("🔄 Reset View", key="genre_reset"):
                st.session_state.genre_count = 10
                st.rerun()

# ------------------ TAB 2: TITLE SEARCH ------------------
with tab2:
    st.markdown('<div class="section-header">Search and Get Recommendations</div>', unsafe_allow_html=True)
    
    # Search interface
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_query = st.text_input(
            "🔍 Enter a book title to get recommendations:",
            value=st.session_state.search_query,
            placeholder="e.g., Harry Potter, 1984, Pride and Prejudice...",
            help="Type the title of a book you like to get similar recommendations"
        )
    
    with search_col2:
        search_button = st.button("🚀 Get Recommendations", type="primary")
        clear_button = st.button("🗑️ Clear")
    
    # Handle clear button
    if clear_button:
        st.session_state.search_query = ""
        st.session_state.last_selected_title = None
        st.rerun()
    
    # Update session state
    st.session_state.search_query = search_query
    
    # Search logic
    if search_button or search_query:
        if search_query.strip():
            # Show search results for exact/partial matches
            search_results = books_df[
                books_df["title"].str.contains(search_query, case=False, na=False)
            ].head(5)
            
            if len(search_results) > 0:
                st.markdown("### 📚 Matching Books")
                
                # Create selection interface
                if len(search_results) == 1:
                    selected_book = search_results.iloc[0]
                    st.session_state.last_selected_title = selected_book["title"]
                else:
                    # Multiple results - let user choose
                    book_options = {f"{row['title']} ({row['source']})": row['title'] 
                                  for _, row in search_results.iterrows()}
                    
                    selected_display = st.selectbox(
                        "Select a book for recommendations:",
                        options=list(book_options.keys()),
                        key="book_selector"
                    )
                    
                    if selected_display:
                        selected_title = book_options[selected_display]
                        st.session_state.last_selected_title = selected_title
                        selected_book = search_results[search_results["title"] == selected_title].iloc[0]
                
                # Display selected book
                if st.session_state.last_selected_title:
                    selected_book = books_df[books_df["title"] == st.session_state.last_selected_title].iloc[0]
                    
                    st.markdown("#### 🎯 Selected Book")
                    render_book_card(selected_book, show_description=True)
                    
                    # Get and display recommendations
                    if tfidf is not None and nn_model is not None:
                        recommendations = get_recommendations(
                            st.session_state.last_selected_title,
                            nn_model, tfidf_matrix, index_map, books_df,
                            num_recommendations=6
                        )
                        
                        if len(recommendations) > 0:
                            st.markdown("### 🎯 Recommended Books")
                            st.markdown(f"""
                                <div class="result-header">
                                    🤖 Based on your selection, here are {len(recommendations)} similar books you might enjoy
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display recommendations
                            for idx, (_, book) in enumerate(recommendations.iterrows()):
                                render_book_card(book, show_description=True)
                        else:
                            display_alert("No recommendations found for this book. Try searching for another title!", "warning")
                    else:
                        display_alert("Recommendation system is not available. Please try again later.", "error")
            
            else:
                display_alert(f"No books found matching '{search_query}'. Try a different search term or check the spelling.", "warning")
                
                # Show some random suggestions
                st.markdown("### 💡 Popular Books You Might Like")
                random_books = books_df.sample(n=min(5, len(books_df))).reset_index(drop=True)
                
                for _, book in random_books.iterrows():
                    render_book_card(book, show_description=True)
        else:
            # Show some featured/popular books when no search query
            st.markdown("### ⭐ Featured Books")
            st.info("👆 Enter a book title above to get personalized recommendations, or browse these popular titles:")
            
            # Show top-rated books
            try:
                # Get books with numeric ratings
                rated_books = books_df.copy()
                rated_books["rating_numeric"] = pd.to_numeric(rated_books["rating"], errors='coerce')
                top_books = rated_books.dropna(subset=['rating_numeric']).nlargest(8, 'rating_numeric')
                
                if len(top_books) == 0:
                    top_books = books_df.sample(n=min(8, len(books_df)))
                
            except:
                top_books = books_df.sample(n=min(8, len(books_df)))
            
            for _, book in top_books.iterrows():
                render_book_card(book, show_description=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 20px;">
        <p>📚 <strong>BookWise</strong> - Discover your next favorite book with AI-powered recommendations</p>
        <p style="font-size: 0.9rem;">
            Data sources: Goodreads & Kindle • Built with Streamlit • Powered by Machine Learning
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR (OPTIONAL) ------------------
with st.sidebar:
    st.markdown("### 📊 Dataset Info")
    st.write(f"📚 Total Books: {len(books_df):,}")
    
    # Count books by source
    source_counts = books_df["source"].value_counts()
    for source, count in source_counts.items():
        st.write(f"📖 {source}: {count:,}")
    
    st.markdown("---")
    st.markdown("### 🔧 Features")
    st.write("• Genre-based discovery")
    st.write("• AI-powered recommendations")
    st.write("• Cross-platform search")
    st.write("• Detailed book information")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write("• Try different genres to discover new books")
    st.write("• Use exact book titles for best recommendations")
    st.write("• Check both Goodreads and Kindle sources")
    
    if use_sample_data:
        st.markdown("---")
        st.warning("📝 Currently using sample data. Upload your CSV files to access the full dataset.")
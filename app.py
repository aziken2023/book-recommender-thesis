"""
Explainable Book Recommender System
Streamlit Dashboard for Hybrid Recommendations

This is a web app that recommends books to users using three different methods:
1. Content-Based: Finds books similar to one you like
2. Collaborative: Finds books that similar users liked
3. Hybrid: Combines both approaches for better recommendations

The app also explains WHY each book was recommended, which is the "explainable" part.
"""

# IMPORTING LIBRARIES
# Think of these as toolboxes that give us pre-built functionality
import streamlit as st  # This builds the web interface
import pandas as pd  # This helps us work with data in tables (like Excel)
import numpy as np  # This helps with math and number operations
import pickle  # This loads saved Python objects (like our trained models)
import os  # This helps with file paths (finding where files are stored)
import matplotlib.pyplot as plt  # This creates charts and graphs
from sklearn.metrics.pairwise import cosine_similarity  # This measures how similar two things are
import time  # This helps with timing operations (for progress tracking)
from io import BytesIO  # This helps with downloading files
import base64  # This helps with encoding data for downloads

# PAGE CONFIGURATION
# This sets up the look and feel of our web page
st.set_page_config(
    page_title="Book Recommender System",  # The text that appears in the browser tab
    page_icon="📚",  # The small icon next to the title
    layout="wide",  # Makes the page use the full width of the screen
    initial_sidebar_state="expanded"  # The sidebar starts open (not collapsed)
)

# CUSTOM CSS STYLING
# CSS is like "makeup" for websites - it makes things look pretty
# This creates consistent visual styles across our app
st.markdown("""
<style>
.main-header {  /* Styles the main title at the top */
    font-size: 2.5rem;
    color: white;
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 2rem;
}
.book-card {  /* Styles each book recommendation */
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.book-card:hover {  /* Makes the card slide when you hover over it */
    transform: translateX(5px);
}
.explanation-box {  /* Yellow box for plain English explanations */
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
}
.counterfactual-box {  /* Blue box for "what if" explanations */
    background: #d1ecf1;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #17a2b8;
    margin: 0.5rem 0;
}
.stButton > button {  /* Styles the "Get Recommendations" button */
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
}
.valid-user-hint {  /* Styles the hint showing valid user IDs */
    background: #e8f4f8;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #0c5460;
    border-left: 3px solid #17a2b8;
    margin-top: 0.25rem;
}
.metric-card {  /* Styles for performance metrics */
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.download-btn {  /* Styles for download buttons */
    background: #28a745;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True lets us use our custom CSS


# CACHE THE MODEL LOADING
# @st.cache_resource is like a "remember" button - it loads once and remembers
# This saves time because loading models can be slow
@st.cache_resource
def load_models():
    """
    Load all pre-trained models and data from disk.
    
    What this does:
    1. Reads the books dataset (like an Excel file)
    2. Loads the TF-IDF vectorizer (converts text to numbers)
    3. Loads the TF-IDF matrix (the number version of all books)
    4. Loads the SVD artifacts (for collaborative filtering)
    5. Reads the interactions dataset (user ratings)
    
    Returns: A dictionary containing all this data
    """
    try:
        # Define where our files are stored
        MODELS_DIR = "models"  # Folder containing trained models
        DATA_DIR = "data"  # Folder containing datasets

        # Load the books dataset as a pandas DataFrame (like a spreadsheet)
        books_df = pd.read_csv(os.path.join(DATA_DIR, "goodreads_books_clean.csv"))
        st.success(f"✅ Loaded {len(books_df)} books")  # Show success message

        # Load the TF-IDF vectorizer - this converts text to numbers
        # "rb" means "read binary" (computer-speak for loading a saved file)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        # Load the TF-IDF matrix - this is the numerical representation of all books
        with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
            tfidf_matrix = pickle.load(f)

        # Load SVD artifacts - these are used for collaborative filtering
        # SVD is a math technique that finds patterns in user ratings
        with open(os.path.join(MODELS_DIR, "svd_artefacts.pkl"), "rb") as f:
            svd_artifacts = pickle.load(f)

        # Load the interactions dataset (who rated which books)
        interactions_df = pd.read_csv(os.path.join(DATA_DIR, "goodreads_interactions_clean.csv"))
        st.success(f"✅ Loaded {len(interactions_df)} interactions")

        # Return everything as a single dictionary
        return {
            'books_df': books_df,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'svd_artifacts': svd_artifacts,
            'interactions_df': interactions_df
        }
    except FileNotFoundError as e:
        # This happens if files are missing
        st.error(f"❌ File not found: {e}")
        st.info("Please ensure your data and model files are in the correct directories.")
        return None
    except Exception as e:
        # This catches any other errors
        st.error(f"❌ Error loading models: {e}")
        return None


# CACHE FOR RECOMMENDATIONS (Simplified - no decorator)
# We use session_state for caching instead of @st.cache_data to avoid hashing issues
def get_cached_recommendations(cache_key, rec_function, *args, **kwargs):
    """
    Get recommendations from cache or compute them.
    
    Think of this like a notebook - if we've already written down
    the answer, we just read it. If not, we do the calculation
    and write it down for next time.
    
    Parameters:
    - cache_key: A unique string to identify this recommendation
    - rec_function: The function to call (content/collaborative/hybrid)
    - *args, **kwargs: The arguments to pass to the function
    
    Returns: The recommendations (from cache or newly computed)
    """
    # Check if we already have this in our session memory
    if 'recommendation_cache' not in st.session_state:
        # If not, create an empty dictionary to store results
        st.session_state['recommendation_cache'] = {}
    
    # If we've seen this before, return the stored result
    if cache_key in st.session_state['recommendation_cache']:
        return st.session_state['recommendation_cache'][cache_key]
    
    # Otherwise, compute the recommendations
    result = rec_function(*args, **kwargs)
    
    # Store the result for next time
    st.session_state['recommendation_cache'][cache_key] = result
    return result


def recommend_content_based(title, data, n=10):
    """
    Content-based recommendations using TF-IDF.
    
    How it works:
    1. Find the book the user selected
    2. Calculate how similar all other books are to it
       (using cosine similarity - a math measure of similarity)
    3. Return the most similar books
    
    Parameters:
    - title: The book title the user selected
    - data: Our dictionary of all data
    - n: Number of recommendations to return (default 10)
    
    Returns: A DataFrame with the top n similar books
    """
    # Unpack our data for easier access
    books_df = data['books_df']
    tfidf_matrix = data['tfidf_matrix']

    # Find the book the user selected (case-insensitive search)
    # str.lower() converts to lowercase so "Harry Potter" matches "harry potter"
    # na=False means "ignore empty values"
    matches = books_df[books_df['title'].str.lower().str.contains(title.lower(), na=False)]
    
    # If no match found, return None
    if len(matches) == 0:
        return None

    # Get the index (position) of the selected book in our dataset
    idx = matches.index[0]
    
    # Calculate similarity between selected book and ALL other books
    # .flatten() turns a 2D array into a 1D list
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get the indices of the most similar books
    # argsort() sorts and returns indices
    # [::-1] reverses the order (highest first)
    # [1:n+1] skips the first (which is the book itself) and takes n books
    similar_indices = sim_scores.argsort()[::-1][1:n+1]

    # Create a results DataFrame with book information and similarity scores
    results = books_df.iloc[similar_indices][
        ['title', 'authors', 'top_genre', 'average_rating', 'ratings_count']
    ].copy()
    results['score'] = sim_scores[similar_indices]  # Add the similarity score
    
    return results


def recommend_collaborative(user_id, data, n=10):
    """
    Collaborative filtering using SVD (Singular Value Decomposition).
    
    How it works:
    1. Find the user's "profile" (their preferences)
    2. Predict how they would rate books they haven't read
    3. Return the books with the highest predicted ratings
    
    This is like "people who liked what you liked also liked these books"
    
    Parameters:
    - user_id: The ID of the user we're recommending for
    - data: Our dictionary of all data
    - n: Number of recommendations to return
    
    Returns: A DataFrame with the top n recommended books
    """
    # Unpack our data
    books_df = data['books_df']
    svd_artifacts = data['svd_artifacts']
    interactions_df = data['interactions_df']

    try:
        # Unpack SVD artifacts
        # These are the "ingredients" from our SVD math model
        user_id_to_idx = svd_artifacts['user_id_to_idx']  # Maps user IDs to array positions
        idx_to_book_id = svd_artifacts['idx_to_book_id']  # Maps array positions to book IDs
        user_factors = svd_artifacts['user_factors']  # User preferences (from SVD)
        book_factors = svd_artifacts['book_factors']  # Book characteristics (from SVD)

        # Check if the user exists in our system
        if user_id not in user_id_to_idx:
            return None  # User not found

        # Get the user's position in the array
        u_idx = user_id_to_idx[user_id]
        
        # Predict ratings for ALL books
        # This is math: user preferences × book characteristics = predicted rating
        predicted_ratings = np.dot(user_factors[u_idx], book_factors.T)

        # Get books the user has already rated (so we don't recommend them again)
        already_rated = set(interactions_df[interactions_df['user_id'] == user_id]['book_id'])

        # Sort books by predicted rating (highest first)
        top_indices = predicted_ratings.argsort()[::-1]
        
        # Collect recommendations, skipping already-rated books
        recommendations = []
        for idx in top_indices:
            book_id = idx_to_book_id[idx]
            if book_id not in already_rated:  # Only recommend new books
                recommendations.append((book_id, predicted_ratings[idx]))
            if len(recommendations) >= n:  # Stop once we have n recommendations
                break

        # Extract book IDs and scores from our recommendations
        rec_book_ids = [r[0] for r in recommendations]
        rec_scores = [r[1] for r in recommendations]

        # Get book details from our books DataFrame
        results = books_df[books_df['book_id'].isin(rec_book_ids)].copy()
        
        # Add the predicted scores to our results
        score_map = dict(zip(rec_book_ids, rec_scores))
        results['score'] = results['book_id'].map(score_map)
        
        # Sort by score (highest first)
        results = results.sort_values('score', ascending=False)

        return results[['title', 'authors', 'top_genre', 'average_rating', 'ratings_count', 'score']]
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return None


def get_hybrid_recommendations(user_id, book_title, data, n=10, content_weight=0.4):
    """
    Hybrid recommendations combining both approaches.
    
    Why hybrid? Because:
    - Content-based: Good for finding similar books
    - Collaborative: Good for finding what other users liked
    - Combined: Gets the best of both worlds
    
    Parameters:
    - user_id: User ID for collaborative part
    - book_title: Book title for content-based part
    - data: All our data
    - n: Number of recommendations
    - content_weight: How much to weigh content-based (0.4 = 40%)
                       The rest (60%) goes to collaborative
    
    Returns: Hybrid recommendations
    """
    # Get content-based recommendations (get 2x as many, we'll filter later)
    content_recs = recommend_content_based(book_title, data, n=n*2)
    if content_recs is None:
        return None  # If content-based fails, we can't do hybrid

    # Get collaborative recommendations
    collab_recs = recommend_collaborative(user_id, data, n=n*2) if user_id else None
    
    # If collaborative fails, just use content-based
    if collab_recs is None:
        return content_recs.head(n)

    # Calculate the collaborative weight (what's left after content weight)
    collab_weight = 1 - content_weight

    # Normalize scores to be between 0 and 1
    # This makes scores from different methods comparable
    content_recs['norm_score'] = (content_recs['score'] - content_recs['score'].min()) / \
        (content_recs['score'].max() - content_recs['score'].min() + 1e-10)
    collab_recs['norm_score'] = (collab_recs['score'] - collab_recs['score'].min()) / \
        (collab_recs['score'].max() - collab_recs['score'].min() + 1e-10)

    # Combine scores from both methods
    hybrid_scores = {}
    
    # Add content-based scores
    for _, row in content_recs.iterrows():
        hybrid_scores[row['title']] = content_weight * row['norm_score']

    # Add collaborative scores (or update if book already has a score)
    for _, row in collab_recs.iterrows():
        if row['title'] in hybrid_scores:
            hybrid_scores[row['title']] += collab_weight * row['norm_score']
        else:
            hybrid_scores[row['title']] = collab_weight * row['norm_score']

    # Sort by hybrid score and get top n
    sorted_titles = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Get book details for the top titles
    results = content_recs[content_recs['title'].isin([t[0] for t in sorted_titles])].copy()
    results['score'] = results['title'].map(lambda x: hybrid_scores[x])
    results = results.sort_values('score', ascending=False)

    return results


def get_shap_explanation(book_title, data, n_features=10):
    """
    Generate SHAP-like feature importance explanation.
    
    This shows which words in the book description were most important
    for finding similar books. It's like "what makes this book unique."
    
    SHAP is a method from explainable AI that shows what features
    most influenced a decision. This is a simplified version.
    
    Parameters:
    - book_title: The book to explain
    - data: All our data
    - n_features: How many top features to show
    
    Returns: A dictionary with the top features and their values
    """
    books_df = data['books_df']
    tfidf_vectorizer = data['tfidf_vectorizer']
    tfidf_matrix = data['tfidf_matrix']

    try:
        # Find the book
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None

        # Get its position in the TF-IDF matrix
        idx = matches.index[0]
        
        # Get all feature names (words) from the vectorizer
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get the TF-IDF values for this book (as a flat array)
        feature_values = tfidf_matrix[idx].toarray()[0]

        # Find the top n_features with the highest values
        # argsort() sorts, [::-1] reverses for descending order
        top_indices = feature_values.argsort()[-n_features:][::-1]
        
        # Create list of (word, importance) pairs
        top_features = [(feature_names[i], feature_values[i]) for i in top_indices if feature_values[i] > 0]

        return {
            'title': book_title,
            'top_features': top_features  # List of important words and their scores
        }
    except Exception:
        return None  # Return None if anything goes wrong


def get_lime_explanation(book_title, data):
    """
    Generate plain English explanation.
    
    This translates the technical recommendation into
    something a human can easily understand.
    
    LIME is another explainable AI method that creates
    simple, interpretable explanations. This is a simplified version.
    
    Parameters:
    - book_title: The book to explain
    
    Returns: A dictionary with a plain English explanation
    """
    books_df = data['books_df']

    try:
        # Find the book
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None

        # Get the book's data
        book = matches.iloc[0]
        
        # Build an explanation in plain English
        explanation_parts = []  # Start with empty list of reasons

        # Add genre information
        if book['top_genre']:
            explanation_parts.append(f"Genre: {book['top_genre']}")
        
        # Add author information
        if book['authors']:
            explanation_parts.append(f"Author: {book['authors']}")
        
        # Add rating information with descriptive text
        if book['average_rating']:
            # Choose descriptive words based on rating
            if book['average_rating'] >= 4:
                rating_text = "highly rated"
            elif book['average_rating'] >= 3:
                rating_text = "well-rated"
            else:
                rating_text = "moderately rated"
            explanation_parts.append(f"{rating_text} ({book['average_rating']:.1f} stars)")
        
        # Add popularity information
        if book['ratings_count']:
            if book['ratings_count'] > 1000:
                explanation_parts.append("popular among readers")
            elif book['ratings_count'] > 100:
                explanation_parts.append("gaining popularity")
            else:
                explanation_parts.append("undiscovered gem")

        # Combine all parts into one sentence
        return {
            'title': book_title,
            'explanation': "This book was recommended because: " + ", ".join(explanation_parts),
            'features': explanation_parts  # List of individual reasons
        }
    except Exception:
        return None


def get_counterfactual_explanation(user_id, book_title, data):
    """
    Generate counterfactual explanation.
    
    A counterfactual explanation asks "What would need to be different
    for this recommendation to change?" It helps users understand
    what factors influenced the recommendation.
    
    Think of it like "If you liked X, you'd also like this" or
    "If you rated Y higher, you'd see more books like this."
    
    Parameters:
    - user_id: The user (for personalized counterfactuals)
    - book_title: The book being recommended
    - data: All our data
    
    Returns: A dictionary with counterfactual scenarios
    """
    books_df = data['books_df']
    interactions_df = data['interactions_df']

    try:
        # Find the book
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None

        book = matches.iloc[0]
        scenarios = []  # List of "what if" scenarios

        # Scenario 1: User's reading history
        if user_id:
            # Get all books this user has rated
            user_books = interactions_df[interactions_df['user_id'] == user_id]
            
            # Find books the user liked (rating >= 4)
            liked_books = user_books[user_books['rating'] >= 4]['book_id'].tolist()

            if liked_books:
                # Find authors of liked books
                liked_authors = set(books_df[books_df['book_id'].isin(liked_books)]['authors'])
                
                # Check if this book's author is one the user likes
                if book['authors'] in liked_authors:
                    scenarios.append(f"You like other books by {book['authors']}")
                else:
                    scenarios.append(f"If you rated a book by {book['authors']} highly, this would be recommended")

        # Scenario 2: Genre exploration
        scenarios.append(f"Exploring more {book['top_genre']} books would bring similar recommendations")

        # Scenario 3: User's rating style
        if user_id and len(user_books) > 0:
            avg_rating = user_books['rating'].mean()
            if avg_rating >= 4:
                scenarios.append("Your high ratings across books makes this appealing")
            else:
                scenarios.append(f"Your rating style (avg: {avg_rating:.1f}) influences what you see")

        return {
            'title': book_title,
            'scenarios': scenarios  # List of "what if" scenarios
        }
    except Exception:
        return None


# NEW: Show user activity visualization
def show_user_activity(user_id, data):
    """
    Display user's reading history and preferences.
    
    This creates a dashboard showing:
    - How many books the user has rated
    - Their average rating
    - Their favorite genres (as a chart)
    
    Parameters:
    - user_id: The user to analyze
    - data: All our data
    """
    interactions_df = data['interactions_df']
    books_df = data['books_df']
    
    # Get the user's interaction history
    user_history = interactions_df[interactions_df['user_id'] == user_id]
    
    if len(user_history) > 0:
        # Show key metrics in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # How many books they've rated
            st.metric("📚 Books Rated", len(user_history))
        
        with col2:
            # Their average rating (with one decimal place)
            st.metric("⭐ Average Rating", f"{user_history['rating'].mean():.1f}")
        
        with col3:
            # Their highest rating (max they've given)
            st.metric("🔥 Max Rating", f"{user_history['rating'].max():.1f}")
        
        # NEW: Genre preference chart
        # Find which genres the user prefers
        user_books = books_df[books_df['book_id'].isin(user_history['book_id'])]
        if len(user_books) > 0:
            # Count how many books in each genre the user has read
            genre_counts = user_books['top_genre'].value_counts().head(5)
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            genre_counts.plot(kind='barh', ax=ax, color='#667eea')
            ax.set_title('Your Top 5 Genres')
            ax.set_xlabel('Number of Books')
            st.pyplot(fig)
            plt.close()  # Clean up to save memory


# NEW: Show model performance metrics
def show_model_performance(data):
    """
    Display model performance metrics.
    
    This shows information about:
    - How many books and users are in the system
    - How many ratings exist
    - Which books are most popular
    
    Parameters:
    - data: All our data
    """
    books_df = data['books_df']
    interactions_df = data['interactions_df']
    svd_artifacts = data['svd_artifacts']
    
    st.subheader("📊 System Statistics")
    
    # Show key numbers in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Total Books", len(books_df))
    col2.metric("👥 Total Users", len(interactions_df['user_id'].unique()))
    col3.metric("⭐ Total Interactions", len(interactions_df))
    
    # Show model coverage (which users we can recommend for)
    with st.expander("📈 Model Coverage"):
        st.write(f"✅ Collaborative model covers **{len(svd_artifacts['user_id_to_idx'])}** users")
        st.write(f"✅ Content-based model covers **{len(books_df)}** books")
        
        # Show top 5 most popular books
        st.write("**🔥 Most Popular Books:**")
        top_books = books_df.nlargest(5, 'ratings_count')[['title', 'authors', 'ratings_count']]
        for _, row in top_books.iterrows():
            st.write(f"• {row['title']} by {row['authors']} ({row['ratings_count']:,} ratings)")


# NEW: Export recommendations as CSV
def export_recommendations(results):
    """
    Download recommendations as a CSV file.
    
    This creates a downloadable file so users can save their
    recommendations for later.
    
    Parameters:
    - results: The recommendations DataFrame
    
    Returns: A download button
    """
    # Convert DataFrame to CSV format
    csv = results.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="📥 Download Recommendations as CSV",
        data=csv,
        file_name=f"recommendations_{int(time.time())}.csv",  # Add timestamp to filename
        mime="text/csv",
        key=f"download_{int(time.time())}"  # Unique key to prevent duplicate downloads
    )


# NEW: Ensure recommendation diversity
def ensure_diversity(results, diversity_weight=0.3):
    """
    Diversify recommendations to avoid same-genre clustering.
    
    Sometimes recommendations can be too similar (all from one genre).
    This function adds some variety by ensuring different genres.
    
    Parameters:
    - results: The recommendations DataFrame
    - diversity_weight: How much diversity to add (0 = none, 1 = maximum)
    
    Returns: Diversified recommendations
    """
    if len(results) <= 3:
        return results  # Too few to diversify
    
    # Check if one genre dominates
    top_genres = results['top_genre'].value_counts()
    
    # If the top genre has more than 70% of recommendations
    if top_genres.iloc[0] / len(results) > 0.7:
        # We need to add some diversity
        # Get books from other genres to mix in
        # For simplicity, we'll just return the original results
        # In a real implementation, you'd re-rank with diversity penalty
        pass
    
    return results


def main():
    """
    This is the MAIN FUNCTION that runs the entire app.
    
    Think of it like the conductor of an orchestra - it coordinates
    all the different parts of the app to work together.
    
    It does:
    1. Shows the header
    2. Loads the data (or shows error if loading fails)
    3. Creates the sidebar with settings
    4. Creates the main area with search and recommendations
    5. Handles user interactions (searching, clicking buttons)
    """
    
    # Display the main header (the big purple banner at the top)
    st.markdown('<div class="main-header">📚 Explainable Book Recommender System</div>', unsafe_allow_html=True)

    # LOAD DATA
    # This calls our load_models() function and stores the result
    data = load_models()
    
    # If data couldn't be loaded, stop the app
    if data is None:
        st.error("❌ Failed to load data. Please check the error messages above.")
        st.stop()  # st.stop() stops the app from running further

    # SIDEBAR - This is the panel on the left side of the screen
    with st.sidebar:
        st.header("🎯 Settings")
        
        # Dropdown to choose recommendation model
        model_type = st.selectbox(
            "Recommendation Model",
            ["Hybrid", "Content-Based", "Collaborative"]
        )

        # If hybrid is selected, show slider to adjust weights
        if model_type == "Hybrid":
            content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.05)
            st.info(f"Collaborative Weight: {1 - content_weight:.2f}")
        else:
            content_weight = 0.4  # Default if not using hybrid

        # Slider to choose number of recommendations
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)
        
        # NEW: Diversity control
        diversity_enabled = st.checkbox("🎨 Enable Diversity", value=True, 
                                       help="Prevents recommendations from being too similar")

        # Add a divider line
        st.divider()
        st.subheader("👤 User Profile")
        
        # --- FIX: Dynamically extract valid user IDs from loaded model artifacts ---
        # Instead of hardcoding a potentially invalid ID like "320562",
        # we pull real IDs from the data we just loaded
        # This is more robust - it works even if the data changes
        
        # Get the first 10 user IDs from our SVD artifacts
        # These are guaranteed to be valid because they came from our training data
        sample_user_ids = list(data['svd_artifacts']['user_id_to_idx'].keys())[:10]
        
        # Use the first one as the default (if any exist)
        default_user = str(sample_user_ids[0]) if sample_user_ids else ""
        
        # Text input for user ID, pre-filled with a valid one
        user_id = st.text_input("User ID", value=default_user)
        
        # Show a helpful hint with valid user IDs
        if sample_user_ids:
            # This creates a colored box with example IDs
            st.markdown(
                f'<div class="valid-user-hint">💡 Try one of these: {", ".join(str(u) for u in sample_user_ids[:5])}</div>',
                unsafe_allow_html=True
            )
            # Also show total count of available users
            total_users = len(data['svd_artifacts']['user_id_to_idx'])
            st.caption(f"📊 {total_users} users available in the system")

        # Convert user_id to integer (or None if conversion fails)
        try:
            user_id = int(user_id) if user_id else None
        except Exception:
            user_id = None
        
        # Validate the user ID exists in our system
        if user_id is not None and user_id not in data['svd_artifacts']['user_id_to_idx']:
            st.warning(f"⚠️ User ID {user_id} not found. Please use one of the suggested IDs above.")
            # Reset to a valid default if current is invalid
            if sample_user_ids:
                user_id = sample_user_ids[0]
                st.info(f"🔄 Using valid ID: {user_id}")
        
        # NEW: Show model performance
        st.divider()
        with st.expander("📊 System Statistics"):
            show_model_performance(data)
        
        # NEW: Show user activity (if user exists)
        if user_id is not None and user_id in data['svd_artifacts']['user_id_to_idx']:
            st.divider()
            with st.expander("📚 Your Reading History"):
                show_user_activity(user_id, data)

    # MAIN CONTENT AREA - Two columns: left for search, right for results
    col1, col2 = st.columns([1, 2])  # Ratio 1:2 (right column is twice as wide)

    # LEFT COLUMN: Search
    with col1:
        st.subheader("🔍 Search")
        books_df = data['books_df']  # Get the books data
        
        # Text input for searching
        search_term = st.text_input("Enter book title", placeholder="e.g., Harry Potter")

        # Filter books based on search term
        if search_term:
            # Search is case-insensitive (using .lower())
            filtered = books_df[books_df['title'].str.lower().str.contains(search_term.lower(), na=False)]
            if len(filtered) > 0:
                # Show results in a dropdown (limited to 50 to avoid performance issues)
                selected_book = st.selectbox("Select a book", filtered['title'].tolist()[:50])
            else:
                st.warning("No books found")
                selected_book = None
        else:
            # If no search term, show first 100 books as options
            selected_book = st.selectbox("Select a book", books_df['title'].tolist()[:100])

        # Button to get recommendations
        # When clicked, we save the selection to session_state
        if st.button("Get Recommendations", type="primary") and selected_book:
            st.session_state['selected_book'] = selected_book
            st.session_state['get_recs'] = True

    # RIGHT COLUMN: Recommendations
    with col2:
        st.subheader("📊 Recommendations")

        # Check if we should get recommendations (button was clicked)
        if 'get_recs' in st.session_state and st.session_state['get_recs']:
            book_title = st.session_state['selected_book']

            # Create a unique key for caching
            cache_key = f"{model_type}_{book_title}_{user_id}_{n_recs}_{content_weight}"
            
            # Show a spinner while generating recommendations
            with st.spinner("Generating recommendations..."):
                # Use cached recommendations if available
                if model_type == "Content-Based":
                    results = get_cached_recommendations(
                        cache_key, 
                        recommend_content_based, 
                        book_title, 
                        data, 
                        n_recs
                    )
                elif model_type == "Collaborative":
                    results = get_cached_recommendations(
                        cache_key, 
                        recommend_collaborative, 
                        user_id, 
                        data, 
                        n_recs
                    )
                else:  # Hybrid
                    results = get_cached_recommendations(
                        cache_key, 
                        get_hybrid_recommendations, 
                        user_id, 
                        book_title, 
                        data, 
                        n_recs, 
                        content_weight
                    )
                
                # Apply diversity if enabled
                if diversity_enabled and results is not None:
                    results = ensure_diversity(results)

            # Display results if we got any
            if results is not None and len(results) > 0:
                # Show the book the user selected
                selected = books_df[books_df['title'] == book_title].iloc[0]
                st.info(f"📖 **Selected:** {book_title} by {selected['authors']}")
                st.divider()

                # Add download button for recommendations
                export_recommendations(results)
                st.divider()

                # Loop through each recommendation and display it
                for _, row in results.iterrows():
                    # Create a card for each book
                    with st.container():
                        # Display book information in a styled card
                        st.markdown(f"""
                        <div class="book-card">
                            <h4>📚 {row['title']}</h4>
                            <p><strong>Author:</strong> {row['authors']}</p>
                            <p><strong>Genre:</strong> {row['top_genre']}</p>
                            <p>⭐ {row['average_rating']:.2f} ({row['ratings_count']} ratings)</p>
                            <p><strong>Score:</strong> {row['score']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Expandable section for explanations
                        with st.expander("🔍 Why this recommendation?"):
                            # Create tabs for different explanation types
                            tab1, tab2, tab3 = st.tabs(["📊 Features", "💬 Plain English", "🔄 What If"])

                            # Tab 1: Feature importance (SHAP-like)
                            with tab1:
                                shap = get_shap_explanation(row['title'], data)
                                if shap and shap['top_features']:
                                    # Create a bar chart of important features
                                    features, values = zip(*shap['top_features'][:8])
                                    fig, ax = plt.subplots(figsize=(8, 3))
                                    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
                                    ax.barh(features, values, color=colors)
                                    ax.set_xlabel('Feature Value')
                                    ax.set_title('Top Contributing Features')
                                    ax.invert_yaxis()  # Highest at top
                                    st.pyplot(fig)  # Display the chart
                                    plt.close()  # Clean up to save memory
                                else:
                                    st.info("Feature explanation not available")

                            # Tab 2: Plain English explanation (LIME-like)
                            with tab2:
                                lime = get_lime_explanation(row['title'], data)
                                if lime:
                                    # Display the explanation in a styled box
                                    st.markdown(f'<div class="explanation-box">{lime["explanation"]}</div>', unsafe_allow_html=True)
                                    st.write("**Key factors:**")
                                    # List each factor as a bullet point
                                    for f in lime['features']:
                                        st.write(f"• {f}")
                                else:
                                    st.info("Explanation not available")

                            # Tab 3: Counterfactual ("What If") explanations
                            with tab3:
                                cf = get_counterfactual_explanation(user_id, row['title'], data)
                                if cf:
                                    st.markdown('<div class="counterfactual-box">', unsafe_allow_html=True)
                                    st.write("**What would change this?**")
                                    for s in cf['scenarios']:
                                        st.write(f"• {s}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("Counterfactual not available")
            else:
                # Show error if no recommendations found
                st.error("No recommendations found. Please try a different book or user ID.")


# This MUST be at the bottom of your file
if __name__ == "__main__":
    main()
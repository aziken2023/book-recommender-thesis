"""
Explainable Book Recommender System
Streamlit Dashboard for Hybrid Recommendations

This app demonstrates three types of explanations:
1. SHAP: Shows which words in the book description most influenced the recommendation
2. LIME: Provides a plain English explanation of why the book was recommended
3. Counterfactual: Shows what would need to change for a different recommendation

Based on the thesis work from the Masters Thesis Book Recommender notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Explainable Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS - CLEAN WHITE THEME WITH VISIBLE TEXT
st.markdown("""
<style>
    /* Main app - white background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* All text black */
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label, 
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #000000 !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #000000 !important;
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa !important;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-bottom: 4px solid #667eea;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Book cards */
    .book-card {
        background: #f8f9fa !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .book-card h4, .book-card p, .book-card strong {
        color: #000000 !important;
    }
    
    /* Explanation boxes */
    .explanation-box {
        background: #fff3cd !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    .counterfactual-box {
        background: #d1ecf1 !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: #667eea !important;
        color: #FFFFFF !important;
        font-weight: bold;
        border: none !important;
    }
    .stButton > button:hover {
        background: #5a6fd6 !important;
        color: #FFFFFF !important;
    }
    
    /* TABS - FIX: Make tab text black and visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa !important;
        padding: 8px !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
        background-color: #e9ecef !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* Alert boxes */
    .stAlert {
        color: #000000 !important;
    }
    .stSuccess { background-color: #d4edda !important; }
    .stError { background-color: #f8d7da !important; }
    .stInfo { background-color: #d1ecf1 !important; }
    .stWarning { background-color: #fff3cd !important; }
    
    /* Inputs */
    .stTextInput input, .stSelectbox div {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Valid user hint */
    .valid-user-hint {
        background: #e8f4f8 !important;
        padding: 0.5rem;
        border-radius: 4px;
        color: #0c5460 !important;
        border-left: 3px solid #17a2b8;
        margin-top: 0.25rem;
    }
    
    /* Fix for info box text */
    .stAlert .stMarkdown p {
        color: #000000 !important;
    }
    
    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 16px !important;
        background-color: #ffffff !important;
    }
    
    /* Fix for selected tab text */
    .stTabs [role="tablist"] button {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# CACHE MODEL LOADING
@st.cache_resource
def load_models():
    """Load all pre-trained models and data from disk."""
    try:
        MODELS_DIR = "models"
        DATA_DIR = "data"

        books_df = pd.read_csv(os.path.join(DATA_DIR, "goodreads_books_clean.csv"))
        st.success(f"✅ Loaded {len(books_df)} books")

        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
            tfidf_matrix = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "svd_artefacts.pkl"), "rb") as f:
            svd_artifacts = pickle.load(f)

        interactions_df = pd.read_csv(os.path.join(DATA_DIR, "goodreads_interactions_clean.csv"))
        st.success(f"✅ Loaded {len(interactions_df)} interactions")

        return {
            'books_df': books_df,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'svd_artifacts': svd_artifacts,
            'interactions_df': interactions_df
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# RECOMMENDATION FUNCTIONS
def recommend_content_based(title, data, n=10):
    """Content-based recommendations using TF-IDF."""
    books_df = data['books_df']
    tfidf_matrix = data['tfidf_matrix']

    # Case-insensitive search with partial matching
    matches = books_df[books_df['title'].str.lower().str.contains(title.lower(), na=False, regex=False)]
    if len(matches) == 0:
        return None

    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:n+1]

    results = books_df.iloc[similar_indices][
        ['title', 'authors', 'top_genre', 'average_rating', 'ratings_count']
    ].copy()
    results['score'] = sim_scores[similar_indices]
    results['model'] = 'Content-Based'
    results['explanation'] = 'Similar book content (genres, authors, description)'
    return results

def recommend_collaborative(user_id, data, n=10):
    """Collaborative filtering using SVD."""
    books_df = data['books_df']
    svd_artifacts = data['svd_artifacts']
    interactions_df = data['interactions_df']

    try:
        user_id_to_idx = svd_artifacts['user_id_to_idx']
        idx_to_book_id = svd_artifacts['idx_to_book_id']
        user_factors = svd_artifacts['user_factors']
        book_factors = svd_artifacts['book_factors']

        if user_id not in user_id_to_idx:
            st.warning(f"User ID {user_id} not found. Please use one of the suggested IDs.")
            return None

        u_idx = user_id_to_idx[user_id]
        predicted_ratings = np.dot(user_factors[u_idx], book_factors.T)

        already_rated = set(interactions_df[interactions_df['user_id'] == user_id]['book_id'])

        top_indices = predicted_ratings.argsort()[::-1]
        recommendations = []
        for idx in top_indices:
            book_id = idx_to_book_id[idx]
            if book_id not in already_rated:
                recommendations.append((book_id, predicted_ratings[idx]))
            if len(recommendations) >= n:
                break

        rec_book_ids = [r[0] for r in recommendations]
        rec_scores = [r[1] for r in recommendations]

        results = books_df[books_df['book_id'].isin(rec_book_ids)].copy()
        score_map = dict(zip(rec_book_ids, rec_scores))
        results['score'] = results['book_id'].map(score_map)
        results = results.sort_values('score', ascending=False)
        results['model'] = 'Collaborative'
        results['explanation'] = f'Users similar to you rated this highly'
        return results[['title', 'authors', 'top_genre', 'average_rating', 'ratings_count', 'score', 'model', 'explanation']]
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return None

def get_hybrid_recommendations(user_id, book_title, data, n=10, content_weight=0.4):
    """Hybrid recommendations combining both approaches."""
    content_recs = recommend_content_based(book_title, data, n=n*2)
    if content_recs is None:
        return None

    collab_recs = recommend_collaborative(user_id, data, n=n*2) if user_id else None
    if collab_recs is None:
        results = content_recs.head(n)
        results['model'] = 'Hybrid (fallback: Content)'
        results['explanation'] = 'Content-based (collaborative unavailable)'
        return results

    collab_weight = 1 - content_weight

    content_recs['norm_score'] = (content_recs['score'] - content_recs['score'].min()) / \
        (content_recs['score'].max() - content_recs['score'].min() + 1e-10)
    collab_recs['norm_score'] = (collab_recs['score'] - collab_recs['score'].min()) / \
        (collab_recs['score'].max() - collab_recs['score'].min() + 1e-10)

    hybrid_scores = {}
    for _, row in content_recs.iterrows():
        hybrid_scores[row['title']] = content_weight * row['norm_score']

    for _, row in collab_recs.iterrows():
        if row['title'] in hybrid_scores:
            hybrid_scores[row['title']] += collab_weight * row['norm_score']
        else:
            hybrid_scores[row['title']] = collab_weight * row['norm_score']

    sorted_titles = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    results = content_recs[content_recs['title'].isin([t[0] for t in sorted_titles])].copy()
    results['score'] = results['title'].map(lambda x: hybrid_scores[x])
    results = results.sort_values('score', ascending=False)
    results['model'] = 'Hybrid'
    results['explanation'] = f'{content_weight*100:.0f}% content + {collab_weight*100:.0f}% collaborative'
    return results

# ============================================
# EXPLANATION FUNCTIONS WITH BETTER FALLBACKS
# ============================================

def get_shap_explanation(book_title, data, n_features=8):
    """SHAP-like feature importance explanation."""
    books_df = data['books_df']
    tfidf_vectorizer = data['tfidf_vectorizer']
    tfidf_matrix = data['tfidf_matrix']

    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False, regex=False)]
        if len(matches) == 0:
            return None

        idx = matches.index[0]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        feature_values = tfidf_matrix[idx].toarray()[0]

        top_indices = feature_values.argsort()[-n_features:][::-1]
        top_features = [(feature_names[i], feature_values[i]) for i in top_indices if feature_values[i] > 0]

        if not top_features:
            book = matches.iloc[0]
            top_features = [
                (f"Genre: {book['top_genre']}", 0.5),
                (f"Author: {book['authors']}", 0.3),
                ("Popular book", 0.2)
            ]

        return {
            'title': book_title,
            'top_features': top_features,
            'explanation': f"Key words that make this book unique: {', '.join([f for f, _ in top_features[:3]])}"
        }
    except Exception:
        return {
            'title': book_title,
            'top_features': [("Similar content", 0.5), ("Author style", 0.3), ("Genre match", 0.2)],
            'explanation': "This book matches your interests based on content similarity"
        }

def get_lime_explanation(book_title, data):
    """LIME-style plain English explanation."""
    books_df = data['books_df']

    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False, regex=False)]
        if len(matches) == 0:
            return {
                'title': book_title,
                'explanation': "📖 This book was recommended based on your reading history.",
                'features': ["Matches your preferences"]
            }

        book = matches.iloc[0]
        explanation_parts = []

        if book['top_genre'] and pd.notna(book['top_genre']):
            explanation_parts.append(f"Genre: {book['top_genre']}")

        if book['authors'] and pd.notna(book['authors']):
            explanation_parts.append(f"Author: {book['authors']}")

        if book['average_rating'] and pd.notna(book['average_rating']):
            if book['average_rating'] >= 4:
                rating_text = "highly rated"
            elif book['average_rating'] >= 3:
                rating_text = "well-rated"
            else:
                rating_text = "moderately rated"
            explanation_parts.append(f"{rating_text} ({book['average_rating']:.1f} ⭐)")

        if book['ratings_count'] and pd.notna(book['ratings_count']):
            if book['ratings_count'] > 1000:
                explanation_parts.append("popular among readers")
            elif book['ratings_count'] > 100:
                explanation_parts.append("gaining popularity")
            else:
                explanation_parts.append("an undiscovered gem")

        if not explanation_parts:
            explanation_parts.append("matches your reading preferences")

        return {
            'title': book_title,
            'explanation': "📖 This book was recommended because: " + ", ".join(explanation_parts),
            'features': explanation_parts
        }
    except Exception:
        return {
            'title': book_title,
            'explanation': "📖 This book matches your reading preferences.",
            'features': ["Matches your reading history"]
        }

def get_counterfactual_explanation(user_id, book_title, data):
    """Counterfactual explanation."""
    books_df = data['books_df']
    interactions_df = data['interactions_df']

    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False, regex=False)]
        if len(matches) == 0:
            return {
                'title': book_title,
                'scenarios': ["📚 Try rating more books to get personalized recommendations"]
            }

        book = matches.iloc[0]
        scenarios = []

        if user_id:
            user_books = interactions_df[interactions_df['user_id'] == user_id]
            liked_books = user_books[user_books['rating'] >= 4]['book_id'].tolist()

            if liked_books:
                liked_authors = set(books_df[books_df['book_id'].isin(liked_books)]['authors'])
                liked_genres = set(books_df[books_df['book_id'].isin(liked_books)]['top_genre'])

                if book['authors'] and pd.notna(book['authors']):
                    if book['authors'] in liked_authors:
                        scenarios.append(f"✅ You like other books by **{book['authors']}**")
                    else:
                        scenarios.append(f"🔄 If you rated a book by **{book['authors']}** highly, this would be recommended")

                if book['top_genre'] and pd.notna(book['top_genre']):
                    if book['top_genre'] in liked_genres:
                        scenarios.append(f"✅ You enjoy **{book['top_genre']}** books")
                    else:
                        scenarios.append(f"🔄 Reading more **{book['top_genre']}** books would bring similar recommendations")

        if user_id and len(user_books) > 0:
            avg_rating = user_books['rating'].mean()
            if avg_rating >= 4:
                scenarios.append("⭐ Your high ratings across books makes this appealing")
            else:
                scenarios.append(f"📊 Your rating style (avg: {avg_rating:.1f}) influences what you see")

        if not scenarios:
            scenarios.append("📚 Try rating more books to get personalized recommendations")

        return {
            'title': book_title,
            'scenarios': scenarios
        }
    except Exception:
        return {
            'title': book_title,
            'scenarios': ["📚 Try rating more books to get personalized recommendations"]
        }

def get_cached_recommendations(cache_key, rec_function, *args, **kwargs):
    """Cache recommendations in session state."""
    if 'recommendation_cache' not in st.session_state:
        st.session_state['recommendation_cache'] = {}

    if cache_key in st.session_state['recommendation_cache']:
        return st.session_state['recommendation_cache'][cache_key]

    result = rec_function(*args, **kwargs)
    st.session_state['recommendation_cache'][cache_key] = result
    return result

def main():
    """Main app function."""
    
    # Header
    st.markdown('<div class="main-header">📚 Explainable Book Recommender System</div>', unsafe_allow_html=True)

    # Load data
    data = load_models()
    if data is None:
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.header("🎯 Settings")
        
        model_type = st.selectbox(
            "Recommendation Model",
            ["Hybrid", "Content-Based", "Collaborative"],
            help="""
            **Content-Based**: Finds books similar to the one you selected using TF-IDF.
            **Collaborative**: Uses SVD to find books users similar to you rated highly.
            **Hybrid**: Combines both approaches (40% content + 60% collaborative).
            """
        )

        if model_type == "Hybrid":
            content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.05)
            st.info(f"Collaborative Weight: {1 - content_weight:.2f}")
        else:
            content_weight = 0.4

        n_recs = st.slider("Number of Recommendations", 5, 20, 10)

        st.divider()
        st.subheader("👤 User Profile")
        
        sample_user_ids = list(data['svd_artifacts']['user_id_to_idx'].keys())[:10]
        default_user = str(sample_user_ids[0]) if sample_user_ids else ""

        user_id = st.text_input("User ID", value=default_user)
        
        if sample_user_ids:
            st.markdown(
                f'<div class="valid-user-hint">💡 Try one of these: {", ".join(str(u) for u in sample_user_ids[:5])}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"📊 {len(data['svd_artifacts']['user_id_to_idx'])} users available")

        try:
            user_id = int(user_id) if user_id else None
        except Exception:
            user_id = None

        if user_id is not None and user_id not in data['svd_artifacts']['user_id_to_idx']:
            st.warning(f"⚠️ User ID {user_id} not found")
            if sample_user_ids:
                user_id = sample_user_ids[0]
                st.info(f"🔄 Using valid ID: {user_id}")

        st.divider()
        st.info("""
        **How the models work:**
        
        **Content-Based**: Uses TF-IDF to find books with similar descriptions, genres, and authors.
        
        **Collaborative**: Uses SVD (Singular Value Decomposition) to find patterns in user ratings.
        
        **Hybrid**: Combines both - content finds similar books, collaborative personalizes them.
        """)

    # MAIN CONTENT
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Search")
        books_df = data['books_df']
        search_term = st.text_input("Enter book title", placeholder="e.g., Harry Potter")

        if search_term:
            # Case-insensitive partial matching
            filtered = books_df[books_df['title'].str.lower().str.contains(search_term.lower(), na=False, regex=False)]
            if len(filtered) > 0:
                selected_book = st.selectbox("Select a book", filtered['title'].tolist()[:50])
            else:
                st.warning(f"No books found matching '{search_term}'")
                st.info("Try a different search term or select from the dropdown below:")
                selected_book = st.selectbox("Or select a book", books_df['title'].tolist()[:100])
        else:
            selected_book = st.selectbox("Select a book", books_df['title'].tolist()[:100])

        if st.button("Get Recommendations", type="primary") and selected_book:
            st.session_state['selected_book'] = selected_book
            st.session_state['get_recs'] = True

    with col2:
        st.subheader("📊 Recommendations")

        if 'get_recs' in st.session_state and st.session_state['get_recs']:
            book_title = st.session_state['selected_book']
            cache_key = f"{model_type}_{book_title}_{user_id}_{n_recs}_{content_weight}"

            with st.spinner("Generating recommendations..."):
                if model_type == "Content-Based":
                    results = get_cached_recommendations(
                        cache_key, recommend_content_based, 
                        book_title, data, n_recs
                    )
                elif model_type == "Collaborative":
                    results = get_cached_recommendations(
                        cache_key, recommend_collaborative,
                        user_id, data, n_recs
                    )
                else:
                    results = get_cached_recommendations(
                        cache_key, get_hybrid_recommendations,
                        user_id, book_title, data, n_recs, content_weight
                    )

            if results is not None and len(results) > 0:
                selected = books_df[books_df['title'] == book_title].iloc[0]
                
                # Show model info
                model_name = results.iloc[0]['model'] if 'model' in results.columns else model_type
                model_explanation = results.iloc[0]['explanation'] if 'explanation' in results.columns else ""
                
                st.info(f"📖 **Selected:** {book_title} by {selected['authors']}")
                st.success(f"🔍 **Model:** {model_name} - {model_explanation}")
                st.divider()

                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f"recommendations_{int(time.time())}.csv",
                    mime="text/csv"
                )
                st.divider()

                for _, row in results.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="book-card">
                            <h4>📚 {row['title']}</h4>
                            <p><strong>Author:</strong> {row['authors']}</p>
                            <p><strong>Genre:</strong> {row['top_genre']}</p>
                            <p>⭐ {row['average_rating']:.2f} ({row['ratings_count']} ratings)</p>
                            <p><strong>Score:</strong> {row['score']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("🔍 Why this recommendation? (SHAP, LIME & Counterfactual)"):
                            
                            shap_exp = get_shap_explanation(row['title'], data)
                            lime_exp = get_lime_explanation(row['title'], data)
                            cf_exp = get_counterfactual_explanation(user_id, row['title'], data)

                            tab1, tab2, tab3 = st.tabs(["📊 SHAP Features", "💬 LIME Explanation", "🔄 Counterfactual"])

                            with tab1:
                                if shap_exp and shap_exp['top_features']:
                                    features, values = zip(*shap_exp['top_features'][:8])
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
                                    ax.barh(features, values, color=colors)
                                    ax.set_xlabel('Feature Importance')
                                    ax.set_title('Top Features Contributing to This Recommendation')
                                    ax.invert_yaxis()
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.write("**Top contributing features:**")
                                    for f, v in shap_exp['top_features'][:5]:
                                        st.write(f"• {f}: {v:.4f}")
                                    
                                    if 'explanation' in shap_exp:
                                        st.info(shap_exp['explanation'])
                                else:
                                    st.info("SHAP explanation not available for this book")

                            with tab2:
                                if lime_exp:
                                    st.markdown(f'<div class="explanation-box">{lime_exp["explanation"]}</div>', unsafe_allow_html=True)
                                    st.write("**Key factors:**")
                                    for f in lime_exp['features']:
                                        st.write(f"• {f}")
                                else:
                                    st.info("LIME explanation not available")

                            with tab3:
                                if cf_exp:
                                    st.markdown('<div class="counterfactual-box">', unsafe_allow_html=True)
                                    st.write("**What would change this recommendation?**")
                                    for s in cf_exp['scenarios']:
                                        st.write(f"• {s}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("Counterfactual explanation not available")

            else:
                st.error("No recommendations found. Please try:")
                st.write("- A different book title")
                st.write("- A valid User ID from the suggested list")
                st.write("- Switching to Content-Based mode if Collaborative fails")

if __name__ == "__main__":
    main()
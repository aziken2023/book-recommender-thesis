# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "Creating app.py..." -ForegroundColor Yellow
    
    $appContent = @'
"""
Explainable Book Recommender System
Streamlit Dashboard for Hybrid Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .book-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateX(5px);
    }
    .explanation-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .counterfactual-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_models():
    """Load all pre-trained models and data"""
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
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please ensure your data and model files are in the correct directories.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

def recommend_content_based(title, data, n=10):
    """Content-based recommendations using TF-IDF"""
    books_df = data['books_df']
    tfidf_matrix = data['tfidf_matrix']
    
    matches = books_df[books_df['title'].str.lower().str.contains(title.lower(), na=False)]
    if len(matches) == 0:
        return None
    
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:n+1]
    
    results = books_df.iloc[similar_indices][
        ['title', 'authors', 'top_genre', 'average_rating', 'ratings_count']
    ].copy()
    results['score'] = sim_scores[similar_indices]
    
    return results

def recommend_collaborative(user_id, data, n=10):
    """Collaborative filtering using SVD"""
    books_df = data['books_df']
    svd_artifacts = data['svd_artifacts']
    interactions_df = data['interactions_df']
    
    try:
        user_id_to_idx = svd_artifacts['user_id_to_idx']
        idx_to_book_id = svd_artifacts['idx_to_book_id']
        user_factors = svd_artifacts['user_factors']
        book_factors = svd_artifacts['book_factors']
        
        if user_id not in user_id_to_idx:
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
        
        return results[['title', 'authors', 'top_genre', 'average_rating', 'ratings_count', 'score']]
    
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return None

def get_hybrid_recommendations(user_id, book_title, data, n=10, content_weight=0.4):
    """Hybrid recommendations combining both approaches"""
    content_recs = recommend_content_based(book_title, data, n=n*2)
    if content_recs is None:
        return None
    
    collab_recs = recommend_collaborative(user_id, data, n=n*2) if user_id else None
    
    if collab_recs is None:
        return content_recs.head(n)
    
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
    
    return results

def get_shap_explanation(book_title, data, n_features=10):
    """Generate SHAP-like feature importance explanation"""
    books_df = data['books_df']
    tfidf_vectorizer = data['tfidf_vectorizer']
    tfidf_matrix = data['tfidf_matrix']
    
    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None
        
        idx = matches.index[0]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        feature_values = tfidf_matrix[idx].toarray()[0]
        
        top_indices = feature_values.argsort()[-n_features:][::-1]
        top_features = [(feature_names[i], feature_values[i]) for i in top_indices if feature_values[i] > 0]
        
        return {
            'title': book_title,
            'top_features': top_features
        }
    except:
        return None

def get_lime_explanation(book_title, data):
    """Generate plain English explanation"""
    books_df = data['books_df']
    
    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None
        
        book = matches.iloc[0]
        explanation_parts = []
        
        if book['top_genre']:
            explanation_parts.append(f"Genre: {book['top_genre']}")
        
        if book['authors']:
            explanation_parts.append(f"Author: {book['authors']}")
        
        if book['average_rating']:
            rating_text = "highly rated" if book['average_rating'] >= 4 else "well-rated" if book['average_rating'] >= 3 else "moderately rated"
            explanation_parts.append(f"{rating_text} ({book['average_rating']:.1f} stars)")
        
        if book['ratings_count']:
            if book['ratings_count'] > 1000:
                explanation_parts.append("popular among readers")
            elif book['ratings_count'] > 100:
                explanation_parts.append("gaining popularity")
            else:
                explanation_parts.append("undiscovered gem")
        
        return {
            'title': book_title,
            'explanation': "This book was recommended because: " + ", ".join(explanation_parts),
            'features': explanation_parts
        }
    except:
        return None

def get_counterfactual_explanation(user_id, book_title, data):
    """Generate counterfactual explanation"""
    books_df = data['books_df']
    interactions_df = data['interactions_df']
    
    try:
        matches = books_df[books_df['title'].str.lower().str.contains(book_title.lower(), na=False)]
        if len(matches) == 0:
            return None
        
        book = matches.iloc[0]
        scenarios = []
        
        if user_id:
            user_books = interactions_df[interactions_df['user_id'] == user_id]
            liked_books = user_books[user_books['rating'] >= 4]['book_id'].tolist()
            
            if liked_books:
                liked_authors = set(books_df[books_df['book_id'].isin(liked_books)]['authors'])
                if book['authors'] in liked_authors:
                    scenarios.append(f"You like other books by {book['authors']}")
                else:
                    scenarios.append(f"If you rated a book by {book['authors']} highly, this would be recommended")
        
        scenarios.append(f"Exploring more {book['top_genre']} books would bring similar recommendations")
        
        if user_id and len(user_books) > 0:
            avg_rating = user_books['rating'].mean()
            if avg_rating >= 4:
                scenarios.append("Your high ratings across books makes this appealing")
            else:
                scenarios.append(f"Your rating style (avg: {avg_rating:.1f}) influences what you see")
        
        return {
            'title': book_title,
            'scenarios': scenarios
        }
    except:
        return None

def main():
    st.markdown('<div class="main-header">📚 Explainable Book Recommender System</div>', unsafe_allow_html=True)
    
    data = load_models()
    if data is None:
        st.stop()
    
    with st.sidebar:
        st.header("🎯 Settings")
        
        model_type = st.selectbox(
            "Recommendation Model",
            ["Hybrid", "Content-Based", "Collaborative"]
        )
        
        if model_type == "Hybrid":
            content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.05)
            st.info(f"Collaborative Weight: {1 - content_weight:.2f}")
        else:
            content_weight = 0.4
        
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)
        
        st.divider()
        st.subheader("👤 User Profile")
        user_id = st.text_input("User ID", value="320562")
        try:
            user_id = int(user_id) if user_id else None
        except:
            user_id = None
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Search")
        books_df = data['books_df']
        
        search_term = st.text_input("Enter book title", placeholder="e.g., Harry Potter")
        
        if search_term:
            filtered = books_df[books_df['title'].str.lower().str.contains(search_term.lower(), na=False)]
            if len(filtered) > 0:
                selected_book = st.selectbox("Select a book", filtered['title'].tolist()[:50])
            else:
                st.warning("No books found")
                selected_book = None
        else:
            selected_book = st.selectbox("Select a book", books_df['title'].tolist()[:100])
        
        if st.button("Get Recommendations", type="primary") and selected_book:
            st.session_state['selected_book'] = selected_book
            st.session_state['get_recs'] = True
    
    with col2:
        st.subheader("📊 Recommendations")
        
        if 'get_recs' in st.session_state and st.session_state['get_recs']:
            book_title = st.session_state['selected_book']
            
            with st.spinner("Generating recommendations..."):
                if model_type == "Content-Based":
                    results = recommend_content_based(book_title, data, n_recs)
                elif model_type == "Collaborative":
                    results = recommend_collaborative(user_id, data, n_recs)
                else:
                    results = get_hybrid_recommendations(user_id, book_title, data, n_recs, content_weight)
                
                if results is not None and len(results) > 0:
                    selected = books_df[books_df['title'] == book_title].iloc[0]
                    st.info(f"📖 **Selected:** {book_title} by {selected['authors']}")
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
                            
                            with st.expander("🔍 Why this recommendation?"):
                                tab1, tab2, tab3 = st.tabs(["📊 Features", "💬 Plain English", "🔄 What If"])
                                
                                with tab1:
                                    shap = get_shap_explanation(row['title'], data)
                                    if shap and shap['top_features']:
                                        features, values = zip(*shap['top_features'][:8])
                                        fig, ax = plt.subplots(figsize=(8, 3))
                                        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
                                        ax.barh(features, values, color=colors)
                                        ax.set_xlabel('Feature Value')
                                        ax.set_title('Top Contributing Features')
                                        ax.invert_yaxis()
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.info("Feature explanation not available")
                                
                                with tab2:
                                    lime = get_lime_explanation(row['title'], data)
                                    if lime:
                                        st.markdown(f'<div class="explanation-box">{lime["explanation"]}</div>', unsafe_allow_html=True)
                                        st.write("**Key factors:**")
                                        for f in lime['features']:
                                            st.write(f"• {f}")
                                    else:
                                        st.info("Explanation not available")
                                
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
                    st.error("No recommendations found")
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit | GoodReads Dataset | Explainable AI")

if __name__ == "__main__":
    main()
'@
    
    $appContent | Out-File -FilePath app.py -Encoding UTF8
    Write-Host "✅ app.py created!" -ForegroundColor Green
}
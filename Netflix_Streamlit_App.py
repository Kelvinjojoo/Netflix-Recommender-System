import streamlit as st
import joblib
import pandas as pd
import gdown
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi download yang lebih reliable dengan verifikasi
def download_file_from_gdrive(file_id, output_path):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Coba download dengan gdown
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            gdown.download(url, output_path, quiet=False)
            
            # Verifikasi file benar-benar ada dan tidak kosong
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)/1024/1024  # Size in MB
                st.success(f"✅ File {output_path} downloaded successfully! Size: {file_size:.2f} MB")
                return True
            else:
                st.warning(f"⚠️ Download attempt {attempt + 1} failed - file empty or not created")
                if os.path.exists(output_path):
                    os.remove(output_path)
                
        except Exception as e:
            st.warning(f"⚠️ Download attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    st.error(f"❌ Failed to download {output_path} after {max_retries} attempts")
    return False

def main():
    st.title('🎬 Netflix Movie Recommender System')
    
    # Download hanya cosine_sim.pkl
    COSINE_SIM_FILE_ID = "1aqt1e2VHpgGOFN57nJWDjaPV1l1RPkZv"
    COSINE_SIM_PATH = "cosine_sim.pkl"
    
    with st.spinner("Downloading cosine similarity file..."):
        if not download_file_from_gdrive(COSINE_SIM_FILE_ID, COSINE_SIM_PATH):
            st.error("Failed to download required cosine similarity file. App cannot continue.")
            st.stop()
    
    # Verifikasi file sebelum memuat
    if not os.path.exists(COSINE_SIM_PATH):
        st.error(f"File {COSINE_SIM_PATH} not found after download attempt.")
        st.stop()
    
    # Muat semua file yang diperlukan
    try:
        with st.spinner("Loading data files..."):
            # Asumsi file-file lain sudah ada di direktori
            tfidf = joblib.load("tfidf.pkl")
            cosine_sim = joblib.load(COSINE_SIM_PATH)
            movies_df = joblib.load("movies_df.pkl")
        st.success("All data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        st.stop()

    # ... (rest of your existing code for recommendations)

    st.markdown("""
    <style>
    .recommendation-box{
        border-left: 5px solid #f63366;
        padding: 15px;
        margin: 10px 0;
        background-color: #1e1e1e;
        color: #f5f5f5;
        border-radius: 8px;
        font-size: 15px;
    }
    .recommendation-box h4{
        color: #f63366;
    }
    .movie-details{
        background-color: #2e2e2e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    movie_title = st.selectbox(
        "Search for a movie:",
        options=movies_df['title'].tolist(),
        format_func=lambda x: x,
        index=None,
        placeholder="Type to search...",
    )
    
    if st.button("Get Recommendations", key="recommend_button"):
        if not movie_title:
            st.warning("Please select a movie first!")
        else:
            st.markdown("---")
            
            st.subheader(f"🔍 You searched for: {movie_title}")
            movie_details = movies_df[movies_df['title'] == movie_title].iloc[0]
            
            with st.expander("See movie details", expanded=True):
                st.markdown(f"""
                <div class="movie-details">
                    <p><b>Genre:</b> {movie_details['listed_in']}</p>
                    <p><b>Director:</b> {movie_details['director']}</p>
                    <p><b>Cast:</b> {movie_details['cast']}</p>
                    <p><b>Country:</b> {movie_details['country']}</p>
                    <p><b>Rating:</b> {movie_details['rating']}</p>
                    <p><b>Description:</b> {movie_details['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            recommendations = get_recommendations(movie_title, 5)
            
            if not recommendations.empty:
                st.subheader("🎬 Top 5 Recommendations")
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4>{row['title']}</h4>
                            <p><b>Genre:</b> {row['listed_in']}</p>
                            <p><b>Director:</b> {row['director']}</p>
                            <p><b>Cast:</b> {row['cast']}</p>
                            <p><b>Country:</b> {row['country']}</p>
                            <p><b>Rating:</b> {row['rating']}</p>
                            <p><b>Description:</b> {row['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("No recommendations found. Try another movie!")

def get_recommendations(title, top_n=5):
    try:
        movies_df = joblib.load("movies_df.pkl")
        cosine_sim = joblib.load("cosine_sim.pkl")
        
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
            
        recommendations = movies_df.iloc[movie_indices][['title', 'listed_in', 'director', 'cast', 'country', 'rating', 'description']]
        recommendations['similarity_score'] = [round(sim_scores[i][1], 3) for i in range(1, top_n+1)]
            
        return recommendations
        
    except IndexError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    main()

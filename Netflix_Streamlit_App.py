import streamlit as st
import joblib
import pandas as pd
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity

def download_file_from_gdrive(file_id, output_path):
  if not os.path.exists(output_path):
    url= f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

cosine_sim_file_id= "1aqt1e2VHpgGOFN57nJWDjaPV1l1RPkZv"
tfidf_file_id = "1RC0hiWRoFPqqxSC2O9bdApMuzexGMWZy"
movies_df_file_id = "1D76gqTn3gRN4qx2C6A9a9pvqIJHq1VLC"

download_file_from_gdrive(cosine_sim_file_id, "cosine_sim.pkl")
download_file_from_gdrive(tfidf_file_id, "tfidf.pkl")
download_file_from_gdrive(movies_df_file_id, "movies_df.pkl")

tfidf= joblib.load("tfidf.pkl")
cosine_sim= joblib.load("cosine_sim.pkl")
movies_df= joblib.load("movies_df.pkl")

def get_recommendations(title, top_n= 5):
  try:
    idx= movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
    sim_scores= list(enumerate(cosine_sim[idx]))
    sim_scores= sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices= [i[0] for i in sim_scores[1:top_n+1]]
        
    recommendations= movies_df.iloc[movie_indices][['title', 'listed_in', 'director', 'cast', 'country', 'rating', 'description']]
    recommendations['similarity_score']= [round(sim_scores[i][1], 3) for i in range(1, top_n+1)]
        
    return recommendations
    
  except IndexError:
    return pd.DataFrame()

def main():
  st.title('🎬 Netflix Movie Recommender System')
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
  """, unsafe_allow_html= True)
    

  movie_title= st.selectbox(
    "Search for a movie:",
    options= movies_df['title'].tolist(),
    format_func= lambda x: x,
    index= None,
    placeholder= "Type to search...",
  )
    
  if st.button("Get Recommendations", key= "recommend_button"):
    if not movie_title:
      st.warning("Please select a movie first!")
    else:
      st.markdown("---")
            
      st.subheader(f"🔍 You searched for: {movie_title}")
      movie_details= movies_df[movies_df['title'] == movie_title].iloc[0]
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
            
      recommendations= get_recommendations(movie_title, 5)
            
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

if __name__ == "__main__":
  main()

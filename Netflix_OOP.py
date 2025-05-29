import numpy as np
import pandas as pd
import pickle

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class NetflixRecommender:
  def __init__(self, filepath):
    self.filepath= filepath
    self.df= None
    self.movies_df= None
    self.tfidf= None
    self.tfidf_matrix= None
    self.cosine_sim= None 

  def read_data(self):
    self.df= pd.read_csv(self.filepath)

  def filter_movies(self):
    self.movies_df= self.df[self.df["type"] == "Movie"].reset_index(drop=True)
  
  def drop_identifier(self):
    self.movies_df= self.movies_df.drop(columns=['show_id'])

  def change_data_type(self):
    self.movies_df['date_added'] = pd.to_datetime(self.movies_df['date_added'])

  def handling_missing_values(self):
    self.movies_df["director"]= self.movies_df["director"].fillna("Unknown")
    self.movies_df["cast"]= self.movies_df["cast"].fillna("Unknown")
    self.movies_df["country"]= self.movies_df["country"].fillna("Unknown")
    self.movies_df["rating"]= self.movies_df["rating"].fillna("Unknown")
    self.movies_df["duration"]= self.movies_df["duration"].fillna("Unknown")

  def handling_anomaly(self):
    def clean_country(value):
      cleaned= value.strip().strip(',')
      countries= [c.strip().title() for c in cleaned.split(',') if c.strip()]
      return ', '.join(countries)
        
    self.movies_df['country']= self.movies_df['country'].apply(clean_country)

    anomaly_values= ['74 min', '84 min', '66 min']
    mask= self.movies_df['rating'].isin(anomaly_values)
    self.movies_df.loc[mask, ['rating', 'duration']]= self.movies_df.loc[mask, ['duration', 'rating']].values

  def clean_text(self, text):
    if pd.isna(text):
      return ""
    text= re.sub(r'[^\w\s]', '', str(text).lower())  
    return text

  def create_soup_feature(self):        
    def create_soup(row):
      title= ' '.join([self.clean_text(row['title'])] * 2)

      genres= ' '.join([self.clean_text(g) for g in str(row['listed_in']).split(', ')]) * 3

      director= self.clean_text(row['director']) * 2
      cast= ' '.join([self.clean_text(a) for a in str(row['cast']).split(', ') if a.strip().lower() != 'unknown'])
      description= self.clean_text(row['description'])
      country= self.clean_text(row['country'])
      rating= self.clean_text(row['rating'])
            
      return f"{title} {genres} {director} {cast} {country} {rating} {description}"
        
    self.movies_df['soup']= self.movies_df.apply(create_soup, axis=1)

  def build_recommendation_model(self):
    self.tfidf= TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.8)
    self.tfidf_matrix= self.tfidf.fit_transform(self.movies_df['soup'])
    self.cosine_sim= linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

  def save_models(self):
    pickle.dump(self.tfidf, open('tfidf.pkl', 'wb'))
    pickle.dump(self.cosine_sim, open('cosine_sim.pkl', 'wb'))
    pickle.dump(self.movies_df, open('movies_df.pkl', 'wb'))

  def get_recommendations(self, title, top_n= 5):
    idx= self.movies_df[self.movies_df['title'].str.lower() == title.lower()].index[0]
        
    sim_scores= list(enumerate(self.cosine_sim[idx]))
    sim_scores= sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
    movie_indices= [i[0] for i in sim_scores[1:top_n+1]]
        
    recommendations= self.movies_df.iloc[movie_indices][['title', 'listed_in', 'director', 'cast', 'country', 'rating', 'description']].copy()
    recommendations['similarity']= [round(sim_scores[i][1], 6) for i in range(1, top_n+1)]
        
    print("="*80)
    print(f"🎬 DETAIL FILM: {self.movies_df.iloc[idx]['title'].upper()}")
    print("="*80)
    print(f"Genre: {self.movies_df.iloc[idx]['listed_in']}")
    print(f"Director: {self.movies_df.iloc[idx]['director']}")
    print(f"Cast: {self.movies_df.iloc[idx]['cast']}")
    print(f"Country: {self.movies_df.iloc[idx]['country']}")
    print(f"Rating: {self.movies_df.iloc[idx]['rating']}")
    print(f"Description: {self.movies_df.iloc[idx]['description']}\n\n")
        
    print(f"🍿 REKOMENDASI TOP {top_n}")
    print("="*80)
    for i, row in recommendations.iterrows():
      print(f"\n> {row['title']} (Similarity: {row['similarity']})")
      print("-"*60)
      print(f"Genre: {row['listed_in']}")
      print(f"Director: {row['director']}")
      print(f"Cast: {row['cast']}")
      print(f"Country: {row['country']}")
      print(f"Rating: {row['rating']}")
      print(f"Description: {row['description']}")
        
  
recommender= NetflixRecommender('netflix_titles.csv')
recommender.read_data()
recommender.filter_movies()
recommender.drop_identifier()
recommender.change_data_type()
recommender.handling_missing_values()
recommender.handling_anomaly()
recommender.create_soup_feature()
recommender.build_recommendation_model()
recommender.save_models()

recommender.get_recommendations("The Conjuring")
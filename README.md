# 🎬 Netflix Recommender System

A content based movie recommendation system using the Netflix dataset (up to 2021).  
It uses **Soup (Bag of Words)**, **TF-IDF Vectorization**, and **Cosine Similarity** to suggest the top 5 most similar movies based on a selected title.

Built with **Python** and deployed using **Streamlit**.

> 📝 This project was created as the **final project** for the **Model Deployment** course.

---

## 🌐 Live App

Try it out now on Streamlit:  
🔗 [https://netflix-recommenender-system.streamlit.app/](https://netflix-recommenender-system.streamlit.app/)

---

## 🧠 How It Works

1. Combines movie metadata (title, cast, director, genres, description, etc) into a "soup".
2. Transforms text data into vectors using **TF-IDF**.
3. Calculates similarity between titles using **cosine similarity**.
4. Recommends 5 most similar titles based on selected movie.

---

## 📁 Dataset

The dataset includes Netflix content metadata available **until 2021**.  

---

## 🚀 Deployment Notes (Streamlit)

Planning to deploy your own version on Streamlit with my `.pkl` files? Here’s what you need:

- ✅ Use **Python 3.10** in your Streamlit deployment settings. this ensures your `.pkl` files (like the similarity matrix) load without issues.
- ✅ Make sure all dependencies are included in your `requirements.txt` and properly installed.

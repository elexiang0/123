import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from surprise import Reader, Dataset

# 加载模型
with open('svd_model.pkl', 'rb') as f:
    algo = pickle.load(f)

# 加载笑话数据
jokes_df = pd.read_excel('/home/featurize/Dataset4JokeSet.xlsx')
jokes_df.columns = ['joke']
jokes_df = jokes_df.rename_axis('joke_id').reset_index()

def get_top_n_recommendations(user_id, n=10):
    user_rated_jokes = ratings[ratings['user_id'] == user_id]['joke_id']
    all_jokes = set(ratings['joke_id'].unique())
    unrated_jokes = all_jokes - set(user_rated_jokes)
    predictions = [algo.predict(user_id, joke_id) for joke_id in unrated_jokes]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    top_n_joke_ids = [pred.iid for pred in top_n]
    top_n_estimates = [pred.est for pred in top_n]
    return jokes_df[jokes_df['joke_id'].isin(top_n_joke_ids)][['joke_id', 'joke']], top_n_estimates

def main():
    st.title("Joke Recommendation System")

    # 初始界面随机出现3个笑话
    if 'new_user_ratings' not in st.session_state:
        st.session_state.new_user_ratings = []

    if len(st.session_state.new_user_ratings) < 3:
        st.subheader("Rate the following jokes:")
        random_jokes = jokes_df.sample(3).reset_index(drop=True)
        for i, row in random_jokes.iterrows():
            rating = st.slider(f"Joke {i+1}: {row['joke']}", 0, 5, 0)
            if rating > 0:
                st.session_state.new_user_ratings.append((row['joke_id'], rating))

    if len(st.session_state.new_user_ratings) >= 3:
        st.subheader("You've rated 3 jokes. Here are some recommendations based on your ratings:")

        new_user_id = ratings['user_id'].max() + 1
        new_ratings = pd.DataFrame({
            'user_id': [new_user_id] * 3,
            'joke_id': [rating[0] for rating in st.session_state.new_user_ratings],
            'rating': [rating[1] for rating in st.session_state.new_user_ratings],
        })

        combined_ratings = pd.concat([ratings, new_ratings], ignore_index=True)
        reader = Reader(rating_scale=(1, 5))
        new_train_data = Dataset.load_from_df(combined_ratings[['user_id', 'joke_id', 'rating']], reader)
        new_trainset = new_train_data.build_full_trainset()
        algo.fit(new_trainset)

        recommended_jokes, top_n_estimates = get_top_n_recommendations(new_user_id, n=5)
        recommended_jokes['predicted_rating'] = top_n_estimates

        for idx, row in recommended_jokes.iterrows():
            st.write(f"Joke: {row['joke']}")
            st.write(f"Predicted Rating: {row['predicted_rating']}")

        st.subheader("Please rate the recommended jokes:")

        user_satisfaction_ratings = []
        for i, row in recommended_jokes.iterrows():
            rating = st.slider(f"Recommended Joke {i+1}: {row['joke']}", 0, 5, 0)
            user_satisfaction_ratings.append(rating)

        satisfaction_score = np.mean(user_satisfaction_ratings)
        st.write(f"Your satisfaction score for the recommendations is: {satisfaction_score}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# In[2]:


os.getcwd()

# ## Importing Data

# In[3]:


import pandas as pd
import numpy as np

# In[4]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# In[5]:


# movies.head(1)

# In[6]:


# credits.head(1)

# #### First step will be to merge both the dataframes `movies` and `credits` as a single dataframe

# In[7]:


df = movies.join(credits.set_index("title"), on = "title")

# In[8]:


# df.head(1)

# In[9]:


# movies.shape

# In[10]:


# credits.shape

# In[11]:


# df.shape

# #### Now, I'll just maintain the columns that are relevant to our recommendation system, and the rest will be avoided.

# In[12]:


# df.info()

# In[13]:


df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# In[14]:


# df.head(1)

# So, I retrieved the necessary columns from the merged dataframe for further processing and creating a __content-based recommendation system__.

# ## Handling Missing Values

# In[15]:


# df.isna().sum()

# As we can see, the `overview` feature contains 3 rows of nan values. I am eliminating these because there are just three rows. It will not have a significant impact on the statistics.

# In[16]:


df.dropna(inplace = True)

# ## Data Preprocessing

# In[17]:


# df.genres[0]

# In[18]:


import ast

def convert(obj):
    objects = []
    for i in ast.literal_eval(obj):
        objects.append(i["name"])
    return objects

# In[19]:


df["genres"] = df.genres.apply(convert)

# In[20]:


# df.keywords[0]

# In[21]:


# convert(df.keywords[0])

# In[22]:


df["keywords"] = df.keywords.apply(convert)

# In[23]:


# df["keywords"]

# In[24]:


# df.keywords[0]

# In[25]:


# df.cast[0]

# In[26]:


def fetch_cast(obj):
    names = []

    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            names.append(i["name"])
            count += 1
        else:
            break
            
    return names

# In[27]:


df["cast"] = df.cast.apply(fetch_cast)

# In[28]:


# df["cast"]

# In[29]:


# df["crew"][0]

# In[30]:


jobs = []
for i in ast.literal_eval(df["crew"][0]):
    jobs.append(i["job"])

set(jobs)

# In[31]:


def fetch_director(obj):
    director = []

    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            director.append(i["name"])

    return director

# In[32]:


df["crew"] = df.crew.apply(fetch_director)

# In[33]:


# df["crew"]

# In[34]:


# df.head(1)

# In[35]:


df["overview"] = df["overview"].apply(lambda x: x.split())

# In[36]:


# df.head()

# In[37]:


for feature in df.columns[-4:]:
    df[feature] = df[feature].apply([lambda x: [i.replace(" ", "") for i in x]])

# In[38]:


# df.head(1)

# In[39]:


df["overview"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]

# In[40]:


# df.head()

# In[41]:


# df.overview[0]

# In[42]:


df["overview"] = df["overview"].apply(lambda x: " ".join(x).lower())

# In[43]:


# df.overview

# In[44]:


new_df = df[["movie_id", "title", "overview"]]

# In[45]:


# new_df.head()

# In[46]:


from nltk.stem import SnowballStemmer

def word_stemming(obj):
    stemmer = SnowballStemmer("english")
    clean_words = [stemmer.stem(token) for token in obj.split()]
    return " ".join(clean_words)

# In[47]:


new_df.loc[:, "overview"] = new_df["overview"].apply(word_stemming)

# ### Feature Engineering - Converting Text into Vectors

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 8000, stop_words = "english")

# In[49]:


vectors = cv.fit_transform(new_df["overview"]).toarray()

# In[50]:


# vectors

# In[51]:


# cv.get_feature_names_out()[100:150]

# In[52]:


# len(vectors)

# In[53]:


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# In[54]:


# similarity

# In[56]:


# similarity.shape

# In[57]:


# similarity[0] # first movie

# In[129]:


# new_df.title.sample(10)

# ## Movie Recommendation Function
# Creating a recommendation function that allows us to recommend five films to the user based on their cosine similarity.
# 
# __The final step will be to deploy this project to the `Streamlit Cloud`.__

# In[138]:


def recommender(movie):
    try:
        movie_index = new_df[new_df.title == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x: x[1])[1:6]
    
        for i in movie_list:
            print(new_df.iloc[i[0]].title)
            
    except IndexError:
        print("Error: Movie not found or index out of range.")

# In[139]:


recommender("Spider-Man")

# Converting the `dataframe` and `cosine similarity matrix` to pickle file which will be utilised for developing a streamlit app.

# In[137]:


# pd.to_pickle(new_df, "movies.pkl")

# In[140]:


# pd.to_pickle(similarity, "similarities.pkl")

#!/usr/bin/env python
# coding: utf-8

# # Recommendation System

# There are basically three types of recommender systems:-
# 
# > *  **Demographic Filtering**- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
# 
# ![](https://i.imgur.com/rV4hfnH.jpeg)
# 
# 
# 
# 

# > *  **Content Based Filtering**- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
# 
# ### Amazon
# ![](https://i.imgur.com/Qg5qBgl.png)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ### Netflix
# 
# ![](https://i.imgur.com/Ugkbtfi.png)

# > *  **Collaborative Filtering**- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.
# 
# ### Amazon
# ![](https://i.imgur.com/N3hoabm.png)

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')
df2


# In[4]:


df2.head(5)


# # Demographic Filtering 

# In[5]:


C= df2['vote_average'].mean()
C


# In[6]:


m= df2['vote_count'].quantile(0.9)
m


# In[7]:


q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape


# In[8]:


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[9]:


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[10]:


#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# In[11]:


pop= df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='red')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# # Plot description based Recommender¶
# We will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the overview feature of our dataset.

# In[12]:


df2['overview'].head(5)


# In[13]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[14]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[15]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


# In[16]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# In[17]:


get_recommendations('The Dark Knight Rises')


# In[18]:


get_recommendations('The Avengers')


# # Credits, Genres and Keywords Based Recommender¶
# It goes without saying that the quality of our recommender would be increased with the usage of better metadata. That is exactly what we are going to do in this section. We are going to build a recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.

# In[29]:


from ast import literal_eval
import pandas as pd

def safe_literal_eval(val):
    # Check if the value is a string and try to evaluate it
    if isinstance(val, str):
        try:
            return literal_eval(val)
        except (ValueError, SyntaxError):
            return val  # Return the original value if it can't be evaluated
    return val  # Return the original value if it's not a string

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(safe_literal_eval)
    


# In[30]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[32]:


import pandas as pd

# Function to extract the director's name from the crew list
def get_director(crew):
    for person in crew:
        if person.get('job') == 'Director':
            return person.get('name')
    return None  # Return None if no director is found

# Function to extract a list of names or values from a list of dictionaries
def get_list(feature_list):
    if isinstance(feature_list, list):
        return [item.get('name') for item in feature_list if 'name' in item]
    return []  # Return an empty list if the input is not a list

# Ensure functions are defined before applying them to the DataFrame
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

    


# In[33]:


# Print the new features of the first 3 films
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[34]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[35]:


#Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
   df2[feature] = df2[feature].apply(clean_data)


# In[36]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)


# In[37]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[38]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[39]:


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


# In[40]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[41]:


get_recommendations('The Godfather', cosine_sim2)


# # Collaborative Filtering

# In[52]:


from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()


# In[55]:


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load the dataset
ratings = pd.read_csv('ratings_small.csv')

# Display the first few rows
print(ratings.head())

# Define a Reader object with the appropriate rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load the dataset into Surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Build the SVD algorithm
algo = SVD()

# Perform cross-validation to evaluate the algorithm
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[58]:


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
from surprise import accuracy  # Import the accuracy module

# Load the dataset
ratings = pd.read_csv('ratings_small.csv')
print(ratings.head())

# Define a Reader object with the appropriate rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load the dataset into Surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Build the SVD algorithm
algo = SVD()

# Define a cross-validation iterator
kf = KFold(n_splits=5)

# Perform cross-validation
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Calculate and print RMSE and MAE for the current fold
    print('RMSE:', accuracy.rmse(predictions, verbose=True))
    print('MAE:', accuracy.mae(predictions, verbose=True))


# In[60]:


ratings[ratings['userId'] == 1]







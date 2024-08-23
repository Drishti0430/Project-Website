#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from collections import Counter
import warnings
from wordcloud import WordCloud
warnings.filterwarnings("ignore", category=FutureWarning)

import plotly.express as px

import networkx as nx

import joypy

import plotly.graph_objs as go

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("IMdB_India_Top250.csv")

df.head(), df.info()


# In[3]:


cleaned_df=df

cleaned_df.head()


# # Visualising Dataset

# In[4]:


plt.figure(figsize=(12, 6))
df['Year of release'].value_counts().sort_index().plot(kind='bar', color='yellow')
plt.title('Distribution of Movies by Release Year')
plt.xlabel('Year of Release')
plt.ylabel('Number of Movies')
plt.show()


# In[5]:


genres = df['Genre'].str.split(',', expand=True).stack().str.strip().value_counts()

plt.figure(figsize=(12, 8))
genres.plot(kind='barh', color='purple')
plt.title('Distribution of Movies by Genre')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()


# In[6]:


top_directors = df['Director'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_directors.plot(kind='bar', color='blue')
plt.title('Top 10 Directors by Number of Movies')
plt.xlabel('Director')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()


# In[7]:


plt.figure(figsize=(10, 6))
df['Rating'].hist(bins=20, color='lightblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
plt.show()


# In[8]:


df['Year of release'] = df['Year of release'].astype(int)
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])

pivot_data = df.pivot_table(index='Genre', columns='Year of release', values='Rating', aggfunc='mean')

plt.figure(figsize=(16, 10))
sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap of Average Movie Ratings by Year and Genre', fontsize=16)
plt.xlabel('Year of Release')
plt.ylabel('Genre')
plt.show()


# In[9]:


df['Year of release'] = df['Year of release'].astype(str)

fig = px.sunburst(df, path=['Genre', 'Director', 'Year of release'], values='Rating',
                  color='Rating', hover_data=['Rating'],
                  color_continuous_scale='RdBu',
                  title='Sunburst of Movie Counts by Genre, Director, and Year')
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()


# In[10]:


plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Film Industry', y='Rating', palette='Set3')
plt.title('Box Plot of Movie Ratings by Film Industry', fontsize=16)
plt.xlabel('Film Industry')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[19]:


df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])

plt.figure(figsize=(14, 10))
joypy.joyplot(df, by='Genre', column='Rating', colormap=plt.cm.Spectral, figsize=(14,10), 
              title='Ridgeline Plot of Ratings Distribution by Genre')
plt.xlabel('Rating')
plt.show()


# In[18]:


correlation_data = df[['Rating', 'Year of release', 'User reviews']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title('Correlation Matrix of Key Attributes', fontsize=16)
plt.show()


# In[17]:


fig = go.Figure(data=[go.Scatter3d(
    x=df['Year of release'],
    y=df['Rating'],
    z=df['User reviews'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['Rating'],
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(scene=dict(
    xaxis_title='Year of Release',
    yaxis_title='Rating',
    zaxis_title='User Reviews'),
    title='3D Scatter Plot of Ratings, Year of Release, and User Reviews'
)
fig.show()


# In[16]:


text = ' '.join(df['Description'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Descriptions', fontsize=16)
plt.show()


# In[15]:


fig = px.treemap(df, path=['Genre', 'Movie name'], values='Rating',
                 color='Rating', hover_data=['Director'],
                 color_continuous_scale='RdBu',
                 title='Treemap of Movies by Genre and Rating')
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()


# In[14]:


top_20_movies = df.sort_values('Box office collection', ascending=False).head(20)
top_20_movies = top_20_movies[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.hlines(y=top_20_movies['Movie name'], xmin=0, xmax=top_20_movies['Box office collection'], color='skyblue')
ax.plot(top_20_movies['Box office collection'], top_20_movies['Movie name'], 'o', markersize=8, color='skyblue')
ax.set_title('Top 20 Movies by Box Office Collection', fontsize=16)
ax.set_xlabel('Box Office Collection ($)')
ax.set_ylabel('Movie Name')
plt.show()


# In[13]:


average_ratings = df.groupby('Genre')['Rating'].mean().sort_values()
categories = average_ratings.index.tolist()
ratings = average_ratings.values

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
ratings = np.concatenate((ratings,[ratings[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, ratings, color='green', alpha=0.25)
ax.plot(angles, ratings, color='green', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title('Polar Area Chart of Average Ratings by Genre', fontsize=16)
plt.show()


# In[ ]:


df['User reviews (binned)'] = pd.cut(df['User reviews'], bins=[0, 50, 100, 500, 1000, 2000], 
                                     labels=['0-50', '51-100', '101-500', '501-1000', '1001-2000'])

plt.figure(figsize=(14, 8))
sns.barplot(data=df, x='Film Industry', y='Rating', hue='User reviews (binned)', palette='Set2')
plt.title('Grouped Bar Chart of Ratings vs. Binned User Reviews by Film Industry', fontsize=16)
plt.xlabel('Film Industry')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.legend(title='User reviews (binned)')
plt.show()


# # Recommendation

# In[ ]:





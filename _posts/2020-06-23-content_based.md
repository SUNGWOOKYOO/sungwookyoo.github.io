---
title: "Content-Based(CB) Filtering"
excerpt: "Let's study about CB and implement it"
categories:
 - study
tags:
 - recommender system
use_math: true
last_modified_at: "2020-06-23"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys, os
import pandas as pd
from math import log
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
DFILE = "ml-latest-small"
CHARSET = 'utf8'
ratings = pd.read_csv(os.path.join(DFILE, 'ratings.csv'), encoding=CHARSET)
tags = pd.read_csv(os.path.join(DFILE, 'tags.csv'), encoding=CHARSET)
movies = pd.read_csv(os.path.join(DFILE, 'movies.csv'), encoding=CHARSET)
```

</div>

# Content-Based(CB) Approach

The Content-Based Recommender relies on the similarity of the items being recommended. The basic idea is that if you like an item, then you will also like a “similar” item. It generally works well when it's easy to determine the context/properties of each item.

## Step 1. Movies to Vectors
One way to get a user profile: **TF-IDF**

<img src="https://miro.medium.com/max/728/0*24vERYUr-6Nms-5f.png" width="300">

A movie can be represented as a vector by utilizing **TF-IDF**

[korean description of How to calculate TF-IDF with examples - wikidocs](https://wikidocs.net/31698)

Let's look at a toy example as follows.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
docs = [
    'you know I want your love',
    'I like you',
    'what should I do '] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
vocab
```

</div>




{:.output_data_text}

```
['I', 'do', 'know', 'like', 'love', 'should', 'want', 'what', 'you', 'your']
```



<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
N = len(docs) 

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append('{:.3f}'.format(tfidf(t,d)))

# print(result)
pd.DataFrame(result, columns = vocab)
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>I</th>
      <th>do</th>
      <th>know</th>
      <th>like</th>
      <th>love</th>
      <th>should</th>
      <th>want</th>
      <th>what</th>
      <th>you</th>
      <th>your</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.288</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.288</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.288</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.405</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Use [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) in sklearn library to calculate TF-IDF with [1, 2]-gram easily. 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0).fit(docs)
swords = ['I', 'you', 'what', 'do', 'should', 'your']
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0, stop_words=swords)
# tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english').fit(docs)
tfidf_matrix = tf.fit_transform(docs)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
tfidf_df
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>know</th>
      <th>know want</th>
      <th>like</th>
      <th>love</th>
      <th>want</th>
      <th>want love</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.447214</td>
      <td>0.447214</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.447214</td>
      <td>0.447214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Let's process [movielen dataset](https://grouplens.org/datasets/movielens/). <br>
If you want to know about more details please see this document.

The description of movielen dataset as follows. <br>
**M=610 users**, **N=9742 movies**. <br>
**100836 ratings**, each movie has genres information.

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
tf = TfidfVectorizer(ngram_range=(1, 1), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape
```

</div>




{:.output_data_text}

```
(9742, 23)
```



<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
tfidf_df
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>action</th>
      <th>adventure</th>
      <th>animation</th>
      <th>children</th>
      <th>comedy</th>
      <th>crime</th>
      <th>documentary</th>
      <th>drama</th>
      <th>fantasy</th>
      <th>fi</th>
      <th>...</th>
      <th>imax</th>
      <th>listed</th>
      <th>musical</th>
      <th>mystery</th>
      <th>noir</th>
      <th>romance</th>
      <th>sci</th>
      <th>thriller</th>
      <th>war</th>
      <th>western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.416846</td>
      <td>0.516225</td>
      <td>0.504845</td>
      <td>0.267586</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.482990</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.512361</td>
      <td>0.000000</td>
      <td>0.620525</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.593662</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.570915</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.821009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.505015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.466405</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.726241</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9737</th>
      <td>0.436010</td>
      <td>0.000000</td>
      <td>0.614603</td>
      <td>0.000000</td>
      <td>0.318581</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.575034</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9738</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.682937</td>
      <td>0.000000</td>
      <td>0.354002</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.638968</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9739</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9740</th>
      <td>0.578606</td>
      <td>0.000000</td>
      <td>0.815607</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9741</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>9742 rows × 23 columns</p>
</div>
</div>



## Step 2. Get Similarity Matrix

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix) # similarity matrix
pd.DataFrame(similarity, columns=movies.movieId)
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.813578</td>
      <td>0.152769</td>
      <td>0.135135</td>
      <td>0.267586</td>
      <td>0.000000</td>
      <td>0.152769</td>
      <td>0.654698</td>
      <td>0.000000</td>
      <td>0.262413</td>
      <td>...</td>
      <td>0.360397</td>
      <td>0.465621</td>
      <td>0.196578</td>
      <td>0.516225</td>
      <td>0.0</td>
      <td>0.680258</td>
      <td>0.755891</td>
      <td>0.000000</td>
      <td>0.421037</td>
      <td>0.267586</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.813578</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.804715</td>
      <td>0.000000</td>
      <td>0.322542</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.341376</td>
      <td>0.379331</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.152769</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.884571</td>
      <td>0.570915</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.162848</td>
      <td>0.000000</td>
      <td>0.419413</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.181883</td>
      <td>0.202105</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.570915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.135135</td>
      <td>0.000000</td>
      <td>0.884571</td>
      <td>1.000000</td>
      <td>0.505015</td>
      <td>0.000000</td>
      <td>0.884571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.144051</td>
      <td>0.201391</td>
      <td>0.687440</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.160888</td>
      <td>0.178776</td>
      <td>0.466405</td>
      <td>0.000000</td>
      <td>0.505015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.267586</td>
      <td>0.000000</td>
      <td>0.570915</td>
      <td>0.505015</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.570915</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.285240</td>
      <td>0.000000</td>
      <td>0.734632</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.318581</td>
      <td>0.354002</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9737</th>
      <td>0.680258</td>
      <td>0.341376</td>
      <td>0.181883</td>
      <td>0.160888</td>
      <td>0.318581</td>
      <td>0.239513</td>
      <td>0.181883</td>
      <td>0.000000</td>
      <td>0.436010</td>
      <td>0.241142</td>
      <td>...</td>
      <td>0.599288</td>
      <td>0.554355</td>
      <td>0.234040</td>
      <td>0.614603</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.899942</td>
      <td>0.000000</td>
      <td>0.753553</td>
      <td>0.318581</td>
    </tr>
    <tr>
      <th>9738</th>
      <td>0.755891</td>
      <td>0.379331</td>
      <td>0.202105</td>
      <td>0.178776</td>
      <td>0.354002</td>
      <td>0.000000</td>
      <td>0.202105</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.476784</td>
      <td>0.615990</td>
      <td>0.260061</td>
      <td>0.682937</td>
      <td>0.0</td>
      <td>0.899942</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.557008</td>
      <td>0.354002</td>
    </tr>
    <tr>
      <th>9739</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.466405</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.431794</td>
      <td>0.678466</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9740</th>
      <td>0.421037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.317844</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.578606</td>
      <td>0.320007</td>
      <td>...</td>
      <td>0.674692</td>
      <td>0.735655</td>
      <td>0.000000</td>
      <td>0.815607</td>
      <td>0.0</td>
      <td>0.753553</td>
      <td>0.557008</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9741</th>
      <td>0.267586</td>
      <td>0.000000</td>
      <td>0.570915</td>
      <td>0.505015</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.570915</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.285240</td>
      <td>0.000000</td>
      <td>0.734632</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.318581</td>
      <td>0.354002</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>9742 rows × 9742 columns</p>
</div>
</div>



## Step 3. Recommend: Find Top K Similar Movies

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
idx2t = movies.title.to_dict()
t2idx = {v: k for k, v in idx2t.items()}
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
def content(title, K=10):
    """ Recommend Top K similar movies. """
    if title not in t2idx:
        result = movies[movies['title'].str.contains(title)] 
        if len(result) != 1:
            print("Given title are patially matched! re-find correct title among results!")
            return result
        else:
            print("From substring '{}', recommend movies as follows!".format(title))
            title = result.iloc[0].title
    
    scores = similarity[t2idx[title]]
    tuples = sorted(list(enumerate(scores)), key=lambda e: e[1], reverse=True)[:K]
    indices = [i for i, _ in tuples]
    return movies.iloc[indices][['title', 'genres']]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
content('Toy Story', K=5)
```

</div>

{:.output_stream}

```
Given title are patially matched! re-find correct title among results!

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>3114</td>
      <td>Toy Story 2 (1999)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>7355</th>
      <td>78499</td>
      <td>Toy Story 3 (2010)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy|IMAX</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
content('Toy Story 2', K=3)
```

</div>

{:.output_stream}

```
From substring 'Toy Story 2', recommend movies as follows!

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1706</th>
      <td>Antz (1998)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>Toy Story 2 (1999)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
content('Matrix, The', K=3)
```

</div>

{:.output_stream}

```
From substring 'Matrix, The', recommend movies as follows!

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>Lawnmower Man 2: Beyond Cyberspace (1996)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Screamers (1995)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Johnny Mnemonic (1995)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
content('Saving Private Ryan', K=3)
```

</div>

{:.output_stream}

```
From substring 'Saving Private Ryan', recommend movies as follows!

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>Braveheart (1995)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>909</th>
      <td>Apocalypse Now (1979)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>933</th>
      <td>Boot, Das (Boat, The) (1981)</td>
      <td>Action|Drama|War</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
content('Inception')
```

</div>

{:.output_stream}

```
From substring 'Inception', recommend movies as follows!

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7372</th>
      <td>Inception (2010)</td>
      <td>Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>6797</th>
      <td>Watchmen (2009)</td>
      <td>Action|Drama|Mystery|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>7625</th>
      <td>Super 8 (2011)</td>
      <td>Mystery|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>8358</th>
      <td>RoboCop (2014)</td>
      <td>Action|Crime|Sci-Fi|IMAX</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Strange Days (1995)</td>
      <td>Action|Crime|Drama|Mystery|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>6151</th>
      <td>V for Vendetta (2006)</td>
      <td>Action|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>6521</th>
      <td>Transformers (2007)</td>
      <td>Action|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>7545</th>
      <td>I Am Number Four (2011)</td>
      <td>Action|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>7866</th>
      <td>Battleship (2012)</td>
      <td>Action|Sci-Fi|Thriller|IMAX</td>
    </tr>
    <tr>
      <th>8151</th>
      <td>Iron Man 3 (2013)</td>
      <td>Action|Sci-Fi|Thriller|IMAX</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Step 4. Furthermore; Utilize Tags

We have to extract and processing tag information to implement tag-based recommender engine.

Therefore, We will use MySQL to processing records eaily.
If you want to know about how to deal with MySQL and pymysql follow [this guide](https://sungwookyoo.github.io/tips/PymySql/) in my blog.

### Step 4 - 1. Generate View in MySQL

<span style="color:red">**PreRequsite:**</span> In mysql, please execute this commands to processing tag information as follows.
```mysql
create view tmp as 
    select movieId, group_concat(tag separator '|') as tag 
    from tags 
    group by movieId;

create view tmp2 as 
    select m.movieId, m.title, m.genres, t.tag 
    from tmp t, movies m 
    where t.movieId = m.movieId;
```

### Step 4 - 2. Extract the View using Pymysql

In the previous step, we generated view table `tmp2`. <br>
Therefore, we will convert the view table into `pandas.Dataframe` as `df`.

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
import argparse, pymysql, sys, os
import pandas as pd
from sqlalchemy import create_engine

parser = argparse.ArgumentParser()
parser.add_argument('-user', help="mysql database user", type=str, required=False, default='swyoo')
parser.add_argument('-pw', help="password", type=str, required=False, default='****')
parser.add_argument('-host', help="ip address", type=str, required=False, default='***.***.***.***')
parser.add_argument('-db', help="database name", type=str, required=False, default='movielen')
parser.add_argument('-charset', help="character set to use", type=str, required=False, default='utf8mb4')
parser.add_argument('-verbose', help="table name", type=bool, required=False, default=False)
sys.argv = ['-f']
args = parser.parse_args()

con = pymysql.connect(host=args.host, user=args.user, password=args.pw, use_unicode=True, charset=args.charset)
cursor = con.cursor()

## helper function
sql = lambda command: pd.read_sql(command, con)
def fetch(command):
    cursor.execute(command)
    return cursor.fetchall()

fetch("use {}".format(args.db))
if args.verbose: print(sql("show tables"))
```

</div>

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
df = sql("select * from tmp2")
df[:3]
```

</div>

{:.output_stream}

```
/home/swyoo/anaconda3/envs/torch/lib/python3.7/site-packages/pymysql/cursors.py:170: Warning: (1260, 'Row 309 was cut by GROUP_CONCAT()')
  result = self._query(query)

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>pixar|pixar|fun</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>fantasy|magic board game|Robin Williams|game</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>moldy|old</td>
    </tr>
  </tbody>
</table>
</div>
</div>



### Step 4 - 3. Calculate TF-IDF Based on Tags

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
idx2midTag = df.movieId.to_dict()
mid2idxTag = {v: k for k, v in idx2midTag.items()}
idx2mid = movies.movieId.to_dict()
mid2idx = {v: k for k, v in idx2mid.items()}
```

</div>

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
tf = TfidfVectorizer(ngram_range=(1, 4), min_df=0, stop_words='english')
tag_matrix = tf.fit_transform(df['tag'])
tag_df = pd.DataFrame(tag_matrix.toarray(), columns=tf.get_feature_names())
tag_df
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>06</th>
      <th>06 oscar</th>
      <th>06 oscar nominated</th>
      <th>06 oscar nominated best</th>
      <th>1900s</th>
      <th>1920s</th>
      <th>1920s gangsters</th>
      <th>1950s</th>
      <th>1950s adolescence</th>
      <th>1960s</th>
      <th>...</th>
      <th>zellweger</th>
      <th>zellweger retro</th>
      <th>zither</th>
      <th>zoe</th>
      <th>zoe kazan</th>
      <th>zombie</th>
      <th>zombies</th>
      <th>zombies zombies</th>
      <th>zooey</th>
      <th>zooey deschanel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1572 rows × 8791 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
similarity_tag = cosine_similarity(tag_matrix, tag_matrix) # similarity matrix
pd.DataFrame(similarity_tag, columns=df.movieId)
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>5</th>
      <th>7</th>
      <th>11</th>
      <th>14</th>
      <th>16</th>
      <th>17</th>
      <th>21</th>
      <th>...</th>
      <th>176371</th>
      <th>176419</th>
      <th>179401</th>
      <th>180031</th>
      <th>180985</th>
      <th>183611</th>
      <th>184471</th>
      <th>187593</th>
      <th>187595</th>
      <th>193565</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.091603</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.506712</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.506712</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.077864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.039869</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>0.0</td>
      <td>0.091603</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032774</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.039869</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>1572 rows × 1572 columns</p>
</div>
</div>



### Step 4 - 4. Recommend Top K movies based on Tags.

<div class="prompt input_prompt">
In&nbsp;[21]:
</div>

<div class="input_area" markdown="1">

```python
def content(title, K=10, tag=False):
    """ Recommend Top K similar movies. """
    if title not in t2idx:
        result = movies[movies['title'].str.contains(title)] 
        if len(result) != 1:
            print("Given title are patially matched! re-find correct title among results!")
            return result
        else:
            print("From substring '{}', recommend movies as follows!".format(title))
            title = result.iloc[0].title
    idx_tfidf = t2idx[title]
    scores_tfidf = similarity[idx_tfidf]
    tuples_tfidf = sorted(list(enumerate(scores_tfidf)), key=lambda e: e[1], reverse=True)[:K]
    indices_tfidf = [i for i, _ in tuples_tfidf]
    if tag and idx2mid[idx_tfidf] in mid2idxTag:
        print("utilize tags ... ")
        idx_tag = mid2idxTag[idx2mid[idx_tfidf]]
        scores_tag = similarity_tag[idx_tag]
        tuples_tag = sorted(list(enumerate(scores_tag)), key=lambda e: e[1], reverse=True)[:K]
        indices_tag = [i for i, _ in tuples_tag]
        return df.iloc[indices_tag][['title', 'tag']]
    
    return movies.iloc[indices_tfidf][['title', 'genres']]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
content("Toy Story 2", tag=True)
```

</div>

{:.output_stream}

```
From substring 'Toy Story 2', recommend movies as follows!
utilize tags ... 

```




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>666</th>
      <td>Toy Story 2 (1999)</td>
      <td>animation|Disney|funny|original|Pixar|sequel|T...</td>
    </tr>
    <tr>
      <th>544</th>
      <td>Bug's Life, A (1998)</td>
      <td>Pixar</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Toy Story (1995)</td>
      <td>pixar|pixar|fun</td>
    </tr>
    <tr>
      <th>909</th>
      <td>Road to Perdition (2002)</td>
      <td>cinematography|Tom Hanks</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>Invincible Iron Man, The (2007)</td>
      <td>animation</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Aladdin (1992)</td>
      <td>Disney</td>
    </tr>
    <tr>
      <th>147</th>
      <td>Snow White and the Seven Dwarfs (1937)</td>
      <td>Disney</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Beauty and the Beast (1991)</td>
      <td>Disney</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Pinocchio (1940)</td>
      <td>Disney</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Aristocats, The (1970)</td>
      <td>Disney</td>
    </tr>
  </tbody>
</table>
</div>
</div>



We can also use genres and tags at the same time <br>
However, I only use genres or tag for simplify this problem. 

## Summary

### Pros
* No need for data on other users, thus **no cold-start or sparsity problems**.
* Can recommend to users with **unique tastes**.
* Can recommend **new & unpopular items**.
* Can **provide explanations** for recommended items by listing content-features that caused an item to be recommended (in this case, movie genres)

### Cons
* Finding the appropriate features is hard.
* Does not recommend items outside a user's content profile.
* Unable to exploit quality judgments of other users.

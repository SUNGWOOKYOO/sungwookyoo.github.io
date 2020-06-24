---
title: "Memory Based Collaborative Filtering (User-User and Item-Item)"
excerpt: "Let's study about memory based collaborative filtering for recommender system "
categories:
 - study
tags:
 - recommender system
use_math: true
last_modified_at: "2020-06-25"
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
import os, sys, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time
from ipypb import track, chain

np.set_printoptions(precision=3)
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
# tags = pd.read_csv(os.path.join(DFILE, 'tags.csv'), encoding=CHARSET)
movies = pd.read_csv(os.path.join(DFILE, 'movies.csv'), encoding=CHARSET)
```

</div>

# Memory Based Collabortive Filtering

1. **User-User Collaborative Filtering**: Here we find look alike users based on similarity and recommend movies which first user’s look-alike has chosen in past. This algorithm is very effective but takes a lot of time and resources. It requires to compute every user pair information which takes time. Therefore, for big base platforms, this algorithm is hard to implement without a very strong parallelizable system.
2. **Item-Item Collaborative Filtering**: It is quite similar to previous algorithm, but instead of finding user's look-alike, we try finding movie's look-alike. Once we have movie's look-alike matrix, we can easily recommend alike movies to user who have rated any movie from the dataset. This algorithm is **far less resource consuming than user-user collaborative filtering**. Hence, for a new user, the algorithm takes far lesser time than user-user collaborate **as we don’t need all similarity scores between users**.

<img src="https://miro.medium.com/max/1400/0*47JcAfVNfKtLsjAu.jpeg" width="450">

## Step 1. Preprocessing
Now I use the **scikit-learn library** to split the dataset into testing and training.  **Cross_validation.train_test_split** shuffles and splits the data into two datasets according to the percentage of test examples, which in this case is 0.2.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
df_train, df_test = train_test_split(ratings, test_size=0.2, random_state=0, shuffle=True)
df_train.shape, df_test.shape

R = ratings.pivot(index='userId', columns='movieId', values='rating')
M, N = R.shape
print("num_users: {}, num_movies: {}".format(M, N))
print("density rate: {:.2f}%".format((1 - (R.isna().sum(axis=0).sum() / (M * N))) * 100))
R
```

</div>

{:.output_stream}

```
num_users: 610, num_movies: 9724
density rate: 1.70%

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
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>606</th>
      <td>2.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>607</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>608</th>
      <td>2.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>609</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>610</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 9724 columns</p>
</div>
</div>



Data sparseness can be visualized as follows.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
plt.imshow(R)
plt.grid(False)
plt.xlabel("item")
plt.ylabel("user")
plt.title("train Matrix")
plt.show()
```

</div>


![png](/assets/images/memory_collaborative_filtering_files/memory_collaborative_filtering_6_0.png)


## Step 2. Calculate Similarity

At first, calculate similiarity scores for a toy example.

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# matrix = [[5,4,None,4,2,None,4,1],[None,5,5,4,2,1,2,None],[1,None,1,5,None,5,3,4]]
# users = ['A','B','C']
# items = ['a','b','c','d','e','f','g','h']

matrix = [[4, None, 5, 5], [4, 2, 1, None], [3, None, 2, 4], [4, 4, None, None], [2, 1, 3, 5]]
users = ['u1', 'u2', 'u3', 'u4', 'u5']
items = ['i1', 'i2', 'i3', 'i4']
df_table = pd.DataFrame(matrix, index=users, columns=items, dtype=float)
df_table
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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>u2</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u3</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>u4</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u5</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



### Pearson Similarity

[Pearson Similarity](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)  can be computed as follows.

Let the number of users be $M$ and the number of items be $N$.

The dimension of pearson correlation similarity matrices for users and items are 
$ M \times M $, $ N \times N $. 

### User-Based(UB)
$u, v$ are **users** <br>
**$I$: item set being co-rated by both user $u$ and user $v$**

$$
sim(u, v) = \frac{\sum_{i \in I} (r_{u, i} - \bar{r_u}) (r_{v, i} - \bar{r_v})}{ \sqrt{\sum_{i \in I} (r_{u, i} - \bar{r_u})^2} \sqrt{\sum_{i \in I} (r_{v, i} - \bar{r_v})^2}}
$$

### Item-Based(IB)
<span style="color:red">Note that</span> **item-based similarity is modified**. <br>
Adjusted cosine similarity is used. <br>
Please see [Item-Based Collaborative Filtering Recommendation
Algorithms by Sarwar](https://dl.acm.org/doi/pdf/10.1145/371920.372071)
> Basic cosin similarity has one import drawback that difference in rating scale between different users are not takend into account. Therefore, the adjusted cosine similarity offset this drawback by sub-tracking the corresponding user average from each co-rated pair.
The formula of adjusted cosine similarity as follows. <br>
**$U$: users set that rated both item $i$ and item $j$ (co-rated  pair)**

$$
sim(i, j) = \frac{\sum_{u \in U} (r_{u,i} - \bar{r_u}) (r_{u,j} - \bar{r_u})}{ \sqrt{\sum_{u \in U} (r_{u,i} - \bar{r_u})^2} \sqrt{\sum_{u \in U} (r_{u,j} - \bar{r_u})^2}}
$$

<details>
    if neighbor does not exist, substiture the similarity value as 0.
</details>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
ratings = df_table.to_numpy()
M, N = ratings.shape

def ratedset(ratings, i, kind):
    """ if kind is True, return indices of item set else user set. """
    rates = ratings[i] if kind else ratings[:, i]
    where = np.argwhere(~np.isnan(rates)).flatten()
    return set(where)

def neighbors(ratings, i, j, kind):
    """ return neighbors list. """
    return list(ratedset(ratings, i, kind).intersection(ratedset(ratings, j, kind)))
# neighbors(0, 4, ub=True)
# neighbors(0, 2, ub=False)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def pearson(ratings, ub, adjusted=False):
    M, N = ratings.shape
    epsilon = 1e-20
    if ub:
        sim = np.zeros(shape=(M, M), dtype=float)
        for u in track(range(M)):
            for v in range(u, M):
                # find sim(u, v)
                nei = neighbors(ratings, u, v, kind=True) # indices of common items
                if not nei:
                    sim[u][v] = sim[v][u] = np.nan
                    continue
                ru = ratings[u][nei] - np.mean(ratings[u][nei])
                rv = ratings[v][nei] - np.mean(ratings[v][nei])
                up = ru.dot(rv)
                down = sqrt(np.sum(ru**2)) * sqrt(np.sum(rv**2))
                sim[u][v] = sim[v][u] = up / down
        return sim
    else: 
        sim = np.zeros(shape=(N, N), dtype=float)
        if adjusted:
            umeans = np.nanmean(ratings, axis=1)
        for i in track(range(N)):
            for j in range(i, N):
                # find sim(i, j)
                nei = neighbors(ratings, i, j, kind=False) # indices of common users
                if not nei:
                    sim[i][j] = sim[j][i] = np.nan
                    continue
                if adjusted:
                    ri = ratings[nei, i] - umeans[nei]
                    rj = ratings[nei, j] - umeans[nei]
                else:
                    ri = ratings[nei, i] - np.mean(ratings[nei, i])
                    rj = ratings[nei, j] - np.mean(ratings[nei, j])
                up = ri.dot(rj)
                down = sqrt(np.sum(ri**2)) * sqrt(np.sum(rj**2))
                sim[i][j] = sim[j][i] = up / down
        return sim
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
df_table
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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>u2</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u3</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>u4</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u5</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
usim_toy = pearson(ratings, ub=True, verbose=True)
pd.DataFrame(usim_toy, index=users, columns=users)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="5" value="5" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">5/5</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
/home/swyoo/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:18: RuntimeWarning: invalid value encountered in double_scalars

```

{:.output_stream}

```
WorkingTime[pearson]: 5.10669 ms

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
      <th>u1</th>
      <th>u2</th>
      <th>u3</th>
      <th>u4</th>
      <th>u5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u1</th>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.755929</td>
    </tr>
    <tr>
      <th>u2</th>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.327327</td>
    </tr>
    <tr>
      <th>u3</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.654654</td>
    </tr>
    <tr>
      <th>u4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u5</th>
      <td>0.755929</td>
      <td>-0.327327</td>
      <td>0.654654</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
# pandas library provides calcultation of pearson correlation.
df_table.T.corr(method='pearson', min_periods=1)
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
      <th>u1</th>
      <th>u2</th>
      <th>u3</th>
      <th>u4</th>
      <th>u5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u1</th>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.755929</td>
    </tr>
    <tr>
      <th>u2</th>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.327327</td>
    </tr>
    <tr>
      <th>u3</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.654654</td>
    </tr>
    <tr>
      <th>u4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u5</th>
      <td>0.755929</td>
      <td>-0.327327</td>
      <td>0.654654</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
# This calucated values are adjusted cosine similarity
# so, it is different with pearson correlation values.
isim_toy = pearson(ratings, ub=False, adjusted=True, verbose=True)
pd.DataFrame(isim_toy, index=items, columns=items)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="4" value="4" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">4/4</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[pearson]: 3.65472 ms

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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>i1</th>
      <td>1.000000</td>
      <td>0.232485</td>
      <td>-0.787493</td>
      <td>-0.765945</td>
    </tr>
    <tr>
      <th>i2</th>
      <td>0.232485</td>
      <td>1.000000</td>
      <td>0.002874</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>i3</th>
      <td>-0.787493</td>
      <td>0.002874</td>
      <td>1.000000</td>
      <td>-0.121256</td>
    </tr>
    <tr>
      <th>i4</th>
      <td>-0.765945</td>
      <td>-1.000000</td>
      <td>-0.121256</td>
      <td>1.000000</td>
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
isim_toy_adjust = pearson(ratings, ub=False, adjusted=False, verbose=True)
pd.DataFrame(isim_toy_adjust, index=items, columns=items)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="4" value="4" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">4/4</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
/home/swyoo/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:39: RuntimeWarning: invalid value encountered in double_scalars

```

{:.output_stream}

```
WorkingTime[pearson]: 3.94559 ms

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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>i1</th>
      <td>1.000000</td>
      <td>0.755929</td>
      <td>0.050965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>i2</th>
      <td>0.755929</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>i3</th>
      <td>0.050965</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>0.755929</td>
    </tr>
    <tr>
      <th>i4</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.755929</td>
      <td>1.000000</td>
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
df_table.corr(method='pearson', min_periods=1)
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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>i1</th>
      <td>1.000000</td>
      <td>0.755929</td>
      <td>0.050965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>i2</th>
      <td>0.755929</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>i3</th>
      <td>0.050965</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>0.755929</td>
    </tr>
    <tr>
      <th>i4</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.755929</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Step 3. Predict unrated scores <br>

### User-Based 
$$
\hat{r}_{a, i} = \bar{r_a} + \frac{ \sum_{b \in nei(i)} sim(a, b) * (r_{b, i} - \bar{r_b})}{\sum_{b \in nei(i)} |sim(a, b)|}
$$

where 
* $sim(a,b), r_{b, i} - \bar{r_b}$ are scalar. <br>
* $sim(a,b)$ means a pearson similarity score between user a and user b.
* $r_{b, i} - \bar{r_b}$ means a row of normalized rating corresponding to user b.
* $nei(i)$ means users who have rated the item $i$; $\vert nei(i) \vert \le M$

### Item-Based 
$$
\hat{r}_{a, i} = \bar{r_i} + \frac{ \sum_{j \in nei(a)} sim(i, j) * (r_{a, j} - \bar{r_j})}{\sum_{j \in nei(a)} |sim(i, j)|}
$$
where 
* $nei(a)$ means items who "user a" have rated; $\vert nei(a) \vert \le N$

<details>
    if neighbor does not exist, substiture the value as 0.
</details>

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def predict(ratings, sim, entries, ub):
    epsilon = 1e-20
    sim = np.nan_to_num(sim)
    
    def pred(a, i):
        if ub:
            nei = list(ratedset(ratings, i, kind=False))  # users set who have rated "movie i".
            if not nei: return np.nanmean(ratings[a])
            ru = np.ones(shape=len(nei))
            for k, u in enumerate(nei):
                overlap = neighbors(ratings, a, u, kind=True)
                ru_mean = np.mean(ratings[u, overlap]) if overlap else 0
                ru[k] = ratings[u][i] - ru_mean
            term = ru.dot(sim[a][nei]) / (epsilon + np.sum(np.abs(sim[a][nei])))
            return np.nanmean(ratings[a]) + term
        
        else:
            nei = list(ratedset(ratings, a, kind=True)) # items set that "user a" rated.
            if not nei: return np.nanmean(ratings[:, i])
            rj = np.ones(shape=len(nei))
            for k, j in enumerate(nei):
                overlap = neighbors(ratings, i, j, kind=False)
                rj_mean = np.mean(ratings[overlap, j]) if overlap else 0
                rj[k] = ratings[a][j] - rj_mean
            term = rj.dot(sim[i][nei]) / (epsilon + np.sum(np.abs(sim[i][nei])))
            return np.nanmean(ratings[:, i]) + term
        
    return np.nan_to_num(np.array([pred(u, i) for u, i in track(entries)]))
```

</div>

<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
df_table
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
      <th>i1</th>
      <th>i2</th>
      <th>i3</th>
      <th>i4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>u2</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u3</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>u4</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>u5</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
entries = np.argwhere(np.isnan(ratings))
usim = df_table.T.corr(method='pearson').to_numpy()
isim = df_table.corr(method='pearson').to_numpy()
ub = predict(ratings, usim, entries, ub=True, verbose=True)  # UB
ib = predict(ratings, isim, entries, ub=False, verbose=True)  # IB
print(entries)
print("\npredicted results as follows...")
print(ub)
print(ib)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="5" value="5" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">5/5</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 12.54416 ms

```


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="5" value="5" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">5/5</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 7.43055 ms
[[0 1]
 [1 3]
 [2 1]
 [3 2]
 [3 3]]

predicted results as follows...
[3.947 2.341 1.775 4.    4.   ]
[0.912 2.333 2.19  0.408 4.667]

```

## Step 4. Apply to movielen Dataset

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
ratings = pd.read_csv(os.path.join(DFILE, 'ratings.csv'), encoding=CHARSET)

samples = ratings.sample(frac=1)
df_train, df_test = train_test_split(samples, test_size=0.1, random_state=0, shuffle=True)
df_train.shape, df_test.shape

R = samples.pivot(index='userId', columns='movieId', values='rating')
M, N = R.shape
print("num_users: {}, num_movies: {}".format(M, N))
print("density rate: {:.2f}%".format((1 - (R.isna().sum(axis=0).sum() / (M * N))) * 100))
```

</div>

{:.output_stream}

```
num_users: 610, num_movies: 9724
density rate: 1.70%

```

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
df_train.shape, df_test.shape
```

</div>




{:.output_data_text}

```
((90752, 4), (10084, 4))
```



<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
mid2idx = {mid: i for i, mid in enumerate(R.columns)}
idx2mid = {v: k for k, v in mid2idx.items()}
uid2idx = {uid: i for i, uid in enumerate(R.index)}
idx2uid = {v: k for k, v in uid2idx.items()}
rmatrix = R.to_numpy().copy()
rmatrix
```

</div>




{:.output_data_text}

```
array([[4. , nan, 4. , ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan],
       ...,
       [2.5, 2. , 2. , ..., nan, nan, nan],
       [3. , nan, nan, ..., nan, nan, nan],
       [5. , nan, nan, ..., nan, nan, nan]])
```



### 4 - 1. conceal test and validation dataset

<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
for uid, mid in zip(df_test.userId, df_test.movieId):
    uidx, midx = uid2idx[uid], mid2idx[mid]
    rmatrix[uidx][midx] = np.nan
    
rtable = pd.DataFrame(rmatrix, index=uid2idx.keys(), columns=mid2idx.keys())
rtable
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
      <th>1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>606</th>
      <td>2.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>607</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>608</th>
      <td>2.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>609</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>610</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 9724 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[21]:
</div>

<div class="input_area" markdown="1">

```python
M, N = rtable.shape
print("num_users: {}, num_movies: {}".format(M, N))
print("density rate: {:.2f}%".format((1 - (rtable.isna().sum(axis=0).sum() / (M * N))) * 100))
```

</div>

{:.output_stream}

```
num_users: 610, num_movies: 9724
density rate: 1.53%

```

### 4 - 2. calculate similarity and predict ratings.

<div class="prompt input_prompt">
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
entries = [(uid2idx[uid], mid2idx[mid]) for uid, mid in zip(df_test.userId, df_test.movieId)]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[23]:
</div>

<div class="input_area" markdown="1">

```python
usim = rtable.T.corr(method='pearson')
upreds = predict(rmatrix, usim, entries, ub=True, verbose=True)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="10084" value="10084" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">9700/10084</span>
<span class="Time-label">[01:31<00:00, 0.01s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 91141.55078 ms

```

<div class="prompt input_prompt">
In&nbsp;[24]:
</div>

<div class="input_area" markdown="1">

```python
isim = rtable.corr(method='pearson')
ipreds = predict(rmatrix, isim, entries, ub=False, verbose=True)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="10084" value="10084" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">9600/10084</span>
<span class="Time-label">[05:38<00:00, 0.03s/it]</span></div>
</div>


{:.output_stream}

```
/home/swyoo/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:27: RuntimeWarning: Mean of empty slice

```

{:.output_stream}

```
WorkingTime[predict]: 338750.83947 ms

```

This calucated values are adjusted cosine similarity <br>
so, it is different with pearson correlation values. <br>

<div class="prompt input_prompt">
In&nbsp;[25]:
</div>

<div class="input_area" markdown="1">

```python
isim_adjust = pearson(rmatrix, ub=False, adjusted=True, verbose=True)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="9724" value="9724" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">9021/9724</span>
<span class="Time-label">[34:59<00:00, 0.22s/it]</span></div>
</div>


{:.output_stream}

```
/home/swyoo/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:39: RuntimeWarning: invalid value encountered in double_scalars

```

{:.output_stream}

```
WorkingTime[pearson]: 2099248.79313 ms

```

<div class="prompt input_prompt">
In&nbsp;[28]:
</div>

<div class="input_area" markdown="1">

```python
ipreds_adjusted = predict(rmatrix, isim_adjust, entries, ub=False, verbose=True)
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="10084" value="10084" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">9600/10084</span>
<span class="Time-label">[05:34<00:00, 0.03s/it]</span></div>
</div>


{:.output_stream}

```
/home/swyoo/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:27: RuntimeWarning: Mean of empty slice

```

{:.output_stream}

```
WorkingTime[predict]: 335237.43176 ms

```

## Step 5. Evaluation
There are many evaluation metrics but one of the most popular metric used to evaluate accuracy of predicted ratings is **Root Mean Squared Error (RMSE)**. I will use the **mean_square_error (MSE)** function from sklearn, where the RMSE is just the square root of MSE.

$$\mathit{RMSE} =\sqrt{\frac{1}{N} \sum (x_i -\hat{x_i})^2}$$

<div class="prompt input_prompt">
In&nbsp;[29]:
</div>

<div class="input_area" markdown="1">

```python
def rmse(ratings, preds, entries):
    """ calculate RMSE
    ratings: np.array
    preds: List[float]
    entries: List[List[int]] """
    N = len(preds)
    diff = np.zeros_like(preds)
    for k, (i, j) in enumerate(entries):
        diff[k] = ratings[i][j] - preds[k]
    return sqrt(np.sum(diff ** 2) / N)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[30]:
</div>

<div class="input_area" markdown="1">

```python
entries = [(uid2idx[uid], mid2idx[mid]) for uid, mid in zip(df_test.userId, df_test.movieId)]
rmse1 = rmse(R.to_numpy(), upreds, entries)
rmse2 = rmse(R.to_numpy(), ipreds, entries)
rmse3 = rmse(R.to_numpy(), ipreds_adjusted, entries)
print('User-based CF RMSE: {:.3f}'.format(rmse1))
print('Item-based CF RMSE: {:.3f}'.format(rmse2))
print('Item-based CF RMSE with ajdusted: {:.3f}'.format(rmse3))
```

</div>

{:.output_stream}

```
User-based CF RMSE: 0.892
Item-based CF RMSE: 1.127
Item-based CF RMSE with ajdusted: 1.151

```

## Recommend Top K Movies

Assume that our recommender system recommends **top K movies** to a user. <br>
The recommendation precedure is as follows.
1. **Predict scores** for unseen movies by utilizing **similarity matrix**.
2. Sort and **determine Top K movies** from predicted scores for unseen movies.
3. **Recommend the top K movies**.

Before recommend movies, let's look at the user's profile.

<div class="prompt input_prompt">
In&nbsp;[31]:
</div>

<div class="input_area" markdown="1">

```python
M, N = R.shape
ratings = R.to_numpy()
uid = random.randint(1, M)
print(uid)
R[R.index==uid]
```

</div>

{:.output_stream}

```
425

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
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 9724 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[104]:
</div>

<div class="input_area" markdown="1">

```python
seen_indices = np.argwhere(~np.isnan(np.squeeze(R[R.index==uid].to_numpy()))).flatten()
seen = [idx2mid[i] for i in seen_indices]
user_profile = pd.merge(left=movies[movies.movieId.isin(seen)], right=R[R.index==uid][seen].T, on='movieId')
user_profile.columns = ['movieId', 'title', 'genres', 'rating']
user_profile.sort_values(by=['rating'], ascending=False)
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>72</th>
      <td>778</td>
      <td>Trainspotting (1996)</td>
      <td>Comedy|Crime|Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>356</td>
      <td>Forrest Gump (1994)</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>318</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>Crime|Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>293</td>
      <td>Léon: The Professional (a.k.a. The Professiona...</td>
      <td>Action|Crime|Drama|Thriller</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1061</td>
      <td>Sleepers (1996)</td>
      <td>Thriller</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>259</th>
      <td>6764</td>
      <td>Rundown, The (2003)</td>
      <td>Action|Adventure|Comedy</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>68</th>
      <td>736</td>
      <td>Twister (1996)</td>
      <td>Action|Adventure|Romance|Thriller</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>69</th>
      <td>741</td>
      <td>Ghost in the Shell (Kôkaku kidôtai) (1995)</td>
      <td>Animation|Sci-Fi</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>71</th>
      <td>762</td>
      <td>Striptease (1996)</td>
      <td>Comedy|Crime</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>64</th>
      <td>608</td>
      <td>Fargo (1996)</td>
      <td>Comedy|Crime|Drama|Thriller</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
<p>306 rows × 4 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
def recommend(R, sim, uid, K):
    """ R: pandas.Dataframe, rating pivot. """
    rates = np.squeeze(R[R.index==uid].to_numpy())
    empties = np.argwhere(np.isnan(rates)).flatten()
    entries = [(uid, mid) for mid in empties]
    preds = predict(ratings, sim, entries, ub=True, verbose=True)
    topK_indices = np.argsort(preds)[-K:][::-1]
    topK = [idx2mid[idx] for idx in topK_indices]
    return topK
```

</div>

<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
topK = recommend(R, usim, uid, K=10)
print(topK)
movies[movies.movieId.isin(topK)]
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="9418" value="9418" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">8554/9418</span>
<span class="Time-label">[00:13<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 13215.18350 ms
[6863, 63768, 87867, 60941, 147250, 72603, 5959, 7264, 7394, 5328]

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
      <th>3807</th>
      <td>5328</td>
      <td>Rain (2001)</td>
      <td>Drama|Romance</td>
    </tr>
    <tr>
      <th>4143</th>
      <td>5959</td>
      <td>Narc (2002)</td>
      <td>Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>4608</th>
      <td>6863</td>
      <td>School of Rock (2003)</td>
      <td>Comedy|Musical</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>7264</td>
      <td>An Amazing Couple (2002)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>4929</th>
      <td>7394</td>
      <td>Those Magnificent Men in Their Flying Machines...</td>
      <td>Action|Adventure|Comedy</td>
    </tr>
    <tr>
      <th>6812</th>
      <td>60941</td>
      <td>Midnight Meat Train, The (2008)</td>
      <td>Horror|Mystery|Thriller</td>
    </tr>
    <tr>
      <th>6899</th>
      <td>63768</td>
      <td>Tattooed Life (Irezumi ichidai) (1965)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>7195</th>
      <td>72603</td>
      <td>Merry Madagascar (2009)</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>7637</th>
      <td>87867</td>
      <td>Zookeeper (2011)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>9138</th>
      <td>147250</td>
      <td>The Adventures of Sherlock Holmes and Doctor W...</td>
      <td>(no genres listed)</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
topK = recommend(R, isim, uid, K=10)
print(topK)
movies[movies.movieId.isin(topK)]
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="9418" value="9418" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">0/9418</span>
<span class="Time-label">[00:14<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 14716.72630 ms
[166558, 4215, 4235, 4234, 4233, 4232, 4231, 4229, 4228, 4226]

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
      <th>3132</th>
      <td>4215</td>
      <td>Revenge of the Nerds II: Nerds in Paradise (1987)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>4226</td>
      <td>Memento (2000)</td>
      <td>Mystery|Thriller</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>4228</td>
      <td>Heartbreakers (2001)</td>
      <td>Comedy|Crime|Romance</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>4229</td>
      <td>Say It Isn't So (2001)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>4231</td>
      <td>Someone Like You (2001)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>4232</td>
      <td>Spy Kids (2001)</td>
      <td>Action|Adventure|Children|Comedy</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>4233</td>
      <td>Tomcats (2001)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>4234</td>
      <td>Tailor of Panama, The (2001)</td>
      <td>Drama|Thriller</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>4235</td>
      <td>Amores Perros (Love's a Bitch) (2000)</td>
      <td>Drama|Thriller</td>
    </tr>
    <tr>
      <th>9435</th>
      <td>166558</td>
      <td>Underworld: Blood Wars (2016)</td>
      <td>Action|Horror</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[35]:
</div>

<div class="input_area" markdown="1">

```python
topK = recommend(R, isim_adjust, uid, K=10)
print(topK)
movies[movies.movieId.isin(topK)]
```

</div>


<div markdown="0">
<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="9418" value="9418" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">6204/9418</span>
<span class="Time-label">[00:14<00:00, 0.00s/it]</span></div>
</div>


{:.output_stream}

```
WorkingTime[predict]: 15062.08897 ms
[70301, 72696, 6577, 1024, 3689, 6301, 4883, 1551, 1490, 81156]

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
      <th>782</th>
      <td>1024</td>
      <td>Three Caballeros, The (1945)</td>
      <td>Animation|Children|Musical</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>1490</td>
      <td>B*A*P*S (1997)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>1170</th>
      <td>1551</td>
      <td>Buddy (1997)</td>
      <td>Adventure|Children|Drama</td>
    </tr>
    <tr>
      <th>2751</th>
      <td>3689</td>
      <td>Porky's II: The Next Day (1983)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3566</th>
      <td>4883</td>
      <td>Town is Quiet, The (Ville est tranquille, La) ...</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4313</th>
      <td>6301</td>
      <td>Straw Dogs (1971)</td>
      <td>Drama|Thriller</td>
    </tr>
    <tr>
      <th>4454</th>
      <td>6577</td>
      <td>Kickboxer 2: The Road Back (1991)</td>
      <td>Action|Drama</td>
    </tr>
    <tr>
      <th>7092</th>
      <td>70301</td>
      <td>Obsessed (2009)</td>
      <td>Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>7201</th>
      <td>72696</td>
      <td>Old Dogs (2009)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>7442</th>
      <td>81156</td>
      <td>Jackass 3D (2010)</td>
      <td>Action|Comedy|Documentary</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Reference

[1] [Item-Based Recommender System original paper written by sarwar 2001](http://files.grouplens.org/papers/www10_sarwar.pdf)

[2] [Survey paper](http://downloads.hindawi.com/archive/2009/421425.pdf)

[3] [Korean Blog](https://yeo0.github.io/data/2019/02/21/Recommendation-System_Day6/)

[4] [English Blog](https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d)

[5] [Amazon Item-Based Recommeder System 2003](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1167344)

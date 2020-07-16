---
title: "Unsupervised Clustering: K-means and K-medoids"
excerpt: "Let's learn about representative unsupervised clustering algorithms"
categories:
 - study
tags:
 - clustering
use_math: true
last_modified_at: "2020-07-16"
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
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
```

</div>

# Unsupervised Clustering 

Among Clustering methods, [K-means](https://en.wikipedia.org/wiki/K-means_clustering) and [K-medoiod](https://en.wikipedia.org/wiki/K-medoids) belong to [Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning). <br> 
Also, they are in [EM algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) categories. 

## Notations
$K$ = # of clusters <br>
$N$ = # of dataset <br>
$t$ = # of iteration <br> 
$p$ = the dimension of the data <br>
$m_k$ = # a centroid of cluster $k$ <br>
                                    
The objective of $K$-means clustering and $K$-medoids is same, which is to find $K$ clustering given dataset by minimizing the cost as you can see below.  
$$ \underset{C}{\operatorname{argmin}}\sum_{k=1}^{K}{\sum_{x \in C_k}{d(x,m_k)}} $$ where $d(x,m_k)$ means squared Euclidean distance 
                                    
**The big difference** of K-means and K-mediods is **the way how the $m_k$ are updated**

## Load Toy example, and Test dataset

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# Load data
toy = np.load('./data/clustering/data_4_1.npy') # data_1, for visualization
test = np.load('./data/clustering/data_4_2.npz') # data_2, for test 
X = test['X']
y = test['y']

# Sanity check
print('data shape: ', toy.shape) # (1500, 2)
print('X shape: ', X.shape) # (1500, 6)
print('y shape: ', y.shape) # (1500,)

df_toy = pd.DataFrame(toy, columns=['f1', 'f2'])
df_toy  # toy example does not have ground truth label.
```

</div>

{:.output_stream}

```
data shape:  (1500, 2)
X shape:  (1500, 6)
y shape:  (1500,)

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
      <th>f1</th>
      <th>f2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.630551</td>
      <td>2.254497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.452222</td>
      <td>-7.716821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.408706</td>
      <td>-5.010174</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.444588</td>
      <td>-0.194091</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.929980</td>
      <td>-7.329755</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>3.054506</td>
      <td>-3.679957</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>0.282762</td>
      <td>-8.043163</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>3.323046</td>
      <td>-6.140348</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>0.779888</td>
      <td>-8.446471</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>1.227547</td>
      <td>-9.741289</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 2 columns</p>
</div>
</div>



Visualize toy examples

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
toy = df_toy.to_numpy()
plt.scatter(toy[:,0], toy[:,-1], color = 'orange', s = 7)
```

</div>




{:.output_data_text}

```
<matplotlib.collections.PathCollection at 0x7fa4de1f9410>
```




![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_5_1.png)


<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
test = pd.DataFrame(X, columns=features)
test = pd.merge(left=test, right=pd.DataFrame(y, columns=['label']), on=test.index)
test = test[features + ['label']]
test
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
      <th>f1</th>
      <th>f2</th>
      <th>f3</th>
      <th>f4</th>
      <th>f5</th>
      <th>f6</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.200999</td>
      <td>-0.506804</td>
      <td>-0.024489</td>
      <td>-0.039783</td>
      <td>0.067635</td>
      <td>-0.029724</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.572715</td>
      <td>1.414097</td>
      <td>0.097670</td>
      <td>0.155350</td>
      <td>-0.058382</td>
      <td>0.113278</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.183954</td>
      <td>-0.163354</td>
      <td>0.045867</td>
      <td>-0.257671</td>
      <td>-0.015087</td>
      <td>-0.047957</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.540300</td>
      <td>0.308409</td>
      <td>0.007325</td>
      <td>-0.068467</td>
      <td>-0.130611</td>
      <td>-0.016363</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.995599</td>
      <td>1.275279</td>
      <td>-0.015923</td>
      <td>0.062932</td>
      <td>-0.072392</td>
      <td>0.034399</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1495</th>
      <td>3.054506</td>
      <td>-3.679957</td>
      <td>0.042136</td>
      <td>0.012698</td>
      <td>0.121173</td>
      <td>0.076322</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>0.282762</td>
      <td>-8.043163</td>
      <td>0.049892</td>
      <td>-0.055319</td>
      <td>-0.031131</td>
      <td>0.213752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>3.323046</td>
      <td>-6.140348</td>
      <td>-0.104975</td>
      <td>-0.257898</td>
      <td>-0.000928</td>
      <td>-0.094528</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>0.779888</td>
      <td>-8.446471</td>
      <td>0.105319</td>
      <td>0.166633</td>
      <td>0.018645</td>
      <td>0.108826</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>1.227547</td>
      <td>-9.741289</td>
      <td>0.010518</td>
      <td>-0.121976</td>
      <td>-0.020301</td>
      <td>-0.077125</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 7 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# exploration
print("min info:\n", test.min())
print("max info:\n", test.max())
```

</div>

{:.output_stream}

```
min info:
 f1       -2.802532
f2      -11.719322
f3       -0.291218
f4       -0.378744
f5       -0.336366
f6       -0.383337
label     0.000000
dtype: float64
max info:
 f1       17.022084
f2        7.953588
f3        0.402071
f4        0.310862
f5        0.316762
f6        0.342837
label     2.000000
dtype: float64

```

## $K$-means Clustering 

### Procedure
1. Initialization: Choose $K$ initial cluster centriods
2. Expectation
    * Compute point-to-cluster-centroid distances of all data points to each centroid
    * Assign each point to the cluster with the closest centroid.    
$$
\text{assign } x_i \text{ to the cluster label } C_k \text{such that } k = \underset{k}{\operatorname{argmin}}{d(x_i, m_k)} ,~ \forall i \in [1, 2, ..., N]
$$
3. Maximization
    * Update the centroid values: **the average of the points in each cluster** 
    
$$
m_k = \mathbb{E}(x|x \in C_k) = \frac{1}{\vert C_k \vert}{\sum_{x \in C_k}{x} }, \forall k
$$

4. Repeat 2, 3 until satisfying stopping condition: converge(by computing loss) or iteration over. 

$$
loss = \sum_{k=1}^{K} \sum_{x \in C_k} d(x,m_k)
$$

so, the time complexity is $O(tpKN)$

Going through iteration, loss is monotonically decreases, 
and finally loss is converged. 

#### Initialization
Initialize $K$ centriods and visualize it to 2D space for given toy examples. <br>
star stickers means $K$ centriods. 

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
K = 2
data = toy
p = np.size(data, axis=1)  # the number of features.
print("the number of features:", p)
centroids = data[random.sample(range(len(data)), K), :]  # [K, p], choose K centroids.
print(centroids)
# 2D visualization 
plt.scatter(data[:, 0], data[:, 1], c='orange', s=7)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='b', s=150)
plt.show()
```

</div>

{:.output_stream}

```
the number of features: 2
[[  4.77497947  -5.60411467]
 [  2.744412   -11.12111773]]

```


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_10_1.png)


#### Expectation

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
def distance_matrix(data, centroids, verbose=False):
    """ using broadcasting of numpy, compute distance matrix for each cluster.
    Args:
        data: [N, p]
        centroids: [K, p]
    notations:
        :N: the number of examples, 
        :K: the number of clusters, 
        :p: the feature dimension of each sample.
    return distance matrix, of shape=[N, K]"""
    squares = ((data[:, :, np.newaxis] - centroids.T[np.newaxis, :]) ** 2)  # [N, p, K]
    distances = np.sum(squares, axis=1)  # [N, K]
    if verbose:
        # visualize distance matrix for each cluster.
        df = pd.DataFrame(distances, columns=['C1', 'C2']) 
        print(df)
    return distances

def expectation(data, centroids):
    """ Assigning each value to its closest cluster label. """ 
    distances = distance_matrix(data, centroids)  # [N, K]
    clusters = np.argmin(distances, axis=1) # [N,]
    return clusters
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
distance_matrix(data, centroids, verbose=True)
```

</div>

{:.output_stream}

```
              C1          C2
0     123.467777  276.642807
1       9.858734   11.674608
2       2.219469   37.784922
3      36.395165  141.491565
4      17.761854   17.666598
...          ...         ...
1495    6.662411   55.467025
1496   26.128970   15.533526
1497    2.395656   25.142883
1498   24.039747   11.013091
1499   29.700489    4.204808

[1500 rows x 2 columns]

```




{:.output_data_text}

```
array([[123.46777735, 276.64280678],
       [  9.85873415,  11.67460837],
       [  2.21946911,  37.7849224 ],
       ...,
       [  2.39565643,  25.14288287],
       [ 24.03974697,  11.01309085],
       [ 29.70048881,   4.20480814]])
```



<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
clusters = expectation(data, centroids)
clusters  # [N,]
```

</div>




{:.output_data_text}

```
array([0, 0, 0, ..., 0, 1, 1])
```



Visualize the result of expectation step

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
colors = ['r', 'b', 'g', 'y', 'c', 'm']
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

ax1.scatter(data[:, 0], data[:, 1], c='orange', s=7)
ax1.scatter(centroids[:,0], centroids[:,1], marker='*', c='b', s=150)
ax1.set_title("Before Expectation")

for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax2.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax2.legend()
ax2.scatter(centroids[:,0], centroids[:,1], marker='*', c='#050505', s=150)
ax2.set_title("After Expectation")
plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_16_0.png)


#### Maximization 

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
def maximization(data, clusters, K):
    """ update centroids by taking the average value of each group. """
    centroids = np.zeros((K, np.size(data, axis = 1)))
    for k in range(K):
        group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
        centroids[k] = np.mean(group, axis=0)
    return centroids
```

</div>

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
new_centroids = maximization(data, clusters, K)
new_centroids  # [K, p]
```

</div>




{:.output_data_text}

```
array([[ 5.59722207, -3.28657162],
       [ 1.03157988, -8.83044793]])
```



<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax1.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax1.legend()
ax1.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#050505', s=150)
ax1.set_title("Before Maximization")

for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax2.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax2.legend()
ax2.scatter(new_centroids[:, 0], new_centroids[:,1 ], marker='*', c='#050505', s=150)   
ax2.set_title("After Maximization")

plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_20_0.png)


#### Compute Loss and repeat until converge 

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
data.shape
```

</div>




{:.output_data_text}

```
(1500, 2)
```



<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
def cost(data, clusters, centroids):
    K = len(centroids)
    loss = 0
    for k in range(K):
        group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
        squares = (group - centroids[k][np.newaxis, :]) ** 2  # [*(# of sample in group[k]), p]
        loss += np.sum(np.sum(squares, axis=1), axis=0) # scalar
    return loss
```

</div>

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
def kmeans(data, K, iteration=10, bound=1e-7, verbose=False):
    # Initialization 
    centroids = data[random.sample(range(len(data)), K), :]  # [K, p], choose K centroids.
    # Repeat EM algorithm
    error = 1e7
    while iteration:
        clusters = expectation(data, centroids)  # [N,]
        centroids = maximization(data, clusters, K)  # [K, p]
        loss = cost(data, clusters, centroids) # scalar
        if verbose: print("loss: {}".format(loss))
        if (error - loss) < bound: 
            if verbose: print(error - loss)
            if verbose: print("converge")
            break
        error = loss
        iteration -= 1
    
    return clusters, centroids
```

</div>

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
K = 5
clusters, centroids = kmeans(data, K, verbose=True)
```

</div>

{:.output_stream}

```
loss: 7990.191121047926
loss: 7212.691330989318
loss: 7152.881427488944
loss: 7130.270405265313
loss: 7116.576976812909
loss: 7096.425347518904
loss: 7055.719485522531
loss: 6954.189159846061
loss: 6722.085830042569
loss: 6330.968027295821

```

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
colors = ['r', 'b', 'g', 'y', 'c', 'm']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax.legend()
ax.scatter(centroids[:, 0], centroids[:,1 ], marker='*', c='#050505', s=150)   
ax.set_title("Final Clustering")
plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_26_0.png)


## $K$-medoids Clustering 

### Procedure
1. Initialization: same with $K$-means 
2. Expectation
    * same with $K$-means 
$$
\text{assign } x_i \text{ to the cluster label } C_k \text{such that } k = \underset{k}{\operatorname{argmin}}{d(x_i, m_k)} ,~ \forall i \in [1, 2, ..., N]
$$
3. Maximization
    * For a given cluster assignment $C$, find a data point in each group that minimizes the sum of distances to other, and update the point as a centroid in the group.
points in that group
<br>$C(i)=k$ means the index $i$ of $x_i \in C_k$
$$m_k = x_{i_k^*}$$
$$i_k^* = \underset{i:C(i)=k}{\operatorname{argmin}}{\sum_{C(j)=k}{d(x_i, x_j)}}$$

4. Repeat 2, 3 until satisfying stopping condition: same with $K$-means 

so, the time complexity is $O(tdN^2)$

Implement maximization of $K$-medoids

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
def maximization_v2(data, clusters, K):
    """ update centroids by taking the average value of each group. 
    Description: a**2 - 2ab + b**2 technique used for computing pairwise distance. """
    centroids = np.zeros((K, np.size(data, axis = 1)))
    for k in range(K):
        group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [G, p]
        asquares = np.sum(np.square(group), axis=1)  # [G,]
        bsquares = np.sum(np.square(group), axis=1)  # [G,]
        ab = np.einsum('ip,jp->ij', group, group)  # [G, G]
        # pairwise distances in the group k 
        distances = asquares[:, np.newaxis] - 2 * ab + bsquares[np.newaxis, :] # [G, G]
        sum_dist = np.sum(distances, axis=1)
        i = np.argmin(sum_dist, axis=0)  # this is i_k^*
        centroids[k] = group[i]  # update centroid
    return centroids
```

</div>

Visualization of different computation between K-means and K-mediods.

<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))

# initialization
centroids = data[random.sample(range(len(data)), K), :]  # [K, p], choose K centroids.

# expectation 
clusters = expectation(data, centroids)

# maximization of K-means 
centroids_v1 = maximization(data, clusters, K)

# maximization of K-medoids
centroids_v2 = maximization_v2(data, clusters, K)

for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax.legend()
ax.scatter(centroids_v1[:,0], centroids_v1[:,1], c='orange', marker='*', s=150, 
            label='K means', alpha=0.5)
ax.legend()
ax.scatter(centroids_v2[:,0], centroids_v1[:,1], c='black', marker='*', s=150,
            label='K medoids', alpha=0.5)
ax.legend()
plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_31_0.png)


Implement overall algorithm 

<div class="prompt input_prompt">
In&nbsp;[21]:
</div>

<div class="input_area" markdown="1">

```python
def kmedoids(data, K, iteration=10, bound=1e-7, verbose=False):
    # Initialization 
    centroids = data[random.sample(range(len(data)), K), :]  # [K, p], choose K centroids.
    # Repeat EM algorithm
    error = 1e7
    while iteration:
        clusters = expectation(data, centroids)  # [N,]
        centroids = maximization_v2(data, clusters, K)  # [K, p]
        loss = cost(data, clusters, centroids) # scalar
        if verbose: print("loss: {}".format(loss))
        if (error - loss) < bound: 
            if verbose: print(error - loss)
            if verbose: print("converge")
            break
        error = loss
        iteration -= 1
    
    return clusters, centroids
```

</div>

<div class="prompt input_prompt">
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
K = 5
clusters, centroids = kmedoids(data, K, verbose=True)
```

</div>

{:.output_stream}

```
loss: 6933.893463699935
loss: 5892.018779804686
loss: 5779.071902042521
loss: 5730.8803618351485
loss: 5668.112584773427
loss: 5630.283712467555
loss: 5614.363123046829
loss: 5610.936338275873
loss: 5610.685620916394
loss: 5609.612766647277

```

<div class="prompt input_prompt">
In&nbsp;[23]:
</div>

<div class="input_area" markdown="1">

```python
colors = ['r', 'b', 'g', 'y', 'c', 'm']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
for k in range(K):
    group = np.array([data[j] for j in range(len(data)) if clusters[j] == k])  # [*, p]
    ax.scatter(group[:, 0], group[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax.legend()
ax.scatter(centroids[:, 0], centroids[:,1 ], marker='*', c='#050505', s=150)   
ax.set_title("Final Clustering")
plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_35_0.png)


## Compare Two methods

<div class="prompt input_prompt">
In&nbsp;[24]:
</div>

<div class="input_area" markdown="1">

```python
K = 5
%timeit kmeans(data, K, verbose=False)
%timeit kmedoids(data, K, verbose=False)

clusters_v1, centroids_v1 = kmeans(data, K, verbose=False)
clusters_v2, centroids_v2 = kmedoids(data, K, verbose=False)
```

</div>

{:.output_stream}

```
58.2 ms ± 1.34 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
96.6 ms ± 7.68 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

```

<div class="prompt input_prompt">
In&nbsp;[25]:
</div>

<div class="input_area" markdown="1">

```python
colors = ['r', 'b', 'g', 'y', 'c', 'm']
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
for k in range(K):
    group_v1 = np.array([data[j] for j in range(len(data)) if clusters_v1[j] == k])  # [*, p]
    ax1.scatter(group_v1[:, 0], group_v1[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax1.legend()
    
    group_v2 = np.array([data[j] for j in range(len(data)) if clusters_v2[j] == k])  # [*, p]
    ax2.scatter(group_v2[:, 0], group_v2[:, 1], s=7, c=colors[k], label='cluser {}'.format(k))
    ax2.legend()
    
ax1.scatter(centroids_v1[:, 0], centroids_v1[:,1 ], marker='*', c='#050505', s=150)   
ax2.scatter(centroids_v2[:, 0], centroids_v2[:,1 ], marker='*', c='#050505', s=150)   
ax1.set_title("K-means Clustering")
ax2.set_title("K-medoids Clustering")
plt.show()
```

</div>


![png](/assets/images/UnsupervisedClustering_files/UnsupervisedClustering_38_0.png)


# Test cases

<div class="prompt input_prompt">
In&nbsp;[26]:
</div>

<div class="input_area" markdown="1">

```python
test
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
      <th>f1</th>
      <th>f2</th>
      <th>f3</th>
      <th>f4</th>
      <th>f5</th>
      <th>f6</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.200999</td>
      <td>-0.506804</td>
      <td>-0.024489</td>
      <td>-0.039783</td>
      <td>0.067635</td>
      <td>-0.029724</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.572715</td>
      <td>1.414097</td>
      <td>0.097670</td>
      <td>0.155350</td>
      <td>-0.058382</td>
      <td>0.113278</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.183954</td>
      <td>-0.163354</td>
      <td>0.045867</td>
      <td>-0.257671</td>
      <td>-0.015087</td>
      <td>-0.047957</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.540300</td>
      <td>0.308409</td>
      <td>0.007325</td>
      <td>-0.068467</td>
      <td>-0.130611</td>
      <td>-0.016363</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.995599</td>
      <td>1.275279</td>
      <td>-0.015923</td>
      <td>0.062932</td>
      <td>-0.072392</td>
      <td>0.034399</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1495</th>
      <td>3.054506</td>
      <td>-3.679957</td>
      <td>0.042136</td>
      <td>0.012698</td>
      <td>0.121173</td>
      <td>0.076322</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>0.282762</td>
      <td>-8.043163</td>
      <td>0.049892</td>
      <td>-0.055319</td>
      <td>-0.031131</td>
      <td>0.213752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>3.323046</td>
      <td>-6.140348</td>
      <td>-0.104975</td>
      <td>-0.257898</td>
      <td>-0.000928</td>
      <td>-0.094528</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>0.779888</td>
      <td>-8.446471</td>
      <td>0.105319</td>
      <td>0.166633</td>
      <td>0.018645</td>
      <td>0.108826</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>1.227547</td>
      <td>-9.741289</td>
      <td>0.010518</td>
      <td>-0.121976</td>
      <td>-0.020301</td>
      <td>-0.077125</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 7 columns</p>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[27]:
</div>

<div class="input_area" markdown="1">

```python
data = test.to_numpy()
y = data[:, -1]
data = data[:, :-1]
K = 3
y.shape, data.shape
```

</div>




{:.output_data_text}

```
((1500,), (1500, 6))
```



<div class="prompt input_prompt">
In&nbsp;[28]:
</div>

<div class="input_area" markdown="1">

```python
clusters_v1, centroids_v1 = kmeans(data, K, iteration=100, verbose=False)
clusters_v2, centroids_v2 = kmedoids(data, K, iteration=100, verbose=False)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[29]:
</div>

<div class="input_area" markdown="1">

```python
confusion1 = confusion_matrix(clusters_v1, y)
confusion2 = confusion_matrix(clusters_v2, y)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[30]:
</div>

<div class="input_area" markdown="1">

```python
cluster_labels_true = ['c0', 'c1', 'c2']
cluster_labels_pred = ['g0', 'g1', 'g2']
pd.DataFrame(confusion1, index=cluster_labels_true, columns=cluster_labels_pred)
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
      <th>g0</th>
      <th>g1</th>
      <th>g2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c0</th>
      <td>9</td>
      <td>13</td>
      <td>456</td>
    </tr>
    <tr>
      <th>c1</th>
      <td>59</td>
      <td>487</td>
      <td>44</td>
    </tr>
    <tr>
      <th>c2</th>
      <td>432</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[31]:
</div>

<div class="input_area" markdown="1">

```python
pd.DataFrame(confusion2, index=cluster_labels_true, columns=cluster_labels_pred)
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
      <th>g0</th>
      <th>g1</th>
      <th>g2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c0</th>
      <td>9</td>
      <td>13</td>
      <td>453</td>
    </tr>
    <tr>
      <th>c1</th>
      <td>48</td>
      <td>487</td>
      <td>47</td>
    </tr>
    <tr>
      <th>c2</th>
      <td>443</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



The K-average ans K-method is an unsupervised method. <br>
Therefore, the label order may not match the actual label. <br>
So you need to adjust the order before assessing the performance.  <br>

<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
def change(clusters, confusion):
    reorder = confusion.argmax(axis=-1)
    for i in range(len(clusters)):
        clusters[i] = reorder[clusters[i]]

change(clusters_v1, confusion1)
change(clusters_v2, confusion2)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[35]:
</div>

<div class="input_area" markdown="1">

```python
confusion1 = confusion_matrix(clusters_v1, y)
confusion2 = confusion_matrix(clusters_v2, y)
pd.DataFrame(confusion1, index=cluster_labels_true, columns=cluster_labels_pred)
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
      <th>g0</th>
      <th>g1</th>
      <th>g2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c0</th>
      <td>432</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c1</th>
      <td>59</td>
      <td>487</td>
      <td>44</td>
    </tr>
    <tr>
      <th>c2</th>
      <td>9</td>
      <td>13</td>
      <td>456</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[36]:
</div>

<div class="input_area" markdown="1">

```python
pd.DataFrame(confusion2, index=cluster_labels_true, columns=cluster_labels_pred)
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
      <th>g0</th>
      <th>g1</th>
      <th>g2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c0</th>
      <td>443</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c1</th>
      <td>48</td>
      <td>487</td>
      <td>47</td>
    </tr>
    <tr>
      <th>c2</th>
      <td>9</td>
      <td>13</td>
      <td>453</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
precision_v1 = precision_score(y_true=y, y_pred=clusters_v1, average='macro')
precision_v2 = precision_score(y_true=y, y_pred=clusters_v2, average='macro')
print(precision_v1, precision_v2)

recall_v1 = recall_score(y_true=y, y_pred=clusters_v1, average='macro')
recall_v2 = recall_score(y_true=y, y_pred=clusters_v2, average='macro')
print(recall_v1, recall_v2)
```

</div>

{:.output_stream}

```
0.9264662080703495 0.930151323325496
0.9166666666666666 0.922

```

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
f1_v1 = 2 * (precision_v1 * recall_v1) / (precision_v1 + recall_v1)
f1_v2 = 2 * (precision_v2 * recall_v2) / (precision_v2 + recall_v2)

print(f1_v1, f1_v2)
```

</div>

{:.output_stream}

```
0.9215403863406526 0.9260577246639942

```

# Summary

The objective of K-means clustering and K-medoids is same, which is to find K clustering given dataset by minimizing the cost as you can see below.  
$$ \underset{C}{\operatorname{argmin}}\sum_{k=1}^{K}{\sum_{x \in C_k}{d(x,m_k)}} $$

<br> In this situation, the big difference of K-means and K-mediods is the way how the $m_k$ are updated 

1. $K$-means algorithm 
<br> some properties: 
 - sensitive to initial seed points
 - susceptible to outlier, density, distribution
 - not possible for computing clusters with non-convex shapes
2. $K$-mediod algorithm 
<br> some properties: 
 - applies well when dealing with categorical data, non-vector space data
 - applies well when data point coordinates are not available, but only pair-wise distances are available

## Reference:

[1] [K mean cluster algoritm - ratsgo blog](https://ratsgo.github.io/machine%20learning/2017/04/19/KC/) <br>
[2] [code1](https://mubaris.com/posts/kmeans-clustering/) [code2](https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python) <br>

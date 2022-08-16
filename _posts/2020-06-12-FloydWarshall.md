---
title: "Floyd Warshall Algorithm in Python"
excerpt: "All pair shortest paths algorithm"
categories:
 - algorithms
tags:
 - graph
 - DP
use_math: true
last_modified_at: "2020-06-12"
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
import sys, random, string
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import visualize_graph, logging_time
from utils.generator import randomString
from collections import defaultdict
from pprint import pprint
from copy import deepcopy
from typing import List, Tuple
import numpy as np
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
plot = lambda a: print(np.array(a))

def generate_graph(n, m, randrange:Tuple[int, int] ,verbose=False):
    """ |V|: n, |E|: m """
    S = set(' '.join(string.ascii_lowercase).split()[:n])
    seen = set()
    edges = []
    for _ in range(m):
        while True:
            start = randomString(length=1, samples=list(S))
            end = randomString(length=1, samples=list(S - {start}))
            if (start, end) not in seen: 
                seen.add((start, end))
                break
        edges.append((start, end, random.randint(randrange[0], randrange[1])))
    if verbose: visualize_graph(edges, weighted=True)
    graph = defaultdict(list)
    for i in S: graph[i]
    for u, v, w in edges:
        graph[u].append((v, w))
    return graph, edges

# n, m = 10, 5
# graph = generate_graph(n, m, verbose=True)
# graph
```
# Floyd Warshall algorithm

</div>

## Toy example 
<img src="https://cdn.programiz.com/sites/tutorial2program/files/fw-Graph.png" width="300">

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
# toy example
graph = {'1':[('2', 3), ('4', 5)], 
        '2':[('1', 2), ('4', 4)], 
        '3':[('2', 1)], 
        '4':[('3', 2)]}
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
INF = 1e20
def g2m(graph):
    n, nodes =  len(graph.keys()), sorted(graph.keys())
    n2i = {k: v for k, v in zip(nodes, range(n))}
    weights = [[INF] * n for _ in range(n)]
    for i in nodes: weights[n2i[i]][n2i[i]] = 0
    for i in nodes:
        for j, w in graph[i]:
            weights[n2i[i]][n2i[j]] = w
    return n2i, weights
n2i, weights = g2m(graph)
weights
```

</div>




{:.output_data_text}

```
[[0, 3, 1e+20, 5],
 [2, 0, 1e+20, 4],
 [1e+20, 1, 0, 1e+20],
 [1e+20, 1e+20, 2, 0]]
```

## Constraints
Floyd Warshall Algorithm is an algorithm for **finding shortest paths** <br>
in **a weighted graph** with positive or negative edge weights <br>
(but with <span style="color:red">**no negative cycles**</span>)


## Naive DP

### Idea
$ l_{ij}^{(m)}$: node $i$ 부터 node $j$ 까지 가는데 최대  $m$ 개의 edge를 거쳐서 가는 path의 **minimum weight** <br>
  
>**[intuition]** <br>
총 노드의수 를 $n$ 이라하면, 거쳐가는 edge의 수 $m$이 $n-1$보다 많으면 반복되는 node가 존재한다는 뜻이므로 cycle이 있다는 뜻인데, <br>
negative edge가 없다고 가정했으므로 당연히 cycle을 돌면 shortest path의 wieght sum 보다 높은 path sum이 된다.  <br>
즉, $m = 1$ 부터 $n - 1$ 까지 update하면 optimal solution이 됨
  
$ l_{ij}^{(m)} = {min}{(l_{ij}^{(m-1)},  \underset{1 \le k \le n}{min}{(l_{ik}^{(m-1)} + w_{kj} )} )}  = \underset{1 \le k \le n}{min}{(l_{ik}^{(m-1)} + w_{kj} )} ~~~~ \text{if } m \ge 1$ 
  
$\because k = j $ 이면 $w_{jj}=0$ 이 되므로 case가 합쳐질수 있다. 
  
따라서, recursive formula 는 다음과 같다.  
$$
l_{ij}^{(m)} =
\begin{cases}
\underset{1 \le k \le n}{min}{(l_{ik}^{(m-1)} + w_{kj})} & \text{if } m \ge 1  \\
l_{ij}^{(0) } = 
	\begin{cases} 
		0 & \text{if } i = j\\ 
		\infty & \text{if } i \neq j
    \end{cases} & \text{if } m = 0  \\
\end{cases}
$$

여기서 또 주목할 만한점은 $m=1 $ 일때는 $l_{ij}^{(1)} = w_{ij}$ 이므로 $l_{ij}^{(0)}$을 굳이 계산할 필요는 없다.

### Time Complexity 

$O(n^4)$ $ \because n^3$ entries, each entry takes $O(n)$


### Pseudo Code
```python
# update next step for m 
update(L,W)
	n = len(L)
	let L`[1..n, 1..n] be a new array
	for i = 1 to n 
		for j = 1 to n 
			L`[i,j] = INF
			# each entry can be calculated in O(n)
            for k = 1 to n 
            	L`[i,j] = min(L`[i.j], L[i.j] + W[k,j])
    return L`

# given a graph's weight matrix W[1..n, 1..n]
Naive(W)
	n = len(W)
	let L[1..n, 1..n] be a  new array
	
	# initialization
	L = W
	
	# update L for m 
	for m = 2 to n-1
		L =  update(L, W)
    
    retrun L
```

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def naive(weights):
    n = len(weights)
    def update(a):
        b = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    b[i][j] = min(b[i][j], a[i][k] + weights[k][j])
        return b
    
    ans = weights  # initial states
    for m in range(2, n):
        ans = update(ans)
    
    return ans

plot(naive(weights, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[naive]: 0.05555 ms
[[0 3 7 5]
 [2 0 6 4]
 [3 1 0 5]
 [5 3 2 0]]

```

## Improved Naive


$m$ 에 대해 linear 하게 update 하는 위의 방식을 조금 더 개선해보자. 

L 의 계산이 *associative*(결합법칙을 만족)한 성질을 가지며, $m \ge n - 1$이면 shortest path weight는 고정되어 $L$ 은 바뀌지 않는다.

따라서, associative 하게 $k$번 계산하여 $2^k \ge n-1$ 일 경우 optimal sol으로 고정된다.

따라서, optimal sol에 도달하기 까지 $k = O(logn)$ 번 연산하게 됨.    
$$
\begin{aligned}
L^{(1)} &= W \\
L^{(2)} &= W^2 = WW \\
L^{(4)} &= W^4 = W^2W^2\\
... \\
L^{(2^k)} &= W^{2^k} = W^kW^k\\
L^{(2^k \ge~ n-1)} &= W^{2^k \ge ~ n-1} (fixed)\\
\end{aligned}
$$

### Pseudo Code
```python
# given a graph's weight matrix W[1..n, 1..n]
faster_APSP(W)
	n = len(W)
	let L[1..n, 1..n] be a  new array
	
	# initialization
	L = W
	
	# update L for m 
    m = 2
	while m < n-1
		L =  update(L, L)
		m = m*2
    
    retrun L
```

$O(n^3logn)$ 으로 계선되었지만 여전히 비싼 알고리즘이다. 

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def improved(weights):
    n = len(weights)
    def update(a):
        b = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    b[i][j] = min(b[i][j], a[i][k] + a[k][j])
        return b
    
    ans = weights  # initial states
    m = 1
    while m < n:
        ans = update(ans)
        m *= 2
    
    return ans

plot(improved(weights, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[improved]: 0.05364 ms
[[0 3 7 5]
 [2 0 6 4]
 [3 1 0 5]
 [5 3 2 0]]

```

## Idea of Floyd Warshall Algorithm

All pair shortest path with DP 를 조금더 효율적으로 해보자는 접근

**기본 가정: no negative cycle** 

 먼저 주어진 그래프에 대한 edge정보로 부터 matrix $C$ 정의 
$$
C_{ij} = \left \{ 
\begin{matrix}
0 & \text{if } i=j \\
c(i,j) \ge 0 & \text{if } i \ne j, (i,j) \in E  \\
\infty & \text{if } i \ne j, (i,j) \notin E \\
\end{matrix}\right.
$$
$d_{ij}^{(k)}$: $v_i  \text{~} v_j$ 까지 가는데 $v_1, .., v_k$를 거쳐가는지에 대한 유무가 update된 shortest path distance <br>
($k$ 가 증가함에따라 점점 더 많은 노드정보를 거쳐가는것에 대한 정보를 업데이트 된다).

<img src="/assets/images/FloydWarshall_files/floyd_overview.jpg" width="400">
$$
d_{ij}^{(k)} = \left \{ 
\begin{matrix}
c(i,j) \ge 0 & \text{if } k=0 \\
min \{ d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)}   \} & \text{if } k \ge 1  \\
\end{matrix} \right.
$$

### Time Complexity 
$O(n^3)$ because all entry $(1\le i,j,k\le n)$ ,  is $n^3$, each entry takes $O(n)$ time

Back Propagation: $P^{(k)}$의 각 entry $P_{ij}^{(k)}$가 의미하는것은 현재까지 업데이트된 $v_k$ 를 지나는 $v_i \rightsquigarrow v_k \rightsquigarrow  v_j$ 의 shortest path 정보를 의미한다. <br>
($k = 1,..,n$ 까지 모두 update되어야 진짜 shortest path가 됨)

<figure>
    <img src="/assets/images/FloydWarshall_files/floyd.PNG" width="500">
    <figcaption> Procedure of Floyd Warshall Algorithm </figcaption>
</figure>

### Implementation
[c++ Implementation](https://github.com/SUNGWOOKYOO/Algorithm/blob/master/src_Cplus/graphAlgo/FloydWarshall.cpp) 

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def floyd(weights):
    n = len(weights)
    ans = deepcopy(weights)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                ans[i][j] = min(ans[i][j], ans[i][k] + ans[k][j])
    return ans

plot(floyd(weights, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[floyd]: 0.05054 ms
[[0 3 7 5]
 [2 0 6 4]
 [3 1 0 5]
 [5 3 2 0]]

```

## Application
### Detect Negative Cycles

Floyd Warshall Algorithm based solution is discussed that works for **both connected and disconnected graphs**. <br>
connected의 유무와 상관없이 negative cycle들을 detect할 수 있다!

[wiki](https://www.wikiwand.com/en/Floyd%E2%80%93Warshall_algorithm)의 **Behavior with negative cycles** part 에도 설명이 나와있다. 

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
def hasNcycles(weights, verbose=False):
    n = len(weights)
    ans = deepcopy(weights)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                ans[i][j] = min(ans[i][j], ans[i][k] + ans[k][j])
    
    # check if negative cycle exist
    for i in range(n):
        if ans[i][i] < 0:
            if verbose: print("negative cycle exists from node[{}] to node[{}]".format(i, i))
            return True
    return False

hasNcycles(weights)
```

</div>




{:.output_data_text}

```
False
```



## Analysis

### For small dataset 

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
def generate_graph_no_neg_cycle(n, m, verbose=False):
    weights = graph = None
    while True:
        graph, edges = generate_graph(n, m, randrange=(-5, 10), verbose=verbose)
        n2i, W = g2m(graph)
        if not hasNcycles(W): 
            weights = deepcopy(W)
            return n2i, weights, graph, edges
n, m = 5, 10
n2i, weights, graph, edges = generate_graph_no_neg_cycle(n, m, verbose=False)
visualize_graph(edges=edges, weighted=True)
pprint(graph)
pprint(n2i)
plot(weights)
ans1 = naive(weights, verbose=True)
ans2 = improved(weights, verbose=True)
ans3 = floyd(weights, verbose=True)
# plot(ans1), plot(ans2), plot(ans3)
assert ans1 == ans2 == ans3
plot(ans1)
```

</div>


![png](/assets/images/FloydWarshall_files/FloydWarshall_14_0.png)


{:.output_stream}

```
defaultdict(<class 'list'>,
            {'a': [],
             'b': [('e', -3), ('d', 9), ('c', -1)],
             'c': [('b', 6), ('a', 1)],
             'd': [('e', 7), ('a', 8), ('c', -2)],
             'e': [('a', 6), ('d', 6)]})
{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
[[ 0.e+00  1.e+20  1.e+20  1.e+20  1.e+20]
 [ 1.e+20  0.e+00 -1.e+00  9.e+00 -3.e+00]
 [ 1.e+00  6.e+00  0.e+00  1.e+20  1.e+20]
 [ 8.e+00  1.e+20 -2.e+00  0.e+00  7.e+00]
 [ 6.e+00  1.e+20  1.e+20  6.e+00  0.e+00]]
WorkingTime[naive]: 0.14782 ms
WorkingTime[improved]: 0.13828 ms
WorkingTime[floyd]: 0.06795 ms
[[ 0.e+00  1.e+20  1.e+20  1.e+20  1.e+20]
 [ 0.e+00  0.e+00 -1.e+00  3.e+00 -3.e+00]
 [ 1.e+00  6.e+00  0.e+00  9.e+00  3.e+00]
 [-1.e+00  4.e+00 -2.e+00  0.e+00  1.e+00]
 [ 5.e+00  1.e+01  4.e+00  6.e+00  0.e+00]]

```

### For large dataset 

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
def generate_graph(n, m, randrange:Tuple[int, int], verbose=False):
    """ |V|: n, |E|: m """
    # S = set(' '.join(string.ascii_lowercase).split()[:n])
    S = set(range(n))
    seen = set()
    edges = []
    for _ in range(m):
        while True:
            # start = randomString(length=1, samples=list(S))
            # end = randomString(length=1, samples=list(S - {start}))
            start, end = random.choices(population=range(n - 1), k=2)
            if (start, end) not in seen: 
                seen.add((start, end))
                break
        edges.append((start, end, random.randint(randrange[0], randrange[1])))
    if verbose: visualize_graph(edges, weighted=True)
    graph = defaultdict(list)
    for i in S: graph[i]
    for u, v, w in edges:
        graph[u].append((v, w))
    return graph, edges

def generate_graph_no_neg_cycle(n, m, randrange):
    weights = graph = None
    while True:
        graph, edges = generate_graph(n, m, randrange, verbose=False)
        n2i, W = g2m(graph)
        if not hasNcycles(W): 
            weights = deepcopy(W)
            return n2i, weights, graph, edges
        
n, m = 50, 300
n2i, weights, graph, edges = generate_graph_no_neg_cycle(n, m, randrange=(-10, 100))
print("A graph is generated!")
visualize_graph(edges=edges, weighted=True)
ans1 = naive(weights, verbose=True)
ans2 = improved(weights, verbose=True)
ans3 = floyd(weights, verbose=True)
# plot(ans1), plot(ans2), plot(ans3)
# assert ans1 == ans2 == ans3
plot(ans3)
```

</div>

{:.output_stream}

```
A graph is generated!

```


![png](/assets/images/FloydWarshall_files/FloydWarshall_16_1.png)


{:.output_stream}

```
WorkingTime[naive]: 1775.65432 ms
WorkingTime[improved]: 208.99677 ms
WorkingTime[floyd]: 37.94217 ms
[[0.0e+00 3.8e+01 4.0e+01 ... 5.6e+01 1.9e+01 1.0e+20]
 [4.0e+00 0.0e+00 2.8e+01 ... 4.9e+01 2.3e+01 1.0e+20]
 [4.5e+01 6.2e+01 0.0e+00 ... 8.2e+01 5.7e+01 1.0e+20]
 ...
 [1.9e+01 2.3e+01 5.0e+01 ... 0.0e+00 3.8e+01 1.0e+20]
 [5.0e+00 2.5e+01 2.9e+01 ... 4.5e+01 0.0e+00 1.0e+20]
 [1.0e+20 1.0e+20 1.0e+20 ... 1.0e+20 1.0e+20 0.0e+00]]

```

## Reference

[1] [geeksforgeeks - visualization of a graph](https://www.geeksforgeeks.org/directed-graphs-multigraphs-and-visualization-in-networkx/) <br>
[2] [referenced blog - floyd warshall algorithm](https://www.programiz.com/dsa/floyd-warshall-algorithm )<br>
[3] [geeksforgeeks - detect negative cycles](https://www.geeksforgeeks.org/detect-negative-cycle-graph-bellman-ford/) <br>
[4] [c++ implementation](https://github.com/SUNGWOOKYOO/Algorithm/blob/master/src_Cplus/graphAlgo/FloydWarshall.cpp) <br>
[5] [Scaler Topics - Floyd Warshall Algorithm](https://www.scaler.com/topics/data-structures/floyd-warshall-algorithm/) <br>

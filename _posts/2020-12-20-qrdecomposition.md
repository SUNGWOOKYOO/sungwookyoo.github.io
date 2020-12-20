---
title: "QR Decomposition"
excerpt: "백터공간의 정규 직교 기저를 만드는 방법으로 활용되는 QR 분해를 배워보자. "
categories:
 - study
tags:
 - linear algebra
use_math: true
last_modified_at: "2020-12-20"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: #
 actions:
  - label: "#"
    url: "#"
---

# QR Decomposition

백터공간의 정규 직교 기저를 만드는 방법으로 활용되는 QR 분해를 배워보자.   
QR 분해란 열벡터가 서로 선형독립인 행렬 A를 열벡터가 정규 직교 기저로 이루어진 Q 와 상 삼각행렬인 R의 곱으로 나타내는 것이다.    
열벡터가 선형독립인 A에 그람-슈미트 과정을 적용하여 정규 직교 기저들을 찾는다.  
A의 열벡터들을 $(a_{1}, ...,a_{n})$이라고 하고, 정규 직교 기저들을$q_{1},...,q_{n}$이라고 하자.  
A의 각 열벡터를 위에서 찾는 정규 직교 기저들의 선형 결합으로 나타내면 다음과 같다.  
$$
a_{k} = \sum_{i=1}^{k}(a_{k}^{T}q_{i})q_{i}
$$
  
위의 식을 매트릭스 표현으로 나타내면 다음과 같다.  

$$
\begin{align}
    \begin{bmatrix}
    a_{1} & \cdots & a_{n} 
    \end{bmatrix}
    &= \begin{bmatrix}
    q_{1} & \cdots & q_{n}
    \end{bmatrix}
    \begin{bmatrix}
    a_{1}^{T}q_{1} & a_{2}^{T}q_{1} &\cdots & a_{n}^{T}q_{1} \\
    0 & a_{2}^{T}q_{2} &\cdots & a_{n}^{T}q_{2} \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 &\cdots & a_{n}^{T}q_{n} \\
    \end{bmatrix} \\
A &= QR
\end{align}
$$

Q는 그람 슈미트 과정을 통해 구할 수 있고 $R = Q^{T}A$ 로 구할 수 있다.  
그람 슈미트 과정은 나눗셈과 유사하기 때문에 몫을 의미하는 Q (quotient)와 나머지를 의미하는 R (remainder)로 표기한다.  

### gram schcmidt orthgonalization
$$
\begin{align}
q_{k} &= a_{k} - \sum_{i=0}^{k-1} \mathcal{proj}_{q_{i}}a_{k} &,for\; k = 0,\dots,n-1 \\
&= a_{k} - \sum_{i=0}^{k-1} \frac{<a_{k}, q_{i}>}{<q_{i}, q_{i}>}q_{i}
\end{align}
$$

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)
import pdb
import copy 
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
def gramschmitz(A, n):
    Q = np.zeros_like(A,dtype=float) # int형으로 되면 반올림 오차가 심함
    for k in range(n):
        tmp = 0        
        for i in range(k):             
            tmp += (np.dot(A[:,k], Q[:,i]) / np.dot(Q[:,i], Q[:,i])) * Q[:,i]
        Q[:,k] = A[:,k] - tmp
        Q[:,k] = Q[:,k] / np.sqrt(np.dot(Q[:,k],Q[:,k]))
    return Q

A = np.array([[-3,-1,1],[2,4,-3],[1,1,1]]).T
Q = gramschmitz(A,3)
R = Q.T @ A
print("Q = {}".format(Q))
print("R = {}".format(R))
np.allclose(A, Q@R)
```

</div>

{:.output_stream}

```
Q = [[-0.9  -0.42  0.08]
 [-0.3   0.76  0.57]
 [ 0.3  -0.49  0.82]]
R = [[ 3.32 -3.92 -0.9 ]
 [ 0.    3.69 -0.15]
 [ 0.    0.    1.47]]

```




{:.output_data_text}

```
True
```



## 단점
  
그람 슈미트 과정에 나눗셈이 존재하므로 부동소수점 연산에서 발생하는 오차의 누적으로 Q가 직교성을 잃게 되어서 수치적으로 불안정하다. 또한 A의 열벡터들에도 마찬가지로 부동소수점으로 인한 직교성 손실의 문제가 있다. 이러한 문제를 다루기 위해서 반사를 기반으로 한 하우스홀더 방법과 회전을 기반으로 한 기븐스 회전 방법이 고안되었다.

## 하우스홀더 방법

부동소수점 연산에서도 오차가 누적을 제거할 수있기 때문에 실제로는 그람슈미트 과정보다는 주로 이방법을 통해서 QR분해를 한다.  
그람-슈미트 과정은 Q의 열벡터를 찾고 다시 R의 열벡터를 n step까지 구한다. k번째 스텝에서 Q의 k번째 백터와 R의 K번째 행백터를 구하면 누적에러로 인한 Q의 직교성 손실을 방지 할 수 있다.  
    
하우스 홀더 변환은 임의의 벡터를 한 요소만 그 벡터의 크기의 값으로 존재하며 나머지 요소들이 0 인 벡터로 변환하는 반사행렬이다. 이를 이용하면 그람슈미트 과정없이 A를 상삼각행렬로 만들 수 있다. A 첫번째 열벡터의 첫째항을 제외하고 0을 만드는 반사행렬 $H_{1}$을 찾고 직교행렬 $Q_{1}$를 만들어 A에 곱한다. 이후 다시 $Q_{1}A$의 부분행렬에서 첫째항을 제외하고 0을 만드는 반사행렬 $H_{2}$을 찾고 직교행렬  $Q_{2}$ 을 만들어 $Q_{1}A_{1}$에 곱한다. 이를 상삼각행렬이 될 때 까지 이행하면 A를 QR분해할 수 있다. 분해 유일성에 대한 조건은 A의 열벡터들이 선형독립이면 된다. 즉, A가 열충족계수를 가지면 된다.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def household(A):
    """
    returns Q.T
    """
    n = A.shape[1]
    Q = np.eye(len(A), dtype=float)
    B = copy.deepcopy(A)
    for i in range(n):
        B = (Q@A)[i:,i:]
        v = B[:,0]
        w = np.zeros_like(v, dtype=float)
        w[0] = np.sqrt(np.dot(v,v))
        a = v-w
        a = a.reshape(len(a),1)
        H = np.eye(len(a), dtype=float) - 2.0 / np.dot(a.T, a) * a @ a.T
        K = np.eye(len(A), dtype=float)
        K[i:,i:] = H
        Q = K@Q        
    return Q

Q_t = household(A)
R = Q_t@A
print('R = {}'.format(R))
print('Q = {}'.format(Q_t.T))
np.allclose(A, Q_t.T@R)
```

</div>

{:.output_stream}

```
R = [[ 3.32 -3.92 -0.9 ]
 [ 0.    3.69 -0.15]
 [-0.    0.    1.47]]
Q = [[-0.9  -0.42  0.08]
 [-0.3   0.76  0.57]
 [ 0.3  -0.49  0.82]]

```




{:.output_data_text}

```
True
```



## 기븐스 회전 방법

R의 특정위치에서 성분값을 0 으로 조작할 수 있는 방법이다.

## 연립방정식 풀기
LU분해와 유사하게 QR분해를 사용하면 1차 연립방정식을 효과적으로 풀 수 있다.
$$
\begin{align}
Ax = B \\
QRx = B \\
Rx = Q^{T}B = C
\end{align}  
$$
후진대입을 통해서 해를 구할 수 있다.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
def BackwardSub(U,b):
    x = np.zeros_like(b)
    n = len(b)
    for m in range(n - 1, -1, -1):
        tmp = 0
        for i in range(n - m):
            tmp += U[m][m+i] * x[m + i]
        x[m] = (b[m] - tmp) / U[m][m]
    return x

A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
B = np.array([1, 1, 1, 1])

# gramschmitz
Q = gramschmitz(A,4)
R = Q.T @ A
x = BackwardSub(R, Q.T@B)
print('R = {}'.format(R))
print('Q = {}'.format(Q))
print('x = {}'.format(x))
print(np.allclose(A@x, B))

# household
Q_t = household(A)
R = Q_t@A
x = BackwardSub(R, Q_t@B)
print('R = {}'.format(R))
print('Q = {}'.format(Q_t.T))
print('x = {}'.format(x))
print(np.allclose(A@x, B))
```

</div>

{:.output_stream}

```
R = [[10.15  7.39  8.67 13.4 ]
 [-0.    3.92  6.61  3.56]
 [ 0.    0.    1.07  0.25]
 [ 0.    0.    0.    4.55]]
Q = [[ 0.2   0.9   0.3   0.23]
 [ 0.49 -0.42  0.46  0.61]
 [ 0.69 -0.02  0.17 -0.7 ]
 [ 0.49  0.09 -0.82  0.28]]
x = [ 0.05 -0.08  0.08  0.09]
True
R = [[10.15  7.39  8.67 13.4 ]
 [ 0.    3.92  6.61  3.56]
 [ 0.   -0.    1.07  0.25]
 [-0.   -0.    0.    4.55]]
Q = [[ 0.2   0.9   0.3   0.23]
 [ 0.49 -0.42  0.46  0.61]
 [ 0.69 -0.02  0.17 -0.7 ]
 [ 0.49  0.09 -0.82  0.28]]
x = [ 0.05 -0.08  0.08  0.09]
True

```

# reference
[qr분해 설명 특징](https://ghebook.blogspot.com/2020/07/qr-qr-decomposition.html)

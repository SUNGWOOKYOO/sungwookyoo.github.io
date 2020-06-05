---
title: "이차원배열과 연산"
excerpt: "행 연산과, 열 연산 연습"
categories:
 - algorithms
tags:
 - enumerate
use_math: true
last_modified_at: "2020-06-05"
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
from sys import stdin
import numpy as np
from collections import defaultdict
from copy import deepcopy
```

</div>

# 17140. 이차원 배열과 연산

[Problem Link](https://www.acmicpc.net/problem/17140)

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
stdin = open('data/array2d.txt')
input = stdin.readline
plot = lambda x: print(np.array(x))
r, c, k = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(3)]
```

</div>

## Idea 

주어진 조건에 맞게 구현
정렬에 우선순위는 count(등장횟수) 가 1순위, <br>
number(숫자 크기)가 2순위로 정렬한 후 <br>
number, count ... 로 펼쳐서 새로운 배열을 만든다. 
1. R연산: 행을 기준으로 수행
2. C연산: 열을 기준으로 수행, transpose하고, R연산, transpose하면 똑같다. 

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def op(a):
    m, n = len(a), len(a[0])
    b, maxlen = [], 0
    for i in range(m):
        cnts = defaultdict(int)
        for j in range(n):
            if not a[i][j]: continue
            cnts[a[i][j]] += 1
        vs = sorted(list(cnts.items()), key=lambda e: (e[1], e[0]))
        z = []
        for x, y in vs:
            z.extend([x, y])
        b.append(z)
        maxlen = min(max(maxlen, len(z)), 100)

    for i in range(m):
        if len(b[i]) > maxlen:
            b[i] = b[i][:maxlen]
        elif len(b[i]) < maxlen:
            b[i] = b[i] + [0] * (maxlen - len(b[i]))
    return b

plot(a), print()
plot(op(deepcopy(a)))
```

</div>

{:.output_stream}

```
[[1 2 1]
 [2 1 3]
 [3 3 3]]

[[2 1 1 2 0 0]
 [1 1 2 1 3 1]
 [3 3 0 0 0 0]]

```

### transpose
C연산을 위해 구현

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
def transpose(a):
    m, n = len(a), len(a[0])
    b = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            b[j][i] = a[i][j]
    return b

plot(a), print()
plot(transpose(deepcopy(a)))
```

</div>

{:.output_stream}

```
[[1 2 1]
 [2 1 3]
 [3 3 3]]

[[1 2 3]
 [2 1 3]
 [1 3 3]]

```

### 전체 과정 구현

주의사항:
1. transpose후에 op 연산을 하고, 또 transpose를 해야 원래결과
2. 0은 무시하고, count해야한다.
3. 1 ~ 100 번까지 돌려야하므로 `range(101)`.

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
def solution(a, r, c, k):
    def op(a):
        m, n = len(a), len(a[0])
        b, maxlen = [], 0
        for i in range(m):
            cnts = defaultdict(int)
            for j in range(n):
                if not a[i][j]: continue
                cnts[a[i][j]] += 1
            vs = sorted(list(cnts.items()), key=lambda e: (e[1], e[0]))
            z = []
            for x, y in vs:
                z.extend([x, y])
            b.append(z)
            maxlen = min(max(maxlen, len(z)), 100)

        for i in range(m):
            if len(b[i]) > maxlen:
                b[i] = b[i][:maxlen]
            elif len(b[i]) < maxlen:
                b[i] = b[i] + [0] * (maxlen - len(b[i]))
        return b

    def transpose(a):
        m, n = len(a), len(a[0])
        b = [[0] * m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                b[j][i] = a[i][j]
        return b

    ans = 0
    for _ in range(101):
        if 0 <= r < len(a) and 0 <= c < len(a[0]) and a[r][c] == k: return ans
        a = transpose(op(transpose(a))) if len(a) < len(a[0]) else op(a)
        ans += 1

    return -1

solution(deepcopy(a), r - 1, c - 1, k)
```

</div>




{:.output_data_text}

```
52
```



## Submitted Code

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
# import numpy as np
from collections import defaultdict

stdin = open('data/array2d.txt')
input = stdin.readline
# plot = lambda x: print(np.array(x))
r, c, k = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(3)]


def solution(a, r, c, k):
    def op(a):
        m, n = len(a), len(a[0])
        b, maxlen = [], 0
        for i in range(m):
            cnts = defaultdict(int)
            for j in range(n):
                if not a[i][j]: continue
                cnts[a[i][j]] += 1
            vs = sorted(list(cnts.items()), key=lambda e: (e[1], e[0]))
            z = []
            for x, y in vs:
                z.extend([x, y])
            b.append(z)
            maxlen = min(max(maxlen, len(z)), 100)

        for i in range(m):
            if len(b[i]) > maxlen:
                b[i] = b[i][:maxlen]
            elif len(b[i]) < maxlen:
                b[i] = b[i] + [0] * (maxlen - len(b[i]))
        return b

    def transpose(a):
        m, n = len(a), len(a[0])
        b = [[0] * m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                b[j][i] = a[i][j]
        return b

    ans = 0
    for _ in range(101):
        if 0 <= r < len(a) and 0 <= c < len(a[0]) and a[r][c] == k: return ans
        a = transpose(op(transpose(a))) if len(a) < len(a[0]) else op(a)
        ans += 1

    return -1


print(solution(a, r - 1, c - 1, k))
```

</div>

{:.output_stream}

```
52

```

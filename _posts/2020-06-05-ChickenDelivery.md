---
title: "치킨배달"
excerpt: "조합에 대한 경우의 수 탐색하는 연습"
categories:
 - algorithms
tags:
 - enumerate
 - samsung
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

# 15686. 치킨 배달

Given $n \times n$ a grid, which represents a city's map, <br>
the map contains `0`(empty space), `1`(house), `2`(chicken store). <br>
**Select $m$ chicken stores** to **minimize chicken distance** of this city <br>

# Idea
0. preprocessing단계: 치킨집과 각 집의 위치를 indexing 해놓는다. (distance를 $O(1)$ 에 계산)
1. 모든 치킨집 중 m개의 치킨집을 선택하는 경우의 수 고려. <br>
`itertools`를 사용하여 $ |stores| \choose m $ 경우의 수를 구한다. 
2. 각 경우의 수 마다 도시의 치킨거리(chicken distance of this city)를 구한다. 
3. 모든 경우의 수 조합중 best case를 찾는다. 


<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
from itertools import combinations
from collections import defaultdict
from typing import Tuple
# import numpy as np

stdin = open('data/chicken.txt')
input = stdin.readline
n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]


def solution(a):
    def dist(hids: Tuple[int, int], sids: Tuple[int, int]):
        return abs(hids[0] - sids[0]) + abs(hids[1] - sids[1])

    home, store = [], []
    for i in range(n):
        for j in range(n):
            if a[i][j] == 1:
                home.append((i, j))
            elif a[i][j] == 2:
                store.append((i, j))

    ans = 1e20  # chicken distance of a city
    for comb in combinations(range(len(store)), m):
        chickdist = defaultdict(lambda: 1e20)
        for k in comb:
            for e in home:
                chickdist[e] = min(chickdist[e], dist(e, store[k]))
        ans = min(ans, sum(chickdist.values()))
    return ans

print(solution(a))
```

</div>

{:.output_stream}

```
10

```

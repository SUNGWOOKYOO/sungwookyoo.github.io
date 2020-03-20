---
title: "1767. 프로세서 연결하기 "
excerpt: "삼성SW 역량테스트 문제. 가능한 프로세서를 모두 연결시키되 필요한 최소 전선 수를 찾아라"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
use_math: true
last_modified_at: "2020-03-20"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "SW 역량테스트"
    url: "https://swexpertacademy.com/main/talk/solvingTalk/boardCommuList.do?searchCondition=COMMU_DETAIL-COMMU_TITLE-NICK_NAME_TAG&commuId=AWD_S-56BPoDFAWR&searchKeyword=%ED%94%84%EB%A1%9C%EC%84%B8%EC%84%9C&orderBy=DATE_DESC&pageSize=20&pageIndex=1"
  - label: "C++ 구현 블로그"
    url: "https://2youngjae.tistory.com/117"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
import numpy as np
import random, math
import logging, argparse, yaml, copy
import matplotlib.pyplot as plt
sys.path.append('/home/swyoo/algorithm')
from utils.verbose import logging_time, printProgressBar
```

</div>

# 1767. 프로세서 연결하기

[삼성 SW link](https://swexpertacademy.com/main/talk/solvingTalk/boardCommuList.do?searchCondition=COMMU_DETAIL-COMMU_TITLE-NICK_NAME_TAG&commuId=AWD_S-56BPoDFAWR&searchKeyword=%ED%94%84%EB%A1%9C%EC%84%B8%EC%84%9C&orderBy=DATE_DESC&pageSize=20&pageIndex=1)

[c++ 로 구현된 좋은 코드](https://2youngjae.tistory.com/117)를 찾아서 이를 바탕으로 python으로 구현하였다.

Notation은 $n \times n$ array 가 주어졌을때, <br>
이 안에 들어있는 processor의 갯수를 $M$ 이라 하자. <br>

각 processor 마다 동,서,남,북,행동X  5가지 cases을 모두 고려해 전선을 까는 경우에 대해 모든 프로세서를 탐색 해보고, <br>
프로세서를 가장 많이 가동 할 수 있으면서 전선이 최소로 필요한 상황에서 최소 전선의 수를 찾는 것이 목표이다.

최악의 경우 모든 cases를 enumerate하는 경우 $O(n5^M)$ 시간이 걸린다. <br>
* 각 프로세서 마다 최대 5번의 방향성을 고려해야 함.
* `inline, drawline` 함수는 $O(n)$ 시간이 걸림.
> 가정상황: $n^2 < 5^M$

하지만, 밑의 방식으로 코딩하면 drawline을 통해 주변 프로세서들이 가능한 방향이 pruning되기 때문에 <br>
최악의 경우와 같이 모두 조사하는 상황은 거의 발생하지 않는다. 

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
pnum = 0  # 찾아야할 파워가 연결된 프로세서 갯수
ans = 1e8  # 찾아야할 사용된 전선들의 최소길이
n = 0
a = []

def isline(r, c, dir):
    global a, n
    if dir == 0:
        for j in range(c + 1, n):
            if a[r][j] != 0:
                return False
    elif dir == 1:
        for j in range(c - 1, -1, -1):
            if a[r][j] != 0:
                return False
    elif dir == 2:
        for i in range(r + 1, n):
            if a[i][c] != 0:
                return False
    elif dir == 3:
        for i in range(r - 1, -1, -1):
            if a[i][c] != 0:
                return False
    return True

def drawline(r, c, dir, fill):
    """ fill=2 means draw
        fill=0 means delete. """
    global a, n
    line = 0
    if dir == 0:
        for j in range(c + 1, n):
            a[r][j] = fill
            line += 1
    elif dir == 1:
        for j in range(c - 1, -1, -1):
            a[r][j] = fill
            line += 1
    elif dir == 2:
        for i in range(r + 1, n):
            a[i][c] = fill
            line += 1
    elif dir == 3:
        for i in range(r - 1, -1, -1):
            a[i][c] = fill
            line += 1
    return line

@logging_time
def solve():
    global a
    def dfs(p, pidx, nump, line):
        """
        p 는 프로세서 인덱싱 정보, pidx는 조사할 프로세서 인덱스,
        nump는 power가 연결된 프로세서 최대 수, line은 조사중인 전선 길이
        """
        global a, pnum, ans, n
        if pidx == len(p):  # 모든 프로세서에 대해 조사 끝났을 경우
            if pnum < nump: # 더 많은 프로세서를 가동할 경우를 찾을 경우
                ans = line
                pnum = nump
            elif pnum == nump: # 프로세서는 이전과 일치하지만, 
                if ans > line: # 더 적은 전선이 필요한 상황을 찾을 경우
                    ans = line
            return
        
        for i in range(4):  # 동, 서, 남, 북 에 대해 조사
            if isline(*p[pidx], i):  # 현재 프로세서를 기준으로 i 방향으로 전선을 까는 것이 가능한지 조사
                # 전선을 drawline을 통해 깔고, 현재 line수를 업데이트한 상태로 다음 process에 대해 조사
                dfs(p, pidx + 1, nump + 1, line + drawline(*p[pidx], dir=i, fill=2))
                drawline(*p[pidx], dir=i, fill=0) # roll back
        dfs(p, pidx + 1, nump, line) # no drawing case

    # dfs 시작지점 지정을 위해 연결되지 않은 프로세서 인덱싱(가장자리는 이미 연결됨)
    p = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if a[i][j] == 1:
                p.append((i, j))
    # 가장 자리를 제외하고, 전선을 연결해가며 프로세서 연결이 최대가 될때의 전선 길이 반환
    dfs(p, pidx=0, nump=0, line=0)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
if __name__ == "__main__":
    sys.stdin = open('data/processor.txt')
    T = int(sys.stdin.readline())

    for tc in range(1, T+1):
        n = int(sys.stdin.readline())
        a = []
        pnum = 0
        ans = 1e8
        for _ in range(n):
            a.append(list(map(int, sys.stdin.readline().split())))
        solve(verbose=True)
        print("#{} {}".format(tc, ans))
```

</div>

{:.output_stream}

```
WorkingTime[solve]: 0.94771 ms
#1 12
WorkingTime[solve]: 8.89897 ms
#2 10
WorkingTime[solve]: 56.05412 ms
#3 24

```

---
title: "스타트와 링크"
excerpt: "팀 나누기, DFS, combination 연습"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
 - samsung
use_math: true
last_modified_at: "2020-06-02"
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
```

</div>

# 14889. 스타트와 링크

Assume that $n$ be a even number. <br>
스타트 팀과 링크 팀으로 나누는데 스코어의 차이를 최소화하여 나누고 싶다. 

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
plot = lambda a: print(np.array(a))
stdin = open('data/startandlink.txt')
input = stdin.readline
n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]
a
```

</div>




{:.output_data_text}

```
[[0, 1, 2, 3], [4, 0, 5, 6], [7, 1, 0, 2], [3, 4, 5, 0]]
```



### Step1. DFS

DFS를 통해 member를 정한다. <br>
일단, 한쪽 맴버를 모두 정하는 경우의 수는 다음과 같다. <br>
대칭적인구조를 띄고있다. <br>
분석해보면 다음과 같다. <br>
${n \choose n/2} = O(n^{n/2}) = O(n^n)$ 가지의 경우의 수를 고려해야하는데, <br>
자세히 보면, `[0, 1]`을 구하면 자동으로 `[2, 3]`을 마지막 `i = n //2`에서 구하기 때문에 <br>
대칭적인 구조를 띄고있다.

이 중복되는 연산을 없애면, $\frac{1}{2}{n \choose n/2}$ 의 경우의 수를 고려하면 된다. <br>
하지만, 수학적으로 시간복잡도를 개선시키지는 못한다. <br>
이 중복되는 연산을 없애는 법? 은 잘 생각이 안나 그대로 사용하였다. <br>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def dfs(i, members):
    if len(members) == n // 2:
        print(members)
        return

    for j in range(i + 1, n):
        dfs(j, members + [j])

for i in range(n // 2 + 1):
    dfs(i=i, members=[i])
```

</div>

{:.output_stream}

```
[0, 1]
[0, 2]
[0, 3]
[1, 2]
[1, 3]
[2, 3]

```

### Discuss

[rebas's blog](https://rebas.kr/754) 로 부터 안 사실인데, <br>
Team A를 True, Team B를 False로 두어 dfs를 하면 <br>
중복되는 부분을 피할 수 있다.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
c = [False]*n
def solve(cnt, idx):
    global ans
    if idx == n:
        return
    if cnt == n//2:
        print(c)
        return
    c[idx] = True
    solve(cnt+1, idx+1)
    c[idx] = False
    solve(cnt, idx+1)
solve(0, 0)
```

</div>

{:.output_stream}

```
[True, True, False, False]
[True, False, True, False]
[False, True, True, False]

```

### Step2. 스코어 계산
팀을 나누고, 팀에 대응되는 스코어를 구하여 두 팀과이 차이로 `ans`를 업데이트. <br>
`[0, 3]` vs `[1, 2]`로 나누었을 때, 스코어 차이는 `6 - 6 = 0`. 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
def solution(a, n, show=False):
    n, ans = len(a), 1e20
    
    def dfs(i, members):
        nonlocal ans
        if len(members) == n // 2:
            if show: print(members, end='| ')
            team = set(members)
            s1 = s2 = 0
            for p in range(n):
                for r in range(p + 1, n):
                    if p in team and r in team:
                        s1 += (a[p][r] + a[r][p])
                    if p not in team and r not in team:
                        s2 += (a[p][r] + a[r][p])
            if show: print("{} vs {}".format(s1, s2))
            ans = min(ans, abs(s1 - s2))
            return

        for j in range(i + 1, n):
            dfs(j, members + [j])

    for i in range(n // 2 + 1):
        dfs(i=i, members=[i])

    return ans

solution(a, n, show=True)
```

</div>

{:.output_stream}

```
[0, 1]| 5 vs 7
[0, 2]| 9 vs 10
[0, 3]| 6 vs 6
[1, 2]| 6 vs 6
[1, 3]| 10 vs 9
[2, 3]| 7 vs 5

```




{:.output_data_text}

```
0
```



### Improved

**Discuss** Part로 부터 중복되는 연산을 피하도록 하는 방법을 사용하여 코딩하면 다음과 같다.

`True`가 `n // 2` 번 나오면 recursion을 멈춘다.

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
ans = 1e20
team = [False] * n
def solve(cnt, idx):
    global ans
    if idx == n:
        return
    if cnt == n//2:
        s1 = s2 = 0
        for i in range(n):
            for j in range(i + 1, n):
                if team[i] and team[j]:
                    s1 += (a[i][j] + a[j][i])
                if not team[i] and not team[j]:
                    s2 += (a[i][j] + a[j][i])
        ans = min(ans, abs(s1 - s2))
        return
    team[idx] = True
    solve(cnt+1, idx+1)
    team[idx] = False
    solve(cnt, idx+1)
    return ans

solve(cnt=0, idx=0)
```

</div>




{:.output_data_text}

```
0
```



### Time Complexity
1. 경우의 수: $O(n^n)$
2. 팀의 스코어 구하는 시간: $O(n^2)$

$$
O(n^{n+2})
$$

## Submitted Code

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
# import numpy as np

# plot = lambda a: print(np.array(a))
stdin = open('data/startandlink.txt')  # 제출시 주석처리
input = stdin.readline
n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]

ans = 1e20
def dfs(i, members):
    global ans
    if len(members) == n // 2:
        # print(members, end='| ')
        res1 = 0
        for p in range(len(members)):
            for r in range(p + 1, len(members)):
                res1 += (a[members[p]][members[r]] + a[members[r]][members[p]])
        # print(res1, end=' vs ')
        complementary = list(set(range(n)) - set(members))
        res2 = 0
        for p in range(len(complementary)):
            for r in range(p + 1, len(complementary)):
                res2 += (a[complementary[p]][complementary[r]] + a[complementary[r]][complementary[p]])
        ans = min(ans, abs(res1 - res2))
        return

    for j in range(i + 1, n):
        dfs(j, members + [j])

for i in range(n // 2 + 1):
    dfs(i=i, members=[i])

print(ans)
```

</div>

{:.output_stream}

```
0

```

#### Improved

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
stdin = open('data/startandlink.txt')  # 제출시 주석처리
input = stdin.readline
n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]

ans = 1e20
team = [False] * n
def solve(cnt, idx):
    global ans
    if idx == n:
        return
    if cnt == n//2:
        s1 = s2 = 0
        for i in range(n):
            for j in range(i + 1, n):
                if team[i] and team[j]:
                    s1 += (a[i][j] + a[j][i])
                if not team[i] and not team[j]:
                    s2 += (a[i][j] + a[j][i])
        ans = min(ans, abs(s1 - s2))
        return
    team[idx] = True
    solve(cnt+1, idx+1)
    team[idx] = False
    solve(cnt, idx+1)
    return ans

solve(cnt=0, idx=0)

print(ans)
```

</div>

{:.output_stream}

```
0

```

### Discuss

`itertools.combinations` 를 이용할 수도 있다.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
from itertools import combinations
combi = list(combinations(range(n), n//2))
ans = 1e20

def teamScore(team: tuple):
    score = 0
    for i in team:
        for j in team:
            score += a[i][j]
    return score

for idx in range(len(combi)//2):
    teamA = combi[idx]
    teamB = combi[len(combi) - idx - 1]
    ans = min(ans, abs(teamScore(teamA) - teamScore(teamB)))

ans
```

</div>




{:.output_data_text}

```
0
```



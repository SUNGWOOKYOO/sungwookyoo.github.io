---
title: "721.Account and Merge"
excerpt: "merge emails by finding same account from given input."
categories:
 - algorithms
tags:
 - DFS
use_math: true
last_modified_at: "2020-03-22"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "Leetcode"
    url: "https://leetcode.com/problems/accounts-merge/"
  - label: "Good code"
    url: "https://leetcode.com/problems/accounts-merge/discuss/109161/Python-Simple-DFS-with-explanation!!!"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
from pprint import pprint
from termcolor import colored
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time, printProgressBar
from collections import defaultdict
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# toy example
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"],
            ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
pprint(accounts)
```

</div>

{:.output_stream}

```
[['John', 'johnsmith@mail.com', 'john00@mail.com'],
 ['John', 'johnnybravo@mail.com'],
 ['John', 'johnsmith@mail.com', 'john_newyork@mail.com'],
 ['Mary', 'mary@mail.com']]

```

# 721. Accounts Merge

## Objective
merge all emails if some emails from same account exist.
> accounts 에는 각각의 user와 user가 가지고 있는 email 주소 리스트가 주어진다. <br>
이때, 만약 user A와 user B가 동일한 유저라면(가지고 있는 email list중 하나라도 똑같은게 있다면) emails list들을 merge 할 수 있다.
모든 accounts에 주어진 emails들을 merge 하고, 사전순으로 sort하자.

일단 가장 쉽게 풀이하면, account마다 email이 겹치는지 모든 case를 비교해보고 겹치는 것이 있다면 merge해서 sorting하면된다. 
하지만, 이 과정에서 redundancy가 매우 많이 발생한다. 

## Key Idea

### Create Graph 
If the algorithm knows that the email is related to some people, a mergeable set is determined. <br>
Constructing the information `who` takes $O(\sum_{i=0}^{n-1}{a_i})$, where $a_i$ means the number of emails from a person `i`.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
who = defaultdict(list) # {email: person index}
for i in range(len(accounts)):
    name = accounts[i][0]
    for j in range(1, len(accounts[i])):
        who[accounts[i][j]].append(i)

# visualize 
print("{:<34} {:}".format(colored("Email", "red"), colored("Person Indices", "red")))
for k, v in who.items():
    print("{:<25} {}".format(k, v))
```

</div>

{:.output_stream}

```
[31mEmail[0m                     [31mPerson Indices[0m
johnsmith@mail.com        [0, 2]
john00@mail.com           [0]
johnnybravo@mail.com      [1]
john_newyork@mail.com     [2]
mary@mail.com             [3]

```

we can easily notice that **emails from person 0 and 2 can be merged** because `johnsmith@mail.com` has `[0, 2]`

How can use this information? see next section: Use DFS

### Use DFS 

DFS를 어떻게 구현할지 생각하는 과정은 다음과 같다. <br>

We want to implement `dfs` to do the following:
* After finishing dfs(i), emails related to person i are merged.
  > dfs(i)가 끝나는 시점에서 person `i` 와 관련된 모든 이메일들이 merge되도록 하고 싶다.
  
`accounts[i]`에는 person i의 email들이 있다. <br>
그런데, 우리는 **사전에 email에 해당되는 people 정보들을 모아놨다**. <br>
따라서, `accounts[i][1:]` 를 explore하는데, `accounts[i][j]` (한 이메일) 마다 `who` dictionary를 보고, <br> 
이 에 해당하는 사람에 대한 이메일들을 recursive 하게 업데이트 하도록 하자 <br>
(똑같은 이메일을 merge하지 않도록 `set`을 사용한다). <br>

* Search all emails related to person i only once.
  > 모든 이메일 리스트를 한번씩만 보고 싶다. (똑같은 것을 보면 redundancy가 발생하니까)
  
**이미 방문한 사람의 정보는 또 보지 않도록** `seen`을 두어 marking한다면 <br>
이메일을 한번씩만 보게 되어 redundancy를 없앨 수 있다. <br>


복잡하게 설명하였는데, **마지막 정리**를 하면 다음과 같다. <br>
`dfs(i)` 가 끝나는 시점에서는 <span style="color:red">**person `i`와 관련된 모든 emails들**이 merge가 되며, <br>
**방문한 person index들에 대해서**는 seen flag가 `True`가 된다.</span> (더 이상 또 볼 필요가 없어진다.)

구현하면 다음과 같다. 

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
seen = [False] * len(accounts)  # mark whether visited before for a person.
def dfs(i, merged: set):
    """ merge all emails related to person i by exploration.
    when fishing time, all emails related to person i are merged. """
    seen[i] = True
    name = accounts[i][0]
    for j in range(1, len(accounts[i])):
        for k in who[accounts[i][j]]:
            if seen[k] != True:
                dfs(k, merged)
            merged.add(accounts[i][j])
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# set is mutable, so after dfs, this will be updated.
emails = set()
dfs(i=0, merged=emails)
# after sorted, set will be transformed into sorted list.
print(seen)
print(emails)
```

</div>

{:.output_stream}

```
[True, False, True, False]
{'john_newyork@mail.com', 'john00@mail.com', 'johnsmith@mail.com'}

```

Explanation: 밑의 정보를 보고 생각하면 쉽다. <br>
person `0` 에 대한 이메일들은 `johnsmith@mail.com`, `john00@mail.com` 이다. <br>
그런데 `who` 에서 보이듯 `johnsmith@mail.com` 과 연관된 people은 `[0, 2]` 이므로, <br>
`dfs(2, merged)`를 call 하여 person `2` 에 있는 이메일들 정보를 업데이트 해놓고 <br>
dfs(2, merged)가 finish 하면 다시 dfs(1, merged)로 돌아와 모든 정보들을 업에이트한다. <br>
그렇게 진행하면 person `0`과 관련된 모든 이메일들이 merge가 되면서 <br>
person `0`과 똑같은 유저들 (여기서는 person `2`)에 해당하는 `seen[2]` 가 `True` 가 된다. <br>
따라서, 위에서 말한 **마지막 정리** 의 내용이 참이 된다.
>`dfs(0, merged=set())`가 끝날때 person `i`와 관련된 모든 이메일들이 merge가 되어 merged에 업데이트가 되며, <br>
방문한 person index들에 대해서는 seen flag가 True가 된다. (더 이상 또 볼 필요가 없어진다.)

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
pprint(who)
pprint(accounts)
```

</div>

{:.output_stream}

```
defaultdict(<class 'list'>,
            {'john00@mail.com': [0],
             'john_newyork@mail.com': [2],
             'johnnybravo@mail.com': [1],
             'johnsmith@mail.com': [0, 2],
             'mary@mail.com': [3]})
[['John', 'johnsmith@mail.com', 'john00@mail.com'],
 ['John', 'johnnybravo@mail.com'],
 ['John', 'johnsmith@mail.com', 'john_newyork@mail.com'],
 ['Mary', 'mary@mail.com']]

```

## Final implementation

**마지막 정리**에 따르면
> `dfs(i)` 가 끝나는 시점에서는 <span style="color:red">**person `i`와 관련된 모든 emails들**이 merge가 되며, <br>
**방문한 person index들에 대해서**는 seen flag가 `True`가 된다.</span> (더 이상 또 볼 필요가 없어진다.)

**이 말은 즉슨, 최소한 person i에 대한 정보는 merge되어서 더 이상 볼필요는 없다.** <br>

따라서, 모든 acount에 대해 merge를 수행하기 위해서는 <br>
**모든 `i` 에 대해 dfs(i)를 하되 seen[k] 가 `True` 라면 굳이 dfs(k)를 또할 필요는 없다.**

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
def solve(accounts):
    """
    :type accounts: List[List[str]]
    :rtype: List[List[str]]
    """
    # Create a graph to get information to check if merging is possible.
    who = defaultdict(list) # {email: person index}
    for i in range(len(accounts)):
        name = accounts[i][0]
        for j in range(1, len(accounts[i])):
            who[accounts[i][j]].append(i)
    
    seen = [False] * len(accounts)  # mark whether visited before for a person.
    def dfs(i, merged: set):
        """ merge all emails related to person i by exploration.
        when fishing time, all emails related to person i are merged. """
        seen[i] = True
        name = accounts[i][0]
        for j in range(1, len(accounts[i])):
            for k in who[accounts[i][j]]:
                if seen[k] != True:
                    dfs(k, merged)
                merged.add(accounts[i][j])
    ans = []
    for i in range(len(accounts)):
        if seen[i] != True:
            emails = set() # set is mutable, so after dfs, this will be updated.
            dfs(i, merged=emails)
            name = accounts[i][0]
            ans.append([name] + sorted(emails)) # after sorted, set will be transformed into sorted list.
    return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
print(solve(accounts))
```

</div>

{:.output_stream}

```
[['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'], ['John', 'johnnybravo@mail.com'], ['Mary', 'mary@mail.com']]

```

## Time Complexity Analysis
`accounts` 는 $n$개의 list를 갖고, 각 list `accounts[i]` 의 원소 갯수를 $a_i$ 라고 하자. <br>
accounts 모든 원소를 적어도 한번씩 보게 되고, accounts[i]마다 dfs(i, merged)가 끝났다면 sort를 해야한다.<br>
따라서, 총 걸리는 시간은 다음과 같이 계산된다.
$$
O(\sum_{i=0}^{n-1}{a_i log a_i})
$$

# Report

문제를 풀 당시에 Graph 를 만들어 놓고, 같은 email를 사용하는 사람들이 있다면, <br>
그 사람들에 해당하는 이메일만을 merge하면 되겠다는 아이디어 까지는 생각해내는데 성공하였다. <br>
(`who` 를 만들어 놓는다)

하지만, 아직도 dfs를 통해 email 들을 모두 보되, 전에 만든 `who`를 이용하여 <br>
recursive search 를 하도록 하는 구현을 하는데는 성공하지 못했다. <br>
비슷한 유형의 문제를 많이 풀어보도록하자. 
이 문제와 상당히 유사한 문제는 [851.Loud and Rich](https://sungwookyoo.github.io/algorithms/LoudRich/#use-caching) 문제이다.

Tips
* set을 이용하면 duplicate가 없도록 update를 할 수 있으며, `mutable` 이기 때문에 recursive 함수에 사용하기 용이하다.
* dfs를 구현할때, 원하는 결과를 얻기위해 어느 곳을 search하도록 해야하나가 중요한 것같다. 여기서 핵심이 되는 부분은 다음과 같다. <br>
(`who` 를 사전에 만들어놨다는 가정하에)
```python
for j in range(1, len(accounts[i])):
    for k in who[accounts[i][j]]:
        if seen[k] != True:
            dfs(k, merged)
        merged.add(accounts[i][j])
```

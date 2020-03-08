---
title: "948.Bag Of Tokens"
excerpt: "solve this problem using python and c++"
categories:
 - algorithms
tags:
 - DP
 - Greedy
use_math: true
last_modified_at: "2020-03-06"
toc: true
toc_ads: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: think and think! 
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/bag-of-tokens/"
---

### BFS
Search all cases, and take a max answer.
<div style="background-color:gray"> too slow, so time limited </div>
<details> <summary> Comment </summary>
<p>BFS search 방법을 사용하여 모든 경우의 수를 따진다는 것은, <br> 
    모든 token을 방문하는데 한 token을 방문 할 경우는 2가지 case 로 나뉘기 때문에 <br>
    Queue 에 seen dictionary를 <font color=red>각각 두어야</font> 올바른 계산이 된다. 따라서, 상당히 복잡해지고, 느려진다.
    </p>
</details>

### Greedy

Sort tokens.  
Buy at the cheapest and sell at the most expensive.

> power를 최대한적게 사용하여 많은 token을 사되, score를 소비하여 power를 얻을때는 최대한 많은 power를 얻어야한다. 



`python` implementation

```python
# Helper
import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1e3
        print("WorkingTime[{}]: {:.5f} ms".format(original_fn.__name__, elapsed_time))
        return result
    return wrapper_fn

def show(msg, i,j,k,m):
    print(msg, "|pos={} |power={} |score={} |seen_positions={}".format(i,j,k,m))
```

```python
from collections import defaultdict, deque

class Solution(object):
    def bagOfTokensScore(self, tokens, P):
        """
        :type tokens: List[int]
        :type P: int
        :rtype: int
        """
        @logging_time
        def BFS_agent():
            def BFS(start, tokens_init, P_init):
                """ BFS search for starting at tokens[start] """
                tokens = tokens_init[:]
                loc = ans = 0
                if P_init >= tokens[start]: # if possible, update inital power and score 
                    P_init -= tokens[start]
                    loc += 1
                    # print("-[case2]: lose power, gain a score")

                Q = deque([(start, P_init, loc, set([start]))])
                # print("inital Q: {}".format(Q))
                Level = deque([0])
                while Q:
                    i, power, loc, seen = Q.popleft()
                    # show("popleft:", i, power, loc, seen)
                    if loc > ans:
                        ans = loc
                        # print("{}\nupdate optimal case|| current ans={}|debug_seen={}\n{}".format('='*50, ans, seen,'='*50))
                    for j, tok in enumerate(tokens):
                        # print("\t> explore tokens[{}]={}|seen={}".format(j, tok, seen))
                        if j not in seen:
                            # print("\t\t> seen_positions={}| not {} in seen".format(seen, j))
                            # print("\t\t> current power={}| score={}".format(power, loc))
                            if power >= tok: # lose power, gain a score
                                new_power = power - tok
                                new_loc = loc + 1
                                Q.append([j, new_power, new_loc, set(list(seen) + [j])])
                                # print("\t\t\t+[case1]|lose power, gain a score |becomes (power={}, score={})".format(new_power, new_loc))
                            if loc > 0: # gain power, lose a score
                                new_power = power + tok
                                new_loc = loc - 1
                                Q.append([j, new_power, new_loc, set(list(seen) + [j])])
                                # print("\t\t\t-[case2]|gain power, lose a score |becomes (power={}, score={})".format(new_power, new_loc))
                return ans
            
            ans = 0
            for i in range(len(tokens)):
                ans = max(ans, BFS(i, tokens, P))
            return ans
        
        @logging_time
        def greedy(tokens_init, P_init):
            tokens, P = tokens_init[:], P_init
            tokens.sort()
            ans = loc = 0 # define optimal and local score.
            dQ = deque(tokens)
            while dQ and (P >= dQ[0] or loc): 
            # buy token and lose power in cheap order.
                if P >= dQ[0]: # you should buy the cheapest one as you can in order to get a score  whenever you have a chance.
                    P -= dQ.popleft() 
                    loc += 1
                    # print("lose power, gain a score|", P, loc)
                else:
                    P += dQ.pop() # if you are short on power, sell most expensive one.
                    loc -= 1
                    # print("gain power, lose a score|", P, loc)
                ans = max(ans, loc)
            return ans
        
        sol1 = BFS_agent()
        sol2 = greedy(tokens, P)
        
        assert sol1 == sol2
        print("so11={}| sol2={}".format(sol1, sol2))
        
sol = Solution()
```



`C ++ `implementation

```python
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm> // sort, max
#include <cstdio> // freopen, stdin
#include <chrono>
#include <iomanip>
using namespace std;

class Solution {
public:
	int bagOfTokensScore(vector<int> &tokens, int P) {
		int sol1 = 0;
		auto start = chrono::system_clock::now();
		sol1 = greedy(tokens, P);
		auto elapsed = chrono::system_clock::now() - start;
		auto nsec = chrono::duration_cast<chrono::nanoseconds>(elapsed);
		cout << setw(15) << "greedy elapsed: " << nsec.count() << "ns" << endl;

		return sol1;
	}

	int greedy(vector<int> tokens, int P) {
		int ans = 0, loc = 0;
		sort(tokens.begin(), tokens.end());
		deque<int> dQ; // push back, and pop front.
		for (auto e : tokens) {
			dQ.push_back(e);
		}

		while ((!dQ.empty()) && (P >= dQ.front() || loc != 0)) {
			if (P >= dQ.front()) {
				P -= dQ.front();
				dQ.pop_front();
				loc += 1;
			} else {
				P += dQ.back();
				dQ.pop_back();
				loc -= 1;
			}
			ans = max(ans, loc);
		}
		return ans;
	}
};

int main(int argc, char **argv) {
	freopen("input.txt", "r", stdin);
	int T, P, n, e, ans = 0;
	cin >> T;
	for (int tc = 1; tc <= T; ++tc) {
		cin >> P >> n;
		vector<int> tokens;
		for (int i = 0; i < n; ++i) {
			cin >> e;
			tokens.push_back(e);
		}

		Solution sol;
		ans = sol.bagOfTokensScore(tokens, P);
		cout << "#" << tc << " " << ans << endl;
	}
	return 0;
}
```


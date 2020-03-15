---
title: "Random Variables"
excerpt: "여러가지 랜덤변수들에 대한 의미와 분포, 각 변수들끼리의 관계에 기반한 평균, 분산의 의미들을 분석해보겠다."
categories:
 - study
tags:
 - random variables
use_math: true
last_modified_at: "2020-03-15"
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

# 랜덤변수들



## Bernoulli

### 변수의 의미
어떤 사건이 일어났을 때 동시에 다른 사건도 일어나는 겨우를 말하며 예를 들면 성공이 있으면 반대로 실패도 있는 경우를 말한다. 베르누이 랜덤변수는 성공했을 때 1이고 실패했을 때는 0 인 indicator function이다. parameter로는 어떤 사건이 성공할 확률 p를 말한다.  

### PMF와 CDF
$$
P_{X}(k) = p^{k}(1-p)^{1-k}, k = 0,1, ... \\
F_{X}(x) = \begin{cases}
			0	& ,x<0 \\
			1-p	& ,0 \leq x < 1 \\
			p	& ,x \geq 1 \\
		 \end{cases}
$$
### 평균과 분산
 평균은 당연히 p확률로 일어나므로 p이고 분산은 정의에 의해 구하면 아래와 같다.
$$
\mu_{X} = p \\
\sigma_{X}^{2} = p(1-p)
$$



## Binomial

### 변수의 의미
 n번의 독립 베르누이 실행  중 성공한 횟수이다. 파라미터로 시행횟수 n 과 성공할 확률 p를 갖는다. n=1 일때는 베르누이 와 같다. 그래서 n개의 Bernoulli 변수 의 합은 binomial 변수 이다.
$$
Y\text{ ~ Binomial R.V with }(n,p) \\
X_{i} \text{ ~ Bernoulli R.V with } (p) = \text{Binomial R.V with } (1,p) \\
Y = \sum_{i=1}^{n}X_{i}
$$

### PMF와 CDF

 CDF는 우측방향에서 만 연속이 된다.
$$
P_{X}(k) = \sum_{k=0}^{n} {n \choose k} p^{k}(1-p)^{n-k} \text{ for } k = 0,1, ... \\ 
F_{X}(x) = \sum_{k=0}^{n} {n \choose k} p^{k}(1-p)^{n-k} \text{ for } n \leq x < n+1 \\
$$

### 평균과 분산

Binomial은 Bernoulli의 n개 합이므로 아래와 같은 값을 갖는다. 직관적으로 n번 시행했을 때 평균은 np가 된다.
$$
\mu_{X} = np \\
\sigma_{X}^{2} = np(1-p)
$$



## Geometric

### 변수의 의미

Bernoulli 시행의 sequence에서 첫번재 성공까지의 시행횟수이며 파라미터로 p 를 갖는다.

중요한 특징으로는  Memoryless property를 갖으며 Exponential distribution도 동일한 특성을 지닌다.

아래서 보면 알겠지만 Exponetial r.v는 연속적인 변수에서 Geometric r.v와 비슷한 것을 알 수있다. 

시행이 무수히 많아지고 (n이 무한으로) 성공확률이 희박해질때 (p가 0으로) 로 근사화시켜서 생각해보면 된다.

Memoryless Property는 아래와 같다.
$$
P(X > i + j | X > i) = P(X > j) \text{ for } i,j \geq 1
$$
i시간 까지 계속 일어나던 사건이 j시간후 에도 일어날 확률은 j 시간 후에 사건이 일어날 확률과같다. 

즉 i시간 까지 일어난 사건은 j시간 후에 그 사건이 일어나는데 아무런 관련이 없다는 말과 같다.

### PMF와 CDF

 0 번 째 부터 사건이 성공할 수 없으므로 k =1 부터 시작이다. 

k-1 번째까지는 계속 실패하다가 k번째에 성공할 확률로 구할 수 있다.
$$
P_{X}(k) = (1-p)^{k-1}p \text{ for } k = 1,2, ... \\
F_{X}(x) = 1 - (1 - p)^{x} \text{ for } k \leq x < k+1
$$

### 평균과 분산

Bernoulli 시행의 성공확률이 높을 수록 빨리 끝나게 되고 낮을 수록 늦게 끝나니까 평균은 Bernoulli와 반비례관계를 갖게 된다.
$$
\mu_{X} = \frac{1}{p} \\
\sigma_{X}^{2} = \frac{(1-p)}{p^2}
$$



## Negative Binomial

 

### 변수의 의미

 Bernoulli 시행의 sequence에서 k번재 성공까지의 시행횟수이며 파라미터로 k와 p 를 갖는다.

k=1 일때는 Geometric R.V 와 같다. 그래서 n개의 Bernoulli 변수 의 합은 binomial 변수 이다.
$$
Y\text{ ~ Negative Binomial R.V with }(k,p) \\
X_{i} \text{ ~ Geometric R.V with } (p) = \text{Negative Binomial R.V with } (1,p) \\
Y = \sum_{i=1}^{n}X_{i}
$$


### PMF와 CDF

최소시행 횟수는 k가 된다.

x-1 번째 시행 
$$
P_{X}(x) = {x-1 \choose k-1} p^{k}(1-p)^{x-k} \text{ for } x = k,k+1, ... \\ 
F_{X}(x) = \sum_{i=0}^{n} {x-1 \choose k-1} p^{k}(1-p)^{i-k} \text{ for } k \leq x < k+1 \\
$$


###  평균과 분산

Negative Binomial R.V은 Geometric R.V의 n개 합이므로 아래와 같은 값을 갖는다. 직관적으로 n번 시행했을 때 평균은 k/p가 된다.
$$
\mu_{X} = \frac{K}{p} \\
\sigma_{X}^{2} = \frac{K(1-p)}{p^2}
$$


## Poisson 

### 변수의 의미

 평균적으로 일정시간동안 $$\lambda$$ 번 일어나는 사건이 일정시간동안 일어나는 실제 횟수를 의미하며 파라미터로 $$\lambda> 0$$를 갖는다. Binomial R.V의 Continuous Version이라고 생각하면 된다. 그러면 시행 n 은 무수히 많아지고 사건이 일어날 확률은 p는 거의 0에 가깝게 된다.
$$
Y \text{ ~ Poisson R.V($\lambda$)} \\
X \text{ ~ Binomial R.V (n, p)} \\
Y = lim_{n \rightarrow \infty, p \rightarrow 0}X
$$


### PMF와 CDF

 시간의 nonegative이므로 정의역 k 는 0부터 시작한다. 
$$
P_{X}(k) = e^{\lambda}\frac{\lambda^{k}}{K!} \text{ for } k = 0,1,...\\
F_{X}(x) = e^{\lambda}\sum_{k=0}^{n}\frac{\lambda^{k}}{K!} \text{ for } n \leq x < n+1
$$

### 평균과 분산

Binomial에서 n의 시행동안 사건이 일어나는 평균은 np였기 때문에 $$\lambda = np$$ 로 생각할 수있고 평균 또한 직관적으로 $$\lambda$$ 가 된다.
$$
\mu_{X} = \lambda \\
\sigma_{X}^{2} = \lambda
$$


## Exponential

### 변수의 의미

 Poisson 분포를 따르는 사건이 처음 발생할 때까지 대기시간이며 파라미터로 $$\lambda > 0$$ 를 갖는다.

 Geometric R.V의 Continous Version이라고 생가하면 되고 유사하게 memoryless property를 갖는다. 즉 memoryless property로부터 그것을 만족하는 distribution을 유도하면 Cauchy Functional Equation에 의해서 Exponential ditribution이 유도된다.

### PDF와 CDF

 랜덤변수가 시간이므로 연속적인 값을 갖기 때문에 정의역도 양의 연속적인 값을 갖는다.
$$
f_{X}(x) = \lambda e^{-\lambda x} \text{ for } x > 0 \\
F_{X}(X) = 1 - e^{-\lambda x} \text{ for } x \geq 0
$$

### 평균과분산

평균은 $$\lambda$$ 가 클수록 대기시간이 길어지고, 작을수록 대기시간이 짧으므로 반비례적인 관계를 갖는다.
$$
\mu_{X} = \frac{1}{\lambda} \\
\sigma_{X}^{2} = \frac{1}{\lambda^{2}}
$$



## Rayleigh

### 변수의 의미

직교성분이 Gaussian R.V 일때 벡터의 크기 혹은 실수부와 허수부가 Gaussian 분포일 때 복소수의 크기를 의미하며 파라미터로 가우시안 분포의 표준편차인 $$\sigma$$ 를 갖는다.

직관적으로 Gaussian R.V 의 제곱합은 exponential R.V이기 때문에 루트 expontential R.V 와 관계를 갖음을 알 수 있다. 따라서 아래와 같은 관계식을 갖는다.
$$
Y \text{ ~ Rayleigh R.V}(\sigma) \\
X \text{ ~ Exponential R.V}(\lambda) \\
Y = \sqrt{X}, \sigma = \frac{1}{\sqrt{2\lambda}}
$$

### PDF와 CDF

크기의 값을 갖으므로 랜덤변수가 음이 아닌 실수값을 갖고 정의역도 음이아닌 실수가된다.
$$
f_{X}(x) = \frac{x}{\sigma^2}e^{-\frac{x^{2}}{2\sigma^{2}}} \text{ for } x \geq 0 \\
F_{X}(x) = 1 - e^{-\frac{x^{2}}{2\sigma^{2}}} \text{ for } x \geq 0
$$

### 평균과분산

각성분의 분산값이 커질수록 평균 벡터들의 값도 커지며 분산도 커지게 될것이다. 그래서 직관적으로 표준편차와 비례하는 형태를 갖을 것으로 예측할 수 있고 값을 아래와 같다.
$$
\mu_{X} = \sigma\sqrt{\frac{\pi}{2}} \\
\sigma_{X}^{2} = \frac{4 - \pi}{2}\sigma^{2}
$$


## Gamma

### 변수의 의미

 poisson분포를 따르는 사건이 $$\alpha$$ 번 일어날 때까지 대기시간을 의미하며 파라미터로는 $$\alpha, \lambda > 0$$ 를 갖는다.

Exponential R.V 와 Gamma R.V과의 관계 Bernoulli 와  Binomial과의 관계의 Continous Version이라고 생각하면 된다.  따라서 Exponential R.V 로 부터  Gamma R.V를 만들 수 있다는 말이다. 잘생각해보면 Exponential R.V 로 Rayleigh R.V를 만들수 있고 그 말은 Rayleigh R.V로 부터 Gamma R.V를 만들 수 있다는 말이다. 관계는 아래와 같다.
$$
Y\text{ ~ Gamma R.V with }(\alpha,\lambda) \\
X \text{ ~ Exponential R.V with } (\lambda) = \text{Gamma R.V with } (1,\lambda) \\
R_{i} \text{ ~ Rayleigh R.V}(\sigma) \\
R_i = \sqrt{X}, \sigma = \frac{1}{\sqrt{2\lambda}} \\
Y = \sum_{i=1}^{n}R_{i} \\
\begin{cases}
\alpha=n \\
\lambda = 2\sigma^2
\end{cases}
$$

### PDF와 CDF

 Exponetial 분포와 마찬가지로 랜덤변수가 시간의 의미를 갖기에 정의역도 양의 연속적인 값을 갖는다.
$$
f_{X}(x) = \frac{\lambda e^{-\lambda x (\lambda x)^{\alpha - 1}}}{\Gamma(\alpha)} \text{ for } x > 0 \\
\Gamma(\alpha) = \int_{0}^{\infty} e^{-x}x^{\alpha-1} dx  \\
F_{X}(x) = \int _{0}^{x} f_{X}(x) dx
$$

여기서 $$\Gamma(\alpha)$$ 는 Factorial 을 복소수로 확장한 의미를 지닌다고한다. 그래서 아래와 같은 성질을 지닌다.
$$
\Gamma(\alpha) =\Gamma(\alpha-1)\Gamma(\alpha-2)...\Gamma(1) \\
\text{양의 정수 n에 대해서 } \Gamma(n) = (n-1)!
$$


### 평균과 분산

 직관적으로 Exponential distribution의 평균에 $$ \alpha $$ 배 를  한 것임을 알 수있다.
$$
\mu_{X} = \frac{\alpha}{\lambda} \\
\sigma_{X}^{2} = \frac{\alpha}{\lambda^{2}}
$$


Gamma R.V를 이용하면 여러가지 다른 분포를 가진 R.V들을 만들어 낼 수있다. 또다른 분포들은 추후에 업데이트 하도록 할 것이다. 관계식을 보면 아래와 같다.
$$
Chi-Squared\text{ R.V with } (\lambda = \frac{1}{2}) = Gamma \text{ R.V with } (\alpha = \frac{n}{2}, \lambda = \frac{1}{2}) \\
Erlang \text{ R.V with } (n, \lambda =  Gamma \text{ R.V with } (n, \lambda)
$$
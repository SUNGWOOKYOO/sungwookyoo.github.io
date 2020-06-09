---
title: "Connection Between GAN, IRL and EBM"
excerpt: "GAN과 EBM과의 연결성"
categories:
 - papers
tags:
 - rl
use_math: true
last_modified_at: "2020-06-09"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption:  sample based MaxEnt IRL 알고리즘으로 EBM기반의 GAN 구조를 사용하는 타당성을 입증
 actions:
  - label: "논문"
    url: "https://arxiv.org/abs/1611.03852"
---



# Connection Between GAN, IRL and EBM

---



## 요약

sample based MaxEnt IRL과 GAN의 특수 case는 equivalent하고 MaxEnt IRL은 EBM의 일종이니까 GAN의 특수 case는 EBM의 일종이다. 즉 GAN과 EBM과의 연결성을 보여주었다. 그래서 sample based MaxEnt IRL 알고리즘으로 EBM을 기반으로 한  GAN 구조를 사용하는 타당성을 입증했다.



---

## 1. Introduction



### 1.1 MLE와 GAN의 장단점  

### 1.1.1 MLE  

 MLE는 data의 분포를 특정 distribution으로 modeling해서 data와 likelihood를 계산해서 이를 maximize하는 방향으로 data분포에 대한 파라미터를 찾아가는 방식이다. 그래서 data distribution을 evaluation할 수 있을 때 주로 사용된다. Diversity 측면에서 장점이 있으나 unrealistic, nonsensical sentence suboptimal behavior를 초래할 수 있다. 

### 1.1.2 GAN   
 GAN은 data 분포를 evaluation하지 않고 적대적 학습방법을 통해서 sample을 생성하는 분포를 학습하는 방법이다. diversity 측면에서는 단점이지만 original data에서 sample한 것과 같은 즉 실제같은 결과를 생성 할 수 있다. 즉 sample의 quality와 diversity 간의 trade-off가 존재한다.



### 1.1.3  Keypoint  

quality와 diversity 간의 trade-off를 타개하기 위해서 우리는 다음과 같은 방법을 생각해 볼수 있다. Discriminator가 주는 cost를 energy function으로 사용해서 data를 modeling해서 분포를 evaluation할 수 있도록 한뒤 MLE 방법으로 training하면 기존에 해결하지 못했던 문제를 해결할 수 있다.



---



## 2. Background  



### 2.1 GAN  

![png](/assets/images/connection/g_loss.png)

generator loss에서 기존식은 discriminator가 학습이 빨라서 잘되면 generator는 계속 실패하게 되어 학습이 잘 안되는 문제가 있었다. 오른쪽 term 을 사용했을때 gradient signal이 약해서 학습이 잘 안되니까 기울기가 큰 왼쪽 term을 더해서 학습이 더 잘되게 만들 수 있다고 한다.

$$
\mathcal{L}_{d}(D) = \mathbb{E}_{x \sim p}[-\log D(x)] + \mathbb{E}_{x \sim G}[-\log (1- D(x))] \\
\mathcal{L}_{g}(D) = \mathbb{E}_{x \sim G}[-\log D(x)] + \mathbb{E}_{x \sim G}[\log (1- D(x))]
$$

### 2.2 EBM  

$$
p_{\theta}(x) = \frac{1}{Z} e^{-c_{\theta}(\tau)}
$$

MaxEnt IRL에서는 cost function을 energy function으로 사용하여 demonstration의 분포를 modeling하고 MLE 방법을 통해서 분포를 추종했다. 그런데 Z를 계산하는 문제는 system dynamics를 알아야 계산할 수 있고 알더라도 SmalI MDP에서는 Dynamic Programinig을 통해서 풀수 있지만 Large MDP에서는 computationally challenging하여 문제가 있었다. 

### 2.3 GCL  

 Z를 근사해서 풀어보자는 게 Guided Cost Learning 방식이었다. MaxEnt formulation에 의해서 sample 기반 방법으로 Z를 근사하여 large MDP에서 문제를 풀었고 non-linear한 cost function을 학습할 수 있었다. 아래와 같이 EBM의 log likelihood를 전개하고 Z를 proposal distribution 혹은  sampling policy q를 도입하여 근사하고importance sampling을 통해서 구했다.

$$
\begin{split}
\mathcal{L}_{cost}(\theta) &= \mathbb{E}_{\tau \sim p}[-\log{p_{\theta}(\tau)}] \\
&= \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \log Z \\
&= \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \log \bigg(\mathbb{E}_{\tau \sim q}[ \frac{e^{-c_{\theta}(\tau)}}{q(\tau)}]\bigg)
\end{split}
$$

여기서 cost가 최적이라는 가정하에서 optimal q는  true cost function에 대한 demonstration distribution과 같다. 그래서 cost가 optimal distribution을 찾도록 guid 해주는 구조가 된다.   

q는 cost를 energy function으로 모델링 했기에 해가  cost function에 대한 지수족이되고 아래의 식에 의해서 구해진다.

$$
\begin{split}
\mathcal{L}_{sampler}(q) &= D_{KL}(q(\tau) || \frac{1}{Z}e^{-c_{\theta}(\tau)}) \\ &\approx D_{KL}(q(\tau) || e^{-c_{\theta}(\tau)})\\
&= \mathbb{E}_{q(\tau)}[\frac{q(\tau)}{e^{-c_{\theta}(\tau)}}] \\
&= \mathbb{E}_{\tau \sim q}[c_{\theta}(\tau)] + \mathbb{E}_{\tau \sim q}[\log q(\tau)]
\end{split}
$$

위의 식은 현재학습하고 있는 cost function의 값을 줄이고 학습되는 분포의 entropy를 최대화 하도록 하는 방향으로 학습하자는 의미이다.  

위의 식에서 문제점은 importance  sampling으로 인해 high variance가 발생한다는 점이다. 그래서 generated sample과 demo sample을 섞어서 mixture distribution $\mu = \frac{1}{2}\tilde{p}(\tau) + \frac{1}{2}q(\tau)$을 만들어 importance sampling을 하므로써 문제를 완화했다.  

$$
\mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \log \bigg(\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{ \frac{1}{2}\tilde{p}(\tau) + \frac{1}{2}q(\tau)}]\bigg)
$$

### 2.4 Direct Maximum Likelihood and Behavioral Cloning

 관측된 data point들을 갖고 MLE방법으로 분포를 직접구하는 방법은 generative model의 capacity가 data를 representation하기에 충분하지 않을 때 moment-matching 분포를 학습하게 되는 경향성을 보인다. 그 경향성 때문에 realistic한 data sample을 만들지 못한다. 그래서 분포를 직접 찾는것보다 각각의 mode를 설명하는 energy function을 찾는게 더 쉽고, 이후 학습된 energy function을 가지고 mode-seeking을 하는 generator를 학습하면 sample diversity와 quality를 둘다 충족할 수 있다.  

  Sequential decision making problem에서 직접 MLE로 분포를 구하는 것은 BC방법론으로 알려져있다.이것은 작고 간단한 문제에서는 잘 적용되었지만 Moment-matching behavior를 보이기 때문에 어려운 문제에서는 적용이 쉽지 않았다. 이는 compounding error때문인데 compound error란 policy가 작은 실수를 했을 때 trainig동안은 data분포와 많이 달라져서  더 많은 실수를  반복하게 되어 발생하는 error이다. 이 문제는 data를 sequential하게 생성할 때 더 심각한 문제가 된다.   

 이 문제를 다루기 위해서 data 분포와 data분포를 추종하는 모델 두가지로  mixture 분포를 만들어서 scheduled sampling을 하므로써 완화하고자 했다.   

---

## 3. GANs and IRL

 이번 절에서는  GAN의 특수 case가 sample based MaxEnt IRL과 equivalent한지를 보인다. discriminator의 special form을 제시하고 이를 따라서 전개하면 GAN의 objective는 the MaxEnt IRL objective를 최적화하는 것임을 밝혔다.  

### 3.1 A special form of discriminator

 $q(\tau)$를 generator의 분포라고하고 $p(\tau)$를 실제 data의 분포라고 할 때  optimal discriminator는 아래와 같다.

$$
D^{*}(\tau) = \frac{p(\tau)}{p(\tau) + q(\tau)}
$$

위의식은 실제 분포와 생성된 분포를 둘다 같은 확률인 1/2 로 판별할 때를 의미한다.  

 generator의 분포 $q(\tau)$를 알고 실제 data의 분포 $p(\tau)$를 모르는 상황에서 generator의 분포 $q(\tau)$ 를 evaluation하는 것이 우리의 목적이다. 그래서 data의 분포 $p(\tau)$ 를 모르는 우리는 energy function에 대한 bolzman 분포로 근사하여 문제를 풀면된다.   
    
$$
\begin{split}
D_{\theta}(\tau) &= \frac{\tilde{p}_{\theta}(\tau)}{\tilde{p}_{\theta}(\tau) + q(\tau)} \\
&= \frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}}{\frac{1}{Z}e^{-c_{\theta}(\tau)} + q(\tau)} \\
&= \frac{1}{1 + e^{c_{\theta}(\tau) +\log q(\tau) + \log Z}}
\end{split}
$$

 위의 식은 binary classification을 하는 sigmoid 식과 유사하며 의미적으로 생성된 샘플이 학습 데이터로 부터 판단하는 분류기의 의미를 지니고 있기 때문에 직관적으로 맞다.  input에서 $\log q(\tau)$ 를 빼고 sigmoid에 넣으면 cost function을 입력받아서 log Z를 bias 로하여 classification하는 식과 동일하고 optimal discrimator가 generator와 independent하게 동작하게 할 수 있다. 이 사실은 매우 중요한데 이유는 generator가 아직 학습이 덜되더라도 training 절차가 계속 일관되게 진행될 수 있기 때문이다. 따라서 우리는 generator의 분포 $q(\tau)$ 를 evaluation 하고자 할 때 data분포를 모르더라도 cheaply evaluation할 수 있고 training의 stability를 크게 향상시킬 수 있다. 

### 3.2 Equivalence between GAN and GCL

 이번절에서 우리는 3.1 절에서 말한 GAN의 variant 가  sample based MaxEnt IRL 방법인 GCL과 정확히 일치한다는 것을 보일 것이다.  

위해서 구한 discriminator의 variant를 Discriminator의 objective에 대입하고 GCL의 cost function objective에 대입하여 보자.  

$$
\require{color}
\begin{split}
\mathcal{L}_{discriminator}(D_{\theta}) &= \mathbb{E}_{\tau \sim p}[-\log D(\tau)] + \mathbb{E}_{\tau \sim q}[-\log (1- D(\tau))] \\
&= \mathbb{E}_{\tau \sim p}[-\log \textcolor{red}{\frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}}{\frac{1}{Z}e^{-c_{\theta}(\tau)} + q(\tau)}}] + \mathbb{E}_{\tau \sim q}[-\log \textcolor{red}{\frac{q(\tau)}{\frac{1}{Z}e^{-c_{\theta}(\tau)} + q(\tau)}}] \\
\mathcal{L}_{cost}(\theta) &= \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \log \bigg(\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{ \frac{1}{2}\tilde{p}(\tau) + \frac{1}{2}q(\tau)}]\bigg) \\
&= \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \log \bigg(\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{ \textcolor{red}{\frac{1}{2Z}e^{-c_{\theta}(\tau)}} + \frac{1}{2}q(\tau)}]\bigg)
\end{split}
$$

 우리는 mixture 분포 $\mu(\tau) = \frac{1}{2}p(\tau) + \frac{1}{2}q(\tau)$ 를  정의했고, 진짜 분포 $p(\tau)$를 모르니까  $\theta$ 와 $Z$ 를 사용해서 mixture 분포를 $\mu(\tau) \approx \tilde{\mu}(\tau) = \textcolor{red}{\frac{1}{2}\tilde{p}(\tau)} + \frac{1}{2}q(\tau) = \textcolor{red}{\frac{1}{2Z}e^{-c_{\theta}(\tau)}} + \frac{1}{2}q(\tau)$ 로 근사한다.

아래와 같은 3가지 사실을 보이면된다.

1. Discriminator의 loss를 최소화하는 Z는 partition function에 대한 importance sampling estimator이다.
2. 위에서 구한 Z에 대해서 구한 discriminator의 loss의 $\theta$에 대한 편미분은 MaxEnt IRL의 편미분과 동일하다.
3. Generator의 loss는 MaxEnt policy loss인 $c_{\theta} - H(q(\tau))$ 와 동일하다.

#### 3.2.1 Z estimates the partition function

discriminator의 loss의 Z에대한 최소점이 GCL에서 근사한 Z와 동일한지를 보이면 된다.  

$$
\begin{split}
\mathcal{L}_{discriminator}(D_{\theta})
&= \mathbb{E}_{\tau \sim p}[-\log \frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}}{\frac{1}{Z}e^{-c_{\theta}(\tau)} + q(\tau)}] + \mathbb{E}_{\tau \sim q}[-\log \frac{q(\tau)}{\frac{1}{Z}e^{-c_{\theta}(\tau)} + q(\tau)}] \\
&= \mathbb{E}_{\tau \sim p}[-\log \frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}] + \mathbb{E}_{\tau \sim q}[-\log \frac{q(\tau)}{\tilde{\mu}(\tau)}] \\
&= \log Z + \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] + \mathbb{E}_{\tau \sim \textcolor{red}{p}}[\log \tilde{\mu}(\tau)] - \mathbb{E}_{\tau \sim q}[\log q(\tau)] +\mathbb{E}_{\tau \sim \textcolor{red}{q}}[\log \tilde{\mu}(\tau)] \\
&= \log Z + \mathbb{E}_{\tau \sim p}[c_{\theta}(\tau)] - \mathbb{E}_{\tau \sim q}[\log q(\tau)] +2 \mathbb{E}_{\tau \sim \textcolor{red}{\mu}}[\log \tilde{\mu}(\tau)]
\end{split}
$$

위의 식에서 Z에 dependent한 term은 첫째와 마지막 term이다. 따라서 Z에 대해 편미분하면 아래와 같다.  

$$
\begin{split}
\partial_{z}\mathcal{L}_{discriminator}(D_{\theta}) 
&=\frac{1}{Z} + 2\mathbb{E}_{\tau \sim \mu}[\frac{\partial_{z}\tilde{\mu}(\tau)}{\tilde{\mu}(\tau)}] \\
&= \frac{1}{Z} - \frac{1}{Z^{2}}\mathbb{E}_{\tau \sim \mu}[\frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}] = 0\\
\therefore Z &= \mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}]
\end{split}
$$

따라서 GAN의 variant에서 discrimintor의 loss를 최적화 하는 Z가 GCL에서 Z인 partition function의 importance sampling estimation과 동일하다는 것을 보였다.

#### 3.2.2 $c_{\theta}$ optimizes the IRL objective

discriminator의 loss의 $\theta$ 에대한 최소점이 GCL의 cost objective와 동일한지를 보이면 된다.

$$
\begin{split}
\partial_{\theta}\mathcal{L}_{discriminator}(D_{\theta}) &= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] +2 \mathbb{E}_{\tau \sim \mu}[\frac{\partial_{\theta}\tilde{\mu}(\tau)}{\tilde{\mu}(\tau)}] \\
&= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] - \mathbb{E}_{\tau \sim \mu}[\frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}\partial_{\theta}c_{\theta}(\tau)}{\tilde{\mu}(\tau)}] 
\end{split}
$$

반면에 GCL의 cost 에 대한 objective의 미분값은 아래와 같다.  

$$
\begin{split}
\partial_{\theta}\mathcal{L}_{cost}(\theta) &= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] + \partial_{\theta}\log \bigg(\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}]\bigg) \\
&= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] + \frac{\partial_{\theta}\bigg( \mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}]\bigg)}{\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}]}
\end{split}
$$

여기서 우리는 $ \tilde{\mu} $ 와
$$
Z = \mathbb{E}_{ \tau \sim \mu}[ \frac{ e^{ -c_{\theta} (\tau) } }{\tilde{\mu} (\tau)} ]
$$
를 상수 취급 한다.   IRL optimization에서 $Z$ 를 구하고 나서 그 값으로 mixture policy $ \tilde{\mu} $ 를 구해서 importance sampling weight으로 사용하기 때문이다. 따라서 아래와 같이 전개되고 두 식이 같음을 보였다.

$$
\begin{split}
\partial_{\theta}\mathcal{L}_{cost}(\theta)
&= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] + \frac{\partial_{\theta}\bigg( \mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\textcolor{red}{\tilde{\mu}(\tau)}}]\bigg)}{\textcolor{red}{\mathbb{E}_{\tau \sim \mu}[ \frac{e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}]}} \\
&= \mathbb{E}_{\tau \sim p}[\partial_{\theta}c_{\theta}(\tau)] + \mathbb{E}_{\tau \sim \mu}[\frac{\frac{1}{\textcolor{red}{Z}}e^{-c_{\theta}(\tau)}\partial_{\theta}c_{\theta}(\tau)}{\textcolor{red}{\tilde{\mu}(\tau)}}] \\
&= \partial_{\theta}\mathcal{L}_{discriminator}(D_{\theta})
\end{split}
$$

#### 3.3 The generator optimizes the MaxEnt IRL objective

이번 절에서는 generator의 loss가 GCL의 sampler loss와 동일함을 보인다.

$$
\begin{split}
\mathcal{L}_{generator}(D) &= \mathbb{E}_{\tau \sim G}[\log (1- D(\tau)) -\log D(\tau)] \\
&= \mathbb{E}_{\tau \sim q}[\log \frac{q(\tau)}{\tilde{\mu}(\tau)} -\log \frac{\frac{1}{Z}e^{-c_{\theta}(\tau)}}{\tilde{\mu}(\tau)}] \\
&= \mathbb{E}_{\tau \sim q}[\log q(\tau)+ \log Z + c_{\theta}(\tau)] \\
&= \mathbb{E}_{\tau \sim q}[c_{\theta}(\tau)] + \mathbb{E}_{\tau \sim q}[\log q(\tau)] + \log Z \\
&= \mathcal{L}_{sample}(q) + \log Z 
\end{split}
$$

generator를 학습할때는 IRL step에서 한 Z를 fix해서 사용하기 때문에 상수로 취급하기 때문에 MaxEnt IRL의 sampler와 동일한 형태가 된다.  

이로써 3가지 사실을 보였기 때문에 GAN의 variant가 sample based MaxEnt IRL과 동일함을 보였다. 따라서 generator의 density 을 효과적으로 그리고 정확히 evaluation할 수 있는 구조를 통해서 GAN training을 함으로 써 sample들의 quality를 개선할 수 있다.    

## 4 GANs for training EBMs

EBM의 관점에서 GAN을 다시 살펴보자. 위에서 사용한 cost function을 energy function으로 생각하면 된다.  

EBM의 문제가 partition function을 구하는 것이 intractable하다는 것이 었다. 하지만 data분포를 모르더라도 EBM으로 근사하고 우리가 generator의 density만 안다면 unbiased partition funtion을 구할 수 있는 식을 유도했다. 이후  이 Z 값을 사용해서 energy function을 최적화하고 다시 구해진 energy function으로 generator를 최적화 하는 과정으로 EBM을 트레이닝하는데 GAN을 사용할 수 있다. 식은 아래와 같다.

$$
\begin{split}
Z &= \mathbb{E}_{x \sim \mu}[ \frac{e^{-E_{\theta}(x)}}{ \frac{1}{2}\tilde{p}(x) + \frac{1}{2}q(x)}] \\
\mathcal{L}_{energy}(\theta) &= \mathbb{E}_{x \sim p}[-\log p_{\theta}(x))] \\
&= \mathbb{E}_{x \sim p}[E_{\theta}(x)] + \log \bigg(\mathbb{E}_{x \sim \mu}[ \frac{e^{-E_{\theta}(x)}}{ \frac{1}{2}\tilde{p}(x) + \frac{1}{2}q(x)}]\bigg) \\
&= \mathbb{E}_{x \sim p}[E_{\theta}(x)] + \log Z  \\
\mathcal{L}_{generator}(q) = &= \mathbb{E}_{x \sim q}[E_{\theta}(x)] + \mathbb{E}_{x \sim q}[\log q(x)]

\end{split}
$$

여기서 우리는 $\tilde{p}(x)$ 를  EBM을 사용해서 $p_{\theta}(x)$ 로 두면 GAN의 special case가 되고 implement하기도 쉬워진다. optimal 한 discriminator를 생각했을 때 discriminator의 output은 $\sigma(E_{\theta} -\log q(x))$ 가 된다. 그래고 discriminator의 loss는 위의 식에서 보이듯이 data에 대한 log probability 이며 generator의 loss는 discriminator의 log odds가 된다.      

여기서 Energy function은 log likelihood로 design했지만 MSE로 할 수도 있고 f-divergence 같은 다양한 형태도 가능하다.  

## 5. Related Work

GAIL와의 차별점을 비교하면 GAIL의 policy 는 MaxEnt IRL 과 같은 objective로 학습되지만, discriminator의 경우 generator의 density를 사용하지 않고 단순히 sample 을 사용했기에 cost function이 discriminator에는 implicit하게 남아있다. 그래서  cost function이 어떻게 되는지 모르게 되기 때문에 학습 이후 discriminator는 버려진다.  

또한 2-level optimization을 하기 때문에 actor-critic과 유사한 문제를 겪는다. 그래서 actor-crtic에서 사용되는 optimization trick이 위의 방법론에도 똑같이 적용될 수 있다.  
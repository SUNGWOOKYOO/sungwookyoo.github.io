---
title: "Transformer"
excerpt: "global dependency를 잡아내고 횡방향으로 병렬 연산을 가능하게 하여 학습속도를 높인 RNN을 대체 가능한 attention model"
categories:
 - papers
tags:
 - attention mechanism
use_math: true
last_modified_at: "2020-03-15"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---
# Attention is all you need

## 1. Contribution

global dependency를 잡아내고 횡방향으로 병렬 연산을 가능하게 하여 학습속도를 높임

RNN 모델을 Attention으로 대체 가능하게 함

## 2. Background

### 2.1. ByteNet

sequential 한 연산을 줄이자는 목표

hidden representation을 병렬처리하기 위해 CNN활용

distant position 에 있는 dependency는 많은 연산을 필요로 함

### 2.2.ConvS2S

input요소들에 대한 절대적인 위치를 embedding

### 2.3. Self Attention =  Transformer

한 sequence 내에서 다른 position과의 대응관계를 통해 연결해주는 attention기법

RNN, CNN 구조를 사용하지 않고 hidden representation을 구한 최초 모델

Attention weight position에 평균을 취하여 효율성을 잃었지만 (이 operation이 상수로 고정되어 있다는 것은 무슨 말 ???) Multi-Head-Attention으로 이를 극복 (2~3배 연산량을 줄임)

## 3. Model

### 집중해서 봐야할 것

- Skip connection

- Layer normalization

- Multi-Head-Attention - 다수의 attention을 병렬로 연결

- position encoding

### 3.1 Encoder and Decoder Stacks

$$N_X$$ 는 그 계층을 N겹으로 쌓아 올렸다는 말 (stack)

#### 3.1.1 Encoder

6개를 stack 하였고 각 layer는 2개의 sub-layer (Multi-Head  Self Attention, FC)로 구성

각 sub-layer마다 skip connection 이후 normalization

수월한 skip connection을 위해 embedding layer의 출력과 sub-layer의 출력의 차원 수는 512로 통일

pseudo code로 과정을 나타내면 아래와 같다

```
Stage1_out = Embedding512 + TokenPositionEncoding512
Stage2_out = layer_normalization(multihead_attention(Stage1_out) + Stage1_out)
Stage3_out = layer_normalization(FFN(Stage2_out) + Stage2_out)

out_enc = Stage3_out
```



#### 3.1.2 Decoder

Encode와 동일하게 6개를 stack하였고 동일한 2개의 sub-layer를 갖음

또한 마찬가지로 각 sub-layer마다 skip connection 이후 normalization 

하지만 하나의 sub-layer인 Masked-Multi-Attention layer가 추가

**Stage 1**  - Decoder input - i번째 위치에 대한 예측이 이전의 예측값들에만 의존 할 수 있도록 디코더의 입력은 하나의 위치만큼 오프셋 된 임베딩의 출력으로 했다고 한다. (??? 정확히 이해가 안됨 그냥 인코더가 한거랑 같은 거 아닌가??)

**Stage2** -  Masked-Multi-Attention layer layer - 각 위치에 뒤따라오는 위치에 attend 하는 것을 방지하기 위해 약간 수정을 했다고 함. (어떻게 수정을 했다는 거지??)

[참조](https://pozalabs.github.io/transformer/)

*순차적으로* 결과를 만들어내야 하기 때문에, self-attention을 변형합니다. 바로 **masking**을 해주는 것이죠. masking을 통해, position i 보다 이후에 있는 position에 attention을 주지 못하게 합니다. 즉, position ii에 대한 예측은 미리 알고 있는 output들에만 의존을 하는 것입니다.

 **Stage2,3,4**  - 각 sub-layer마다 skip connection 이후 normalization

 pseudo code로 나타내면 아래와 같다.

```
Stage1_out = OutputEmbedding512 + TokenPositionEncoding512

Stage2_Mask = masked_multihead_attention(Stage1_out)
Stage2_Norm1 = layer_normalization(Stage2_Mask) + Stage1_out
Stage2_Multi = multihead_attention(Stage2_Norm1 + out_enc) +  Stage2_Norm1
Stage2_Norm2 = layer_normalization(Stage2_Multi) + Stage2_Multi

Stage3_FNN = FNN(Stage2_Norm2)
Stage3_Norm = layer_normalization(Stage3_FNN) + Stage2_Norm2

out_dec = Stage3_Norm
```

### 3.2. Attention

hidden representation 간의 관계를 compatibility function 통해 중요도인 attention weight를 구한뒤 weighted sum 을 하여 구함

#### 3.2.1 Scaled Dot-Product Attention

Compatibility function으로 대표적으로 scaled dot product 를 사용

Query는 decoder의 hidden state

Key는 encoder의 hidden state

value는 key가 얼만큼의 attention을 해야 되는 지를 나타내는 normalized weight 

$$d_k$$는 Q와 K의 dimension이고 $$d_v$$는 value의 dimension이며 각 값들은 linear projection에 의해서 결정된다.

정규화를 해주는 이유는 much faster, space-efficient 하도록 하기 위함이다.

Query와 key 가주어졌을 때

Q, K를 Matmul $$\rightarrow$$ Scaling $$\rightarrow$$ Mask(optional) $$\rightarrow$$ SoftMax $$\rightarrow$$ Value 와 Matmul

$$Attention(Q,K,V)=softmax(\frac{Q\cdot K^T}{\sqrt{d_k}})V$$

```python
def attention(Q, K, V):
    num = np.dot(Q, K.T)
    denum = np.sqrt(K.shape[0])
    return np.dot(softmax(num / denum), V)
```



#### 3.2.2. Multi-Head Attention

transformer가 sequence의 position에서 유사도를 찾는데 상수시간이 들지만 

단점으로 attention weight을 구하는 과정에서 평균을 취하기 때문에 유효해상도가 감소(??왜) 하는 비용을 가져옴

그 cost를 줄이기 위해 multi-head attention을 제시함

1. query와 key, value를 $$d_k$$차원에 linear projection

   $$head_i=Attention(Q\cdot W^Q_i,K\cdot W^K_i,V\cdot W^V_i),i=1,…,h$$

   where $$W^Q_i,W^K_i \in \R^{d_{model}×d_k},W^V_i \in  \R^{d_{model}×d_v} \mbox{ for } d_k=d_v=\frac{d_{model}}{h}=64$$

2. projected된 버젼의 h개의 query, key, value를 가지고 h번의 Scaled Dot-Product Attention

   결과는 $$d_v$$ dimension 

3. h개를 다시 concat 하고 다시 linear project을 시켜 최종값을 얻음

   $$MultiHeadAttention(Q,K,V)=Concat(head_1,…,head_h)W^O$$ where $W^O \in \R^{d_{hdv}\times d_{model}}$ 

여기서 h=8이라고 함

최종적으로 position의 다른 subspace에서의 representation 정보를 얻는다고 함

이유는 linear projection을 통해 다른 subspace에서의 표현방법들을 찾아서 복원시켰기 때문

single attention의 경우는 $d_{model}$의 dimension에서 attention을 하지만 multi-head attention의 경우는 $\frac{d_{model}}{h}$의 줄여진 dimension에서 attention을 한다. 

결과적으로 decoder의 각 위치는 입력 sequence의 모든 위치에 attend할 수 있었다.(?? 잘이해가 안됨)

### 3.3. Position-wise Feed-Forward Network

모든 position에 동일하게 적용되는 FC layer

Relu를 포함한 2개의 linear layer $ FFN(x)=max(0,xW_1+b_1)W_2+b_2 $ 로 구성

또다른 표현방식으로는 kernel size1의 2 convolution 로 디자인을 해도 됨

input과 output차원 $d_{model}$ 은 512이고, hidden (inner) layer의 차원 $d_{ff}$은 2048

### 3.4. Embeding and Softmax

 input output token을 벡터로 만들어주는데 learned embedding을 사용하여 학습을 하면서 embedding값이 변경되지 않도록 함

input과 output은 같은 embedding layer를 사용

decoder를 통해 나온 representation은 FC와 softmax를 거쳐 다음token의 probability로 나옴

### 3.5. Positional Encoding

 인코더 디코더의 하위 layer인 embedding layer에 'positional encoding을 추가하여 sequence 를 ordering (positional embedding)

positional encoding은 임베딩 차원과 동일하며 따라서 합쳐질 수 있음(뭐와 합쳐질 수 있다는 거지?? : embedding과 concat으로 합쳐짐)

positional encoding방법들 중  cosine function을 사용

pos는 위치 i는 차원?? 을 의미

encode time을 **sinusoidal** wave로 표기 하고 시간의 흐름을 위해서 하나의 input (**i**) 를 추가

sinusoidal을 이용하면 더 긴 sequence 에서도 잘 동작할 것이라 기대됨(이유는?)

$$ PE_{(pos,2i)}=sin(pos/10000^{2i/d_k}) $$

$$ PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_k}) $$

 각 positional encoding의 차원이 sin곡선을 가진다는 것을 의미

 $$PE_{pos+k}$$가 $$PE_{pos}$$의 linear function이 되므로 relative position의 정보를 배울 수 있을 것이라 가정

따라서 training때 없었던 길이의 sequence가 들어와도 잘 작동할 수 있을 것이라 생각

## 4. Self Attention

Encoder에서 self attention layer는 input으로 사용되는 Query, Key, value가 같은 위치 즉, 이전 encoder layer의 출력으로 부터 온다. 그래서 encoder의 각 position은 이전 encoder의 모든 위치들을 attend 함 (관계를 고려하게 된다.)

마찬가지로 Decoder에서 self attention layer는 각 position이 이전 decoder layer의 position과 현재의 position을 attend 함 auto regressive property를 유지하기 위해서, mask 를 사용해서 i position이후의 는 attend하지 못하도록 한다. (set to $-\infty$ all values)

RNN이나 CNN대신에 self attention 을 사용한 이유는 아래의 세가지

- Layer 마다 연산복잡도 최소화
- 연산의 병렬화 정도 최대화
- input과 output의 위치 사이에 최대 path length를 최소화,
  -  path길이가 짧을 수록 long range dependency들을 학습하기 쉽다고 함 

위의 세가지를 아래 네가지와 비교했다.

- Self-Attention
- Recurent
- Convolutional
- Self-Attention (restricted)

 Self-Attention이 모든 position을 상수시간이 소요되는 constant sequential operation으로 연결

RNN은 Self-Attention보다 n = sequence길이 < d = representation 차원 수 일때 layer당 연산 복잡도가 낮음

Restricted는 연산을 향상시키기 위해 self-attention을 $$r = neighborhood$$ 이내의 경우로 제한시킨것이고, 이경우 path 길이가 늘어나는 단점이 있는 trade-off가 있음

Conv layer는 커널로 인해 Rnn보다 주로 더 계산이 많다. 그러나 seperable convolution(??? 뭔지모름)은 복잡도를 확연히 줄일수가 있다

Self-attention을 통해 좀더 해석가능한 모델이 되는데, 이를 이용해 다른 task를 하도록 함(appendix참조)

## 5. Training

### 5.1 Training Data and Batching

4.5M의 sentence pair로 구성된 WMT2014 English-German 사용

sentence encoding은 source와 target vocabulary에서 37000개의 단어가 겹쳐 byte-pair encoding을 사용

36M의 sentence pair로 구성된 WMT2014 English-French 사용

sentence pair는 길이에 따라 batch로 묶였다. 각 batch는 약 25000개의 source token과 역시 약 25000개의 target token

### 5.2. Optimizer

 Adam 사용

$$\beta_1=0.9$$, $$\beta_2=0.98$$, $$\epsilon=10^{-9}$$ ,$$ lrate=d^{-0.5}_{model}\cdot min(stepnum^{-0.5}, stepnum\cdot warmstep^{-1.5}) $$ 

warmup_step=4000

warmup_step에서는 learning rate를 선형적으로 증가시켰다가 이후에는 step_number의 square root로 천천히 감소시킨 learning decay를 가능하게 함 

### 5.3.  Regularization

Residual dropout을 사용

각 sub-layer가 skip connection과 더해지고 normalized되기 전에, sub-layer의 output에 dropout

 embedding과정에서도 dropout

drop rate = 0.1

## 6. Result

### 6.1. Machine Translation

training cost = 학습시간 * 사용된 GPU수 * 각 GPU의 연산능력 으로 추정

WMT 2014 Eng-Ger에서 

big transformer model이 앙상블을 포함한 이전모델을 2.0 BLEU score로 앞섬 

base모델 역시 training 비용을 고려하였을때 이전모델들과 견줄만

WMT 2014 Eng-French에서

big model이 이전의 다른 single model보다 training 비용은 1/4로 줄었음에도 BLEU는 더 좋음

### 6.2. Model Variance

Eng-Germ 모델을 newstest 2013이라는 새로운 데이터에 적용

Transformer의 설정을 조금씩 바꿔보며 파라미터에 대한 variance기록

때 beam search를 사용 (beam search가 무엇??)

- 연산량은 유지하면서 attention head의 수나 key, value의 차원을 조절해보았다. head가 너무 많은것도, 적은것도 성능에 악영향을 줌

- attention key size $$d_k$$를 줄이는것 역시 결과가 안좋았다. 아마 compatibility function이 dot product의 이점보다 더욱 복잡한것이 필요

모델의 차원이 크고 head를 8에서 16으로 늘린 big model이 최고의 성능

## 의문점들

### Thoughts on the idea.

To my limited knowledge there are some statements that might benefit form more explanation:

1. How the scaling factor (Equation 3) makes an impact?
2. How actually the **positional encoding** work? Why they have chosen the sin/cos functions and why the position and dimension are in this relation? Finally how sinusoidal helps translate long sentences?
3. Does having separate position-wise FFNs help? (comparing to ConvS2S).
4. The *“cost of reduced effective resolution due to averaging attention-weighted position”* is claimed to be a motivation for multi-head attention. How to understand better what is the issue and how multi-head attention helps?
5. The Transformer brings a significantly improvement over ConvS2S, but where does the improvement come from? It is not clear from the work. ConvS2S lacks the self-attention, is it what brings the advantage?
6. Masked Attention. The problem of using same parts of input on different decoding step is claimed to be solved by penalizing (mask-out to −∞−∞) input tokens that have obtained high attention scores in the past decoding steps – a bit vague. How does it work? Maybe explicitly having a position-wise FFN automatically fixes that problem?
7. Applying multi-head attention might improve performance due to better parallelization. However, Table 3 also show increasing h=1to8h=1to8 improves accuracy. Why? Moving hh to 16 or 32 is not that beneficial. How to interpret this correctly?
8. How important the autoregression is in context of this architecture?

Please leave a comment if you have any other question, or would like to get more explanation on any of the paper’s particularities.


# Transformer 구조분석

Transformer

- Encoder

  - prepare mask
    - slf_attn_mask = get_attn_key_pad_mask
    - non_pad_mask = get_non_pad_mask
  - Embedding = src_word_emb + position_enc
    - src_word_emb = nn.Embedding
    - position_enc = nn.Embdding
  - layer_stack = nn.ModuleList
    - enc_layer = EncoderLayer
      - slf_attn = MultiHeadAttention*non_pad_mask        
      
      - pos_ffn = PositionwiseFeedForward*non_pad_mask
      

- Decoder

  - prepare mask

    - non_pad_mask = get_non_pad_mask
    - slf_attn_mask = slf_attn_mask_keypad + slf_attn_mask_subseq
      - slf_attn_mask_keypad = get_attn_key_pad_mask
      - slf_attn_mask_subseq = get_subsequent_mask
    - dec_enc_attn_mask = get_attn_key_pad_mask

  - Embedding = tgt_word_emb + position_enc

    - src_word_emb = nn.Embedding
    - position_enc = nn.Embdding

  - layer_stack = nn.ModuleList

    - dec_layer = DecoderLayer

      - slf_attn = MuiltiHeadaAttention*non_pad_mask

      - enc_ttn = MultiHeadAttention*non_pad_mask

      - pos_ffn = PositionwiseFeedForward*non_pad_mask

        

- tgt_word_prj = nn.Linear

  

MultiHeadAttention

- Linears and resize
  - w_qs = nn.Linear
  - w_ks = nn.Linear
  - w_vs = nn.Linear
  
- copy mask with n_head

- attention and restore size = ScaledDotProductAttention

  - query와 key를 batch-wise matrix multiplication with scaling 
  - mask값을 -$\infty$로
  - softmax = nn.Softmax
  - dropout = nn.Dropout
  - value와 batch-wise matrix multiplication

- fc = nn.Linear

- dropout = nn.Dropout

- layer_norm with residual connection = nn.LayerNorm

  

PositionwiseFeedForward
- resize
  - Transpose seq with hidden_size
    - [batch_size, seq, d_model] $\rightarrow$ [batch_size, d_model, seq]???
- w_1 = nn.Conv1d
- relu = import torch.nn.functional.relu
- w_2 = nn.Conv1d
- droupout = nn.Dropout
- laternorm with residual connection = nn.LayerNorm

# Transformer 모델 차원분석

[batch_size, src_seq_len]

```
src_seq = torch.Size([64, 29])
src_pos = torch.Size([64, 29])
tgt_seq = torch.Size([64, 34])
tgt_pos = torch.Size([64, 34])
#######################################Encoder###########################################
    slf_attn_mask in Encoder = torch.Size([64, 29, 29])
    non_pad_mask in Encoder = torch.Size([64, 29, 1])
    enc_output after embedding = torch.Size([64, 29, 512])
            q,k,v before Linear = torch.Size([64, 29, 512])
            q,k,v after Linear = torch.Size([64, 29, 8, 64])
            q,k,v after resize = torch.Size([512, 29, 64])
            mask after repeat = torch.Size([512, 29, 29])
                attn before softmax = torch.Size([512, 29, 29])
                attn atfter softmax = torch.Size([512, 29, 29])
            ouput after ScaledDotAttention = torch.Size([512, 29, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 29, 29])
            ouput after restore = torch.Size([64, 29, 512])
            ouput after fc = torch.Size([64, 29, 512])
            ouput after normalization = torch.Size([64, 29, 512])
        enc_output after Self_MultiHeadAttention= torch.Size([64, 29, 512])
        	output before PositionWiseFF = torch.Size([64, 29, 512])
            output after resize in PositionWiseFF = torch.Size([64, 512, 29])
            output after PositionWiseFF = torch.Size([64, 512, 29])
            output after restore size in PositionWiseFF = torch.Size([64, 29, 512])
            output after Norm in PositionWiseFF = torch.Size([64, 29, 512])
		enc_output after PositionwiseFF= torch.Size([64, 29, 512])
	enc_output after enc_layer= torch.Size([64, 29, 512])
	enc_slf_attn ater enc_layer = torch.Size([512, 29, 29])
###################################################encoder반복#########################
            q,k,v before Linear = torch.Size([64, 29, 512])
            q,k,v after Linear = torch.Size([64, 29, 8, 64])
            q,k,v after resize = torch.Size([512, 29, 64])
            mask after repeat = torch.Size([512, 29, 29])
                attn before softmax = torch.Size([512, 29, 29])
                attn atfter softmax = torch.Size([512, 29, 29])
            ouput after ScaledDotAttention = torch.Size([512, 29, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 29, 29])
            ouput after restore = torch.Size([64, 29, 512])
            ouput after fc = torch.Size([64, 29, 512])
            ouput after normalization = torch.Size([64, 29, 512])
		enc_output after Self_MultiHeadAttention= torch.Size([64, 29, 512])
            output before PositionWiseFF = torch.Size([64, 29, 512])
            output after resize in PositionWiseFF = torch.Size([64, 512, 29])
            output after PositionWiseFF = torch.Size([64, 512, 29])
            output after restore size in PositionWiseFF = torch.Size([64, 29, 512])
            output after Norm in PositionWiseFF = torch.Size([64, 29, 512])
		enc_output after PositionwiseFF= torch.Size([64, 29, 512])
	enc_output after enc_layer= torch.Size([64, 29, 512])
	enc_slf_attn ater enc_layer = torch.Size([512, 29, 29])
enc_final_output = torch.Size([64, 29, 512])
#############################################Decoder#####################################
    non_pad_mask in Decoder = torch.Size([64, 34, 1])
    slf_attn_mask in Decoder = torch.Size([64, 34, 34])
    dec_enc_attn_mask in Decoder = torch.Size([64, 34, 29])
    dec_output after embedding = torch.Size([64, 34, 512])
            q,k,v before Linear = torch.Size([64, 34, 512])
            q,k,v after Linear = torch.Size([64, 34, 8, 64])
            q,k,v after resize = torch.Size([512, 34, 64])
			mask after repeat = torch.Size([512, 34, 34])
                attn before softmax = torch.Size([512, 34, 34])
                attn atfter softmax = torch.Size([512, 34, 34])
            ouput after ScaledDotAttention = torch.Size([512, 34, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 34, 34])
            ouput after restore = torch.Size([64, 34, 512])
            ouput after fc = torch.Size([64, 34, 512])
            ouput after normalization = torch.Size([64, 34, 512])
		dec_output after Self_MultiHeadAttention= torch.Size([64, 34, 512])
            q,k,v before Linear = torch.Size([64, 34, 512])
            q,k,v after Linear = torch.Size([64, 34, 8, 64])
            q,k,v after resize = torch.Size([512, 34, 64])
            mask after repeat = torch.Size([512, 34, 29])
                attn before softmax = torch.Size([512, 34, 29])
                attn atfter softmax = torch.Size([512, 34, 29])
            ouput after ScaledDotAttention = torch.Size([512, 34, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 34, 29])
            ouput after restore = torch.Size([64, 34, 512])
            ouput after fc = torch.Size([64, 34, 512])
            ouput after normalization = torch.Size([64, 34, 512])
		dec_output after Cross_MultiHeadAttention= torch.Size([64, 34, 512])
            output before PositionWiseFF = torch.Size([64, 34, 512])
            output after resize in PositionWiseFF = torch.Size([64, 512, 34])
            output after PositionWiseFF = torch.Size([64, 512, 34])
            output after restore size in PositionWiseFF = torch.Size([64, 34, 512])
            output after Norm in PositionWiseFF = torch.Size([64, 34, 512])
		dec_output after PositionwiseFF= torch.Size([64, 34, 512])
    dec_output after dec_layer = torch.Size([64, 34, 512])
    dec_slf_attn after dec_layer = torch.Size([512, 34, 34])
    dec_enc_attn after dec_layer = torch.Size([512, 34, 29])
###########################################################Decoder 반복##################
			q,k,v before Linear = torch.Size([64, 34, 512])
            q,k,v after Linear = torch.Size([64, 34, 8, 64])
            q,k,v after resize = torch.Size([512, 34, 64])
            mask after repeat = torch.Size([512, 34, 34])
                attn before softmax = torch.Size([512, 34, 34])
                attn atfter softmax = torch.Size([512, 34, 34])
            ouput after ScaledDotAttention = torch.Size([512, 34, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 34, 34])
            ouput after restore = torch.Size([64, 34, 512])
            ouput after fc = torch.Size([64, 34, 512])
            ouput after normalization = torch.Size([64, 34, 512])
        dec_output after Self_MultiHeadAttention= torch.Size([64, 34, 512])
            q,k,v before Linear = torch.Size([64, 34, 512])
            q,k,v after Linear = torch.Size([64, 34, 8, 64])
            q,k,v after resize = torch.Size([512, 34, 64])
            mask after repeat = torch.Size([512, 34, 29])
                attn before softmax = torch.Size([512, 34, 29])
                attn atfter softmax = torch.Size([512, 34, 29])
            ouput after ScaledDotAttention = torch.Size([512, 34, 64])
            attn_weight after ScaledDotAttention = torch.Size([512, 34, 29])
            ouput after restore = torch.Size([64, 34, 512])
            ouput after fc = torch.Size([64, 34, 512])
            ouput after normalization = torch.Size([64, 34, 512])
		dec_output after Cross_MultiHeadAttention= torch.Size([64, 34, 512])
            output before PositionWiseFF = torch.Size([64, 34, 512])
            output after resize in PositionWiseFF = torch.Size([64, 512, 34])
            output after PositionWiseFF = torch.Size([64, 512, 34])
            output after restore size in PositionWiseFF = torch.Size([64, 34, 512])
            output after Norm in PositionWiseFF = torch.Size([64, 34, 512])
        dec_output after PositionwiseFF= torch.Size([64, 34, 512])
    dec_output after dec_layer = torch.Size([64, 34, 512])
    dec_slf_attn after dec_layer = torch.Size([512, 34, 34])
    dec_enc_attn after dec_layer = torch.Size([512, 34, 29])
dec_final_output = torch.Size([64, 34, 512])
seq_logit = torch.Size([64, 34, 3149])
seq_logit = torch.Size([2176, 3149])
```

의문점

- 왜 seq_logit 의 seq길이를 src_seq길이로 했을까?

# 분석

- enc_output, enc_slf_attn_list = Encoder(src_seq, src_pos)

  - prepare mask

    - slf_attn_mask = get_attn_key_pad_mask
    - non_pad_mask = get_non_pad_mask

  - Embedding = src_word_emb + position_enc

    - src_word_emb = nn.Embedding
    - position_enc = nn.Embdding

  - layer_stack = nn.ModuleList

    - enc_layer : enc_output, enc_slf_attn = EncoderLayer(enc_input, non_pad_mask, slf_attn_mask)

      - slf_attn : = enc_output, enc_slf_attn = MultiHeadAttention(enc_input, enc_input, enc_input, mask=slf_attn_mask)*non_pad_mask        

      - pos_ffn :  enc_output = PositionwiseFeedForward(enc_output)*non_pad_mask

        ​       

- dec_output, dec_slf_attn_list, dec_enc_attn_list = Decoder(tgt_seq, tgt_pos, src_seq, enc_output)

  - prepare mask

    - non_pad_mask = get_non_pad_mask
    - slf_attn_mask = slf_attn_mask_keypad + slf_attn_mask_subseq
      - slf_attn_mask_keypad = get_attn_key_pad_mask
      - slf_attn_mask_subseq = get_subsequent_mask
    - dec_enc_attn_mask = get_attn_key_pad_mask

  - Embedding = tgt_word_emb + position_enc

    - src_word_emb = nn.Embedding
    - position_enc = nn.Embdding

  - layer_stack = nn.ModuleList

    - dec_layer = DecoderLayer

      - slf_attn = MuiltiHeadaAttention*non_pad_mask

      - enc_ttn = MultiHeadAttention*non_pad_mask

      - pos_ffn = PositionwiseFeedForward*non_pad_mask

        

- tgt_word_prj = nn.Linear

  

MultiHeadAttention

- Linears and resize

  - w_qs = nn.Linear
  - w_ks = nn.Linear
  - w_vs = nn.Linear

- copy mask with n_head

- attention and restore size = ScaledDotProductAttention

  - query와 key를 batch-wise matrix multiplication with scaling 
  - mask값을 -$\infty$로
  - softmax = nn.Softmax
  - dropout = nn.Dropout
  - value와 batch-wise matrix multiplication

- fc = nn.Linear

- dropout = nn.Dropout

- layer_norm with residual connection = nn.LayerNorm

  

PositionwiseFeedForward

- resize
  - Transpose seq with hidden_size
    - [batch_size, seq, d_model] $\rightarrow$ [batch_size, d_model, seq]???
- w_1 = nn.Conv1d
- relu = import torch.nn.functional.relu
- w_2 = nn.Conv1d
- droupout = nn.Dropout
- laternorm with residual connection = nn.LayerNorm

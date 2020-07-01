---
title: "Transformer, Multi-head Attetnion Pytorch Guide Focusing on Masking"
excerpt: "how to use transformer pytorch module with masking details"
categories:
 - tips
 - study
tags:
 - pytorch
 - NLP
 - attention mechanism
use_math: true
last_modified_at: "2020-07-01"
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
import os, torch, sys, pickle, random
from torch import nn
from typing import List, Tuple
```

</div>

# Multi-head Attention - Focusing on Mask

pytorch 1.4.0 version 

I followed the notations in [offical document](https://pytorch.org/docs/stable/nn.html?highlight=multihead%20attention#torch.nn.MultiheadAttention) of pytorch

<img src="https://yjucho1.github.io/assets/img/2018-10-13/transformer.png" width="500">

Basically, multi-head attention mechanism is multiple scaled-dot attention version. <br>
Scaled-dot attention means as follows. <br>
Given `[query, key, value]`, <br>

|name|<div style="width:100px">dimension</div>|how to do|
|---------|---------|---------|
|query| $T \times N \times d_q$|embeded padded target sequence for a mini-batch|
|key| $S \times N \times d_k$|embeded padded input sequence for a mini-batch|
|value| $S \times N \times d_v$|embeded padded inpt sequence for a mini-batch|

<span style="color:red">Warning:</span>: constraints of multi-head attention inputs
* $d_k = d_v$
* $\frac{0}{0}$ makes nan value, 
    1. `src_len` or `trg_len` **should be upper than 1** because all masking in an example lead to `nan` value. 
    2. All `float('inf')` in one row or one column of`[src, tgt, memory]` lead to `nan` value 

Find similairity scores between `query, key`!
And then, apply the attention scores to `value`.

$$
\begin{align}
&B: \text{batch size} \\
&E: \text{embedding dimension} \\
&S: \text{max source sequence length} \\
&T: \text{max target sequence length} \\
\end{align}
$$

For complicated cases, I will use all masking conditions

|mask name|<div style="width:60px">dimension</div>|how to do|
|---------|---------|---------|
|key_padding_mask| $N \times S$| masking padding source sequence and target sequence for **each example**|
|attn_mask| $T \times S$| masking attention weights each positions for **all batch**|

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
B = 5
V = 100
N, E, S, T = 5, 16, 10, 20
nhead = 2
attn = nn.MultiheadAttention(embed_dim=E, num_heads=nhead)
emb = nn.Embedding(num_embeddings=V, embedding_dim=E, padding_idx=0)
```

</div>

Let's prepare a toy example as follows.

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
seq = torch.LongTensor([[random.randint(1, V - 1) for _ in range(S)] for _ in range(N)])  # [bsz, srclen]
for b in range(N):
    seq[b][random.randint(S//5, S - 5):] = 0 
print(seq)
```

</div>

{:.output_stream}

```
tensor([[75, 79, 46, 44, 68,  0,  0,  0,  0,  0],
        [84,  7, 75, 57,  0,  0,  0,  0,  0,  0],
        [65, 30, 41,  0,  0,  0,  0,  0,  0,  0],
        [88, 78,  0,  0,  0,  0,  0,  0,  0,  0],
        [98, 58, 34,  0,  0,  0,  0,  0,  0,  0]])

```

## Step 1. Generate Masking 

Q. Why use look-ahead masking?
> **ans**. [In orginal paper](https://arxiv.org/abs/1706.03762) ... <br>
We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. <br>
This masking, combined with fact that the output embeddings are offset by one position, ensures that the <br>
predictions for position `i` can depend only on the known outputs at positions less than `i`. <br>


<div class="prompt input_prompt">
In&nbsp;[67]:
</div>

<div class="input_area" markdown="1">

```python
# for self-attention masking
def sequence_mask(seq:torch.LongTensor, padding_idx:int=None) -> torch.BoolTensor:
    """ seq: [bsz, slen], which is padded, so padding_idx might be exist.     
    if True, '-inf' will be applied before applied scaled-dot attention"""
    return seq == padding_idx

# for decoder's look-ahead masking 
# (I think the reason to use this masking is taking into consideration that predict to next words only using preceding words in the decoder)
def look_ahead_mask(tgt_len:int, src_len:int) -> torch.FloatTensor:  
    """ this will be applied before sigmoid function, so '-inf' for proper positions needed. 
    look-ahead masking is used for decoder in transformer, 
    which prevents future target label affecting past-step target labels. """
    mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=1)
    mask[mask.bool()] = -float('inf')
    return mask
```

</div>

1. key pad masking 

|mask name|<div style="width:60px">dimension</div>|how to do|
|---------|---------|---------|
|key_padding_mask| $N  \times S$| masking padding source sequence and target sequence for **each example**|

This masking contrains the scope of self-attention for **each examples**.  <br>
Therefore, the model can **apply attention scores to only real sequences by avoiding padding index.** <br>

<span style="color:red"> Notice that </span> <br>
In the scaled-dot attention function ... <br>
another dimension(dim=1) of key_padding_mask are expanded with size $T$(so, it becomes $N \times T \times S$) <br>
in order to broadcast masking values to target sequences. 

<div class="prompt input_prompt">
In&nbsp;[68]:
</div>

<div class="input_area" markdown="1">

```python
key_padding_mask = sequence_mask(seq, padding_idx=0)
print(key_padding_mask)
print(key_padding_mask.shape)  # [bsz, src_len]
print()
padding_mask = key_padding_mask.unsqueeze(1).expand(-1, T, -1)  # [bsz, trg_len, src_len], which is mask for scaled-dot attention
print(padding_mask[0]), print()  # for example 0
# print(padding_mask[1]), print()  # for example 1

before_softmax = torch.rand(B, T, S).masked_fill(padding_mask, -float('inf'))
print(before_softmax[0])  # for example 0
# print(before_softmax[1])  # for example 1
```

</div>

{:.output_stream}

```
tensor([[False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True,  True,  True]])
torch.Size([5, 10])

tensor([[False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True,  True]])

tensor([[0.8794, 0.8458, 0.8516, 0.0447, 0.3812,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.1812, 0.4549, 0.7835, 0.8322, 0.6611,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.4658, 0.6365, 0.2442, 0.9905, 0.5118,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.5298, 0.9668, 0.2158, 0.1174, 0.8734,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.9967, 0.6162, 0.7594, 0.6391, 0.3663,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.8592, 0.8073, 0.2411, 0.2279, 0.6485,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.0724, 0.3111, 0.6000, 0.6570, 0.0180,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.7280, 0.6654, 0.3910, 0.3444, 0.9638,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.3494, 0.8876, 0.0506, 0.3111, 0.7949,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.9763, 0.7667, 0.4739, 0.5561, 0.2022,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.4080, 0.7799, 0.9461, 0.6745, 0.3950,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.2704, 0.6876, 0.6525, 0.1416, 0.9703,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.9985, 0.7673, 0.8472, 0.0541, 0.1273,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.5152, 0.4203, 0.3577, 0.1673, 0.4000,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.8417, 0.8949, 0.8560, 0.0523, 0.0803,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.0683, 0.0212, 0.1341, 0.6816, 0.7462,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.9225, 0.5103, 0.5363, 0.0149, 0.8378,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.6542, 0.2225, 0.4332, 0.9008, 0.9145,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.6001, 0.9155, 0.5877, 0.2048, 0.9796,   -inf,   -inf,   -inf,   -inf,
           -inf],
        [0.5510, 0.4078, 0.2633, 0.6340, 0.7613,   -inf,   -inf,   -inf,   -inf,
           -inf]])

```

2. look ahead masking

|mask name|<div style="width:100px">dimension</div>|how to do|
|---------|---------|---------|
|attn_mask| $T \times S$| masking attention weights each positions for **all batch**|

This masking contrains the scope of self-attention for **all examples**. <br>

look-ahead masking is used for decoder in transformer, which prevents future target label affecting the past-step target labels. 

<div class="prompt input_prompt">
In&nbsp;[69]:
</div>

<div class="input_area" markdown="1">

```python
attn_mask = look_ahead_mask(S, S)  # it is used for decorder 
print(attn_mask)  # [trg_len, src_len]
```

</div>

{:.output_stream}

```
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

```

# Step 2. Multi-head Attention

Q. why `contiguous()` for [q, k, v] are called? <br>

 - [x] I think if they are from same embedding, the algorithm needs to detach the gradient update for each other. <br>
     This is just my opionion. So, later on, I will find the reason.
     
please attention that <br>
if key_padding_mask is input of forward propagation, each example's attention weights after sequence length becomes zeros. <br>
However, `out` does not have zeros, this is because linear layer is propagated after applying attention weights. 

<div class="prompt input_prompt">
In&nbsp;[38]:
</div>

<div class="input_area" markdown="1">

```python
attn.train(mode=False)
```

</div>




{:.output_data_text}

```
MultiheadAttention(
  (out_proj): Linear(in_features=16, out_features=16, bias=True)
)
```



<div class="prompt input_prompt">
In&nbsp;[39]:
</div>

<div class="input_area" markdown="1">

```python
key_padding_mask
```

</div>




{:.output_data_text}

```
tensor([[False, False, False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True,  True,  True]])
```



If $q, k, v$ are projected by same embedding (just same embedding or linear projected), <br>
then sequence length of q, k, v are same. <br>
In this circumstance, multi-head attention is called self-attention.

Please note that if we use mask options, some attention weights become zeros
* key-padding mask: positions corresponding to `True` **parts of key_padding mask** for each example are zeros
* attn_mask: `-inf` **parts of attn_mask** are zeros

I will show you that using masking options lead to **zeros** of attention weights  as follows. <br>

<div class="prompt input_prompt">
In&nbsp;[40]:
</div>

<div class="input_area" markdown="1">

```python
Eseq = emb(seq) # [bsz, src_len, E]
q, k, v = Eseq.transpose(0, 1).contiguous(), Eseq.transpose(0, 1).contiguous(), Eseq.transpose(0, 1).contiguous()
print(q.shape, k.shape, v.shape)  # [T, ]
out, weights = attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # [src_len, bsz, E], [tar_len, bszsrc_len]
# out, weights = attn(q, k, v, key_padding_mask=None, attn_mask=attn_mask)  # [src_len, bsz, E], [tar_len, bszsrc_len]
print(out.shape, weights.shape), print()

print(seq.bool().sum(-1))
print(out[0])  
weights[0] # after sequence lengths vector values and upper triangle parts are zeros
```

</div>

{:.output_stream}

```
torch.Size([10, 5, 16]) torch.Size([10, 5, 16]) torch.Size([10, 5, 16])
torch.Size([10, 5, 16]) torch.Size([5, 10, 10])

tensor([5, 4, 3, 2, 3])
tensor([[ 0.3376, -0.7838,  0.1124,  0.3856,  1.0881, -0.1623,  0.1072,  0.3590,
         -0.6419, -0.0237, -0.2214,  0.0281,  0.8012, -0.0309,  0.0032, -0.2124],
        [ 0.4690,  0.8251, -0.6962, -0.1230,  0.3652, -0.0564,  0.1260,  0.7676,
         -0.2868,  0.9124,  0.0811,  0.8795, -0.4150, -0.9839, -0.0982,  0.4887],
        [-0.1854,  0.4487,  0.1684, -0.3133, -0.7565,  0.0864, -0.7580, -0.2905,
          0.5740, -0.4998, -0.0651, -0.2894, -0.4343,  0.2809, -0.3145,  0.1744],
        [-0.1030,  0.2608,  0.1130,  0.0331, -0.4297, -0.2702, -0.7032, -0.4559,
         -0.0314, -0.8184,  0.4495, -0.9816,  0.1393,  1.1237, -0.1796, -0.0038],
        [-0.3896, -0.4187, -0.5751,  0.6582, -0.3250, -0.4222, -0.2628, -0.5191,
          0.0186,  0.2831, -0.6457,  0.3321,  0.0746,  1.0277, -0.3817, -0.0720]],
       grad_fn=<SelectBackward>)

```




{:.output_data_text}

```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.5078, 0.4922, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2588, 0.4714, 0.2697, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.1556, 0.3744, 0.1879, 0.2821, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2304, 0.1647, 0.2204, 0.1221, 0.2624, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]], grad_fn=<SelectBackward>)
```



However, output corresponding to zeros of attention weights are not zeros because last layer is linear projection. <br>
Therefore, `[memory_key_padding_mask, memory_mask]` is needed to padding the output of encoder to decoder in translation model. <br>
I will explain these masks in decoder part.

---

# Transformer model

How to use in the transformer model? <br>

[Transformer layers document](https://pytorch.org/docs/stable/nn.html?highlight=transformer%20encoder#torch.nn.TransformerEncoderLayer)
describe this, but the examples in there are ambiguous to understand when using maskings. 

Let's do a simple example, but use all maskings. 

## Use Encoder module

For simplicity, I use only one layer of transformer encoder, so I used `nn.TransformerEncoderLayer`

<span style="color:red">Warning:</span> dropout is used, so use `model.train(mode=False)` before forward propagation for test purposes.

Please note that **transformer encoder's src and trg are same**, so scaled-dot attention for the same sequence means **self-attention**!.

`Eseq` is used for input sequence, where the dimension of `Eseq` is `[B, S, E]`.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
encoder = torch.nn.TransformerEncoderLayer(d_model=E, nhead=nhead, dim_feedforward=4*E)
encoder.train(mode=False)
```

</div>




{:.output_data_text}

```
TransformerEncoderLayer(
  (self_attn): MultiheadAttention(
    (out_proj): Linear(in_features=16, out_features=16, bias=True)
  )
  (linear1): Linear(in_features=16, out_features=64, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (linear2): Linear(in_features=64, out_features=16, bias=True)
  (norm1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (norm2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (dropout1): Dropout(p=0.1, inplace=False)
  (dropout2): Dropout(p=0.1, inplace=False)
)
```



`My view:`
lthough it is possible to obtain embedding with self-attention, it is a pity that the model does not output the self-attention score.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
src_mask = torch.zeros(S, S)  # [S, S], which means self-attention
print(src_mask.shape)
print(key_padding_mask.shape)  # [bsz, S]
out = encoder(Eseq.transpose(0, 1), src_mask=src_mask, src_key_padding_mask=key_padding_mask)  # [S, bsz, E] 
print(out.shape)
out[0]
```

</div>

{:.output_stream}

```
torch.Size([10, 10])
torch.Size([5, 10])
torch.Size([10, 5, 16])

```




{:.output_data_text}

```
tensor([[-1.2512, -1.1156, -1.0019,  0.0576, -0.5861, -1.0735, -0.1735,  0.5430,
          0.0823,  1.6950,  1.4143, -0.3691,  1.7459,  0.2939,  0.9126, -1.1740],
        [-0.6578, -0.4739,  0.8454, -1.3896, -0.5582,  0.8627,  0.2286,  0.2346,
          1.4226,  0.1236, -0.5403, -0.1841,  2.4831, -0.9266, -0.0057, -1.4644],
        [ 0.2509, -1.4570,  2.3970, -1.3271, -1.6661,  0.6993, -0.1136, -0.2787,
         -0.2379,  0.0055,  0.8749,  0.2460, -0.9369,  0.9882,  0.1042,  0.4514],
        [ 1.8043, -0.9929,  0.6597, -0.9512,  0.8177, -0.4477,  0.3470,  0.9663,
         -0.3772, -0.2731, -2.0912, -0.2372,  1.4346, -1.1369,  0.6771, -0.1993],
        [ 0.1023, -0.5971,  0.5727, -1.4589,  0.5111,  0.4181,  0.0157, -2.3575,
          1.7297, -0.1671,  1.0217, -0.6159, -0.6257, -0.4689,  0.5827,  1.3372]],
       grad_fn=<SelectBackward>)
```



<div class="prompt input_prompt">
In&nbsp;[44]:
</div>

<div class="input_area" markdown="1">

```python
torch.isnan(out).sum()  # sanity check
```

</div>




{:.output_data_text}

```
tensor(0)
```



## Use Decoder module

> Q. What is the memory mask? [see this discuss in pytorch community](https://discuss.pytorch.org/t/memory-mask-in-nn-transformer/55230) <br>
> **ans.** It’s an attention mask **working on the second input of transformer decoder layer**.  <br>
To be more specific, the memory mask applies to second part of multi-head attention in the decoder, <br>
which means look-ahead mask between input sequence and target sequence. <br>
Within the encoder-decoder architecture, it works on the output of transformer encoder, which we call it “memory”.

<img src="https://miro.medium.com/max/1072/1*MBc5BeHRr6wtc3R0PU81xg.png" width="400">

Encoder outputs embeddings with self-attention, which called memory(or context) vectors and then pass the memory vectors to decoder. <br>
However, in the memory, which position should the decoder focus on? <br>
Therefore, `[memory_key_padding_mask, memory_mask]` needed.

To be more specific, they are used in second part of decoder's multi-head attention<br> 
* `memory_key_padding_mask` is just encoder's key_padding_mask, of shape `[N, S]`.
* `memory_mask` is look-ahead mask to encoder key, value to decoder, of shape `[T, S]`.

<div class="prompt input_prompt">
In&nbsp;[70]:
</div>

<div class="input_area" markdown="1">

```python
mem_mask = look_ahead_mask(T, S)
mem_mask  # [T, S]
```

</div>




{:.output_data_text}

```
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```



<div class="prompt input_prompt">
In&nbsp;[71]:
</div>

<div class="input_area" markdown="1">

```python
decoder = nn.TransformerDecoderLayer(d_model=E, nhead=nhead, dim_feedforward=4*E)
decoder.train(mode=False)
```

</div>




{:.output_data_text}

```
TransformerDecoderLayer(
  (self_attn): MultiheadAttention(
    (out_proj): Linear(in_features=16, out_features=16, bias=True)
  )
  (multihead_attn): MultiheadAttention(
    (out_proj): Linear(in_features=16, out_features=16, bias=True)
  )
  (linear1): Linear(in_features=16, out_features=64, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (linear2): Linear(in_features=64, out_features=16, bias=True)
  (norm1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (norm2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (norm3): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (dropout1): Dropout(p=0.1, inplace=False)
  (dropout2): Dropout(p=0.1, inplace=False)
  (dropout3): Dropout(p=0.1, inplace=False)
)
```



### Generate target sequence

<div class="prompt input_prompt">
In&nbsp;[72]:
</div>

<div class="input_area" markdown="1">

```python
trg = torch.LongTensor([[random.randint(1, V - 1) for _ in range(T)] for _ in range(N)])  # [bsz, srclen]
for b in range(N):
    trg[b][random.randint(T//5, T - 3):] = 0 
print(trg)  # [bsz, T]

Etrg = emb(trg)  # [bsz, T, E]
Etrg.shape
```

</div>

{:.output_stream}

```
tensor([[34, 71, 38, 71,  3, 74, 24, 30, 55, 30, 56, 20,  0,  0,  0,  0,  0,  0,
          0,  0],
        [97, 88, 23, 98, 23, 98,  3,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0],
        [95,  2, 45, 31, 40, 10, 11, 71, 96, 32,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0],
        [47, 10, 62, 76, 24, 36, 93,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0],
        [95, 37, 57, 75,  3, 53, 24, 85,  4, 33, 23, 41, 83, 74, 18, 26, 51,  0,
          0,  0]])

```




{:.output_data_text}

```
torch.Size([5, 20, 16])
```



<center>Follow pytorch community document</center>

| name | dimension   |
| ---- | ----------- |
| src  | $(S, N, E)$ |
|tgt| $(T, N, E)$|
|src_mask|$(S, S)$|
|tgt_mask|$(T, T)$|
|memory_mask|$(T,S)$|
|src_key_padding_mask|$(N, S)$|
|tgt_key_padding_mask|$(N, T)$|
|memory_key_padding_mask|$(N, S)$|

Note that `memory_key_padding_mask` is same with `src_key_padding_mask`. <br>
This is because both query and key of the decoder's second part of multi-head attention is the output of encoder. <br>
Therefore, the encoder's memory(or context) information will be delivered to connected decoder as `[key, value and key_padding_mask]`.<br>

<div class="prompt input_prompt">
In&nbsp;[75]:
</div>

<div class="input_area" markdown="1">

```python
memory = encoder(Eseq.transpose(0, 1), src_mask=src_mask, src_key_padding_mask=key_padding_mask)  # [S, bsz, E]
print(memory.shape)
output = decoder(Etrg.transpose(0, 1), memory=memory, 
                tgt_mask=look_ahead_mask(T, T), memory_mask=mem_mask, 
                 tgt_key_padding_mask=sequence_mask(trg, padding_idx=0), 
                memory_key_padding_mask=key_padding_mask) 
print(output.shape)  # [T, bsz, E]
```

</div>

{:.output_stream}

```
torch.Size([10, 5, 16])
torch.Size([20, 5, 16])

```

<div class="prompt input_prompt">
In&nbsp;[76]:
</div>

<div class="input_area" markdown="1">

```python
out = output.transpose(0, 1)  # [bsz, T, E]
out[0]
```

</div>




{:.output_data_text}

```
tensor([[ 2.0530e-01,  1.5013e+00,  1.0501e+00,  3.5751e-01,  6.3936e-01,
          9.2592e-01,  6.6488e-01, -2.5640e-01, -4.9700e-01,  6.8812e-01,
         -1.7615e+00,  1.0349e+00, -8.9191e-01, -1.6123e+00, -6.2900e-01,
         -1.4193e+00],
        [-4.4284e-01,  1.9421e+00,  6.1783e-01, -4.9344e-01,  2.1229e+00,
          4.7583e-02,  1.6725e-01,  6.9770e-01, -1.1447e+00, -1.4100e+00,
         -7.2436e-01, -1.3630e+00,  6.1289e-01, -3.1280e-01, -4.5737e-01,
          1.4021e-01],
        [ 3.5275e-01, -5.5859e-01, -9.4811e-01,  3.3188e-01, -1.1998e-03,
         -6.0703e-01, -7.5305e-01,  2.1518e+00, -1.2262e+00, -9.4202e-02,
         -1.0129e+00,  4.2685e-01, -1.4186e+00,  1.6577e+00,  1.0287e+00,
          6.7026e-01],
        [-5.0308e-01,  1.9444e+00,  6.1748e-01, -3.7436e-01,  1.9657e+00,
          1.6569e-01,  3.2078e-02,  8.3904e-01, -1.5390e+00, -1.4493e+00,
         -6.0518e-01, -1.2885e+00,  4.4276e-01, -2.1376e-01, -2.5178e-01,
          2.1766e-01],
        [ 1.0492e+00,  1.5197e+00,  6.3454e-01,  1.0607e-01, -6.4794e-01,
          1.0796e+00, -5.2467e-01, -1.7117e-01,  1.5651e-01, -2.3230e+00,
         -1.2304e+00,  2.5204e-01, -2.5927e-01, -3.0521e-01, -8.7363e-01,
          1.5376e+00],
        [ 1.1820e+00, -1.0573e+00,  1.7224e+00,  1.1034e-01,  3.5638e-01,
          2.0074e-01, -2.0558e+00,  5.5258e-01,  7.8305e-01, -6.4929e-01,
         -1.4155e+00, -1.1916e-02, -3.8352e-01,  1.5048e+00, -4.7025e-01,
         -3.6871e-01],
        [ 1.8280e+00,  3.6645e-01,  1.6182e-01, -1.4546e+00,  3.5153e-01,
         -1.2442e+00, -1.1728e+00, -1.1383e+00,  1.3857e+00,  3.4732e-01,
         -3.3480e-01, -3.3509e-01,  1.7011e+00, -1.5128e-01,  4.8057e-01,
         -7.9143e-01],
        [ 8.4219e-01, -2.2292e-01,  1.6207e+00, -1.5880e-01, -1.9941e+00,
         -2.6402e-01, -1.0795e-01, -1.2542e+00, -3.7832e-01, -1.3619e+00,
         -3.0050e-01,  6.4807e-01,  8.8309e-01,  7.8289e-01, -4.2921e-01,
          1.6950e+00],
        [-5.2335e-01, -6.4032e-01, -6.4664e-01,  6.1068e-01, -5.7863e-02,
          2.1731e+00, -1.5652e+00,  3.1220e-01,  2.7384e-01, -1.1730e+00,
         -1.1549e-01,  1.0999e-02, -1.4816e+00,  1.5529e+00,  1.0676e+00,
          2.0224e-01],
        [ 8.0632e-01, -2.7476e-01,  1.7004e+00, -3.7164e-01, -1.9823e+00,
         -3.1552e-01, -8.3502e-02, -1.1603e+00, -4.2286e-01, -1.2626e+00,
         -1.8362e-01,  7.0035e-01,  9.3423e-01,  6.5968e-01, -4.8287e-01,
          1.7390e+00],
        [-1.2699e-01,  1.3019e+00,  2.5125e-01, -4.2549e-01,  1.7312e+00,
         -1.7435e+00,  1.8060e-01,  5.0046e-01, -8.4960e-01, -2.8272e-01,
         -1.3480e+00,  2.2478e-01,  2.4194e-01,  8.5712e-01, -1.6936e+00,
          1.1806e+00],
        [ 3.9400e-01,  1.7380e+00, -7.8590e-01,  4.9670e-01,  4.0187e-01,
          1.5195e+00, -1.9596e+00, -8.9633e-01, -1.1844e+00, -1.5570e-01,
         -6.3172e-01, -8.9354e-02, -1.0397e+00,  1.0618e+00,  8.5174e-01,
          2.7912e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01]], grad_fn=<SelectBackward>)
```



Warnning: decoder does not output attention weights, hard to figure out what parts are masked

<div class="prompt input_prompt">
In&nbsp;[77]:
</div>

<div class="input_area" markdown="1">

```python
trg_lengths = trg.bool().sum(-1)
print(trg_lengths)  # trg sequence lengths
out[0][:trg_lengths[0] + 3]  # after trg sequence lengths vector values are same, buy not zeros.
```

</div>

{:.output_stream}

```
tensor([12,  8, 10,  7, 17])

```




{:.output_data_text}

```
tensor([[ 2.0530e-01,  1.5013e+00,  1.0501e+00,  3.5751e-01,  6.3936e-01,
          9.2592e-01,  6.6488e-01, -2.5640e-01, -4.9700e-01,  6.8812e-01,
         -1.7615e+00,  1.0349e+00, -8.9191e-01, -1.6123e+00, -6.2900e-01,
         -1.4193e+00],
        [-4.4284e-01,  1.9421e+00,  6.1783e-01, -4.9344e-01,  2.1229e+00,
          4.7583e-02,  1.6725e-01,  6.9770e-01, -1.1447e+00, -1.4100e+00,
         -7.2436e-01, -1.3630e+00,  6.1289e-01, -3.1280e-01, -4.5737e-01,
          1.4021e-01],
        [ 3.5275e-01, -5.5859e-01, -9.4811e-01,  3.3188e-01, -1.1998e-03,
         -6.0703e-01, -7.5305e-01,  2.1518e+00, -1.2262e+00, -9.4202e-02,
         -1.0129e+00,  4.2685e-01, -1.4186e+00,  1.6577e+00,  1.0287e+00,
          6.7026e-01],
        [-5.0308e-01,  1.9444e+00,  6.1748e-01, -3.7436e-01,  1.9657e+00,
          1.6569e-01,  3.2078e-02,  8.3904e-01, -1.5390e+00, -1.4493e+00,
         -6.0518e-01, -1.2885e+00,  4.4276e-01, -2.1376e-01, -2.5178e-01,
          2.1766e-01],
        [ 1.0492e+00,  1.5197e+00,  6.3454e-01,  1.0607e-01, -6.4794e-01,
          1.0796e+00, -5.2467e-01, -1.7117e-01,  1.5651e-01, -2.3230e+00,
         -1.2304e+00,  2.5204e-01, -2.5927e-01, -3.0521e-01, -8.7363e-01,
          1.5376e+00],
        [ 1.1820e+00, -1.0573e+00,  1.7224e+00,  1.1034e-01,  3.5638e-01,
          2.0074e-01, -2.0558e+00,  5.5258e-01,  7.8305e-01, -6.4929e-01,
         -1.4155e+00, -1.1916e-02, -3.8352e-01,  1.5048e+00, -4.7025e-01,
         -3.6871e-01],
        [ 1.8280e+00,  3.6645e-01,  1.6182e-01, -1.4546e+00,  3.5153e-01,
         -1.2442e+00, -1.1728e+00, -1.1383e+00,  1.3857e+00,  3.4732e-01,
         -3.3480e-01, -3.3509e-01,  1.7011e+00, -1.5128e-01,  4.8057e-01,
         -7.9143e-01],
        [ 8.4219e-01, -2.2292e-01,  1.6207e+00, -1.5880e-01, -1.9941e+00,
         -2.6402e-01, -1.0795e-01, -1.2542e+00, -3.7832e-01, -1.3619e+00,
         -3.0050e-01,  6.4807e-01,  8.8309e-01,  7.8289e-01, -4.2921e-01,
          1.6950e+00],
        [-5.2335e-01, -6.4032e-01, -6.4664e-01,  6.1068e-01, -5.7863e-02,
          2.1731e+00, -1.5652e+00,  3.1220e-01,  2.7384e-01, -1.1730e+00,
         -1.1549e-01,  1.0999e-02, -1.4816e+00,  1.5529e+00,  1.0676e+00,
          2.0224e-01],
        [ 8.0632e-01, -2.7476e-01,  1.7004e+00, -3.7164e-01, -1.9823e+00,
         -3.1552e-01, -8.3502e-02, -1.1603e+00, -4.2286e-01, -1.2626e+00,
         -1.8362e-01,  7.0035e-01,  9.3423e-01,  6.5968e-01, -4.8287e-01,
          1.7390e+00],
        [-1.2699e-01,  1.3019e+00,  2.5125e-01, -4.2549e-01,  1.7312e+00,
         -1.7435e+00,  1.8060e-01,  5.0046e-01, -8.4960e-01, -2.8272e-01,
         -1.3480e+00,  2.2478e-01,  2.4194e-01,  8.5712e-01, -1.6936e+00,
          1.1806e+00],
        [ 3.9400e-01,  1.7380e+00, -7.8590e-01,  4.9670e-01,  4.0187e-01,
          1.5195e+00, -1.9596e+00, -8.9633e-01, -1.1844e+00, -1.5570e-01,
         -6.3172e-01, -8.9354e-02, -1.0397e+00,  1.0618e+00,  8.5174e-01,
          2.7912e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01],
        [ 9.2153e-01,  4.3841e-02,  3.8642e-01,  7.7932e-01, -6.8301e-02,
          3.4165e-01, -1.3100e+00, -1.2595e+00, -9.2050e-01, -2.1033e+00,
          3.6177e-01,  1.3866e+00,  5.1212e-01,  1.5935e+00, -8.8966e-01,
          2.2454e-01]], grad_fn=<SliceBackward>)
```



<div class="prompt input_prompt">
In&nbsp;[78]:
</div>

<div class="input_area" markdown="1">

```python
torch.isnan(out).sum()
```

</div>




{:.output_data_text}

```
tensor(0)
```



# Summary 

masking is used for helping multi-head-attention properly.
summarize all masking in transformer module.

```python
# for self-attention masking
def sequence_mask(seq:torch.LongTensor, padding_idx:int=None) -> torch.BoolTensor:
    """ seq: [bsz, slen], which is padded, so padding_idx might be exist.     
    if True, '-inf' will be applied before applied scaled-dot attention"""
    return seq == padding_idx

# for decoder's look-ahead masking 
def look_ahead_mask(tgt_len:int, src_len:int) -> torch.FloatTensor:  
    """ this will be applied before sigmoid function, so '-inf' for proper positions needed. 
    look-ahead masking is used for decoder in transformer, 
    which prevents future target label affecting past-step target labels. """
    mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=1)
    mask[mask.bool()] = -float('inf')
    return mask
```


Masking Summary for Translation Model

| used for                       | key_padding_mask                                             | attn_mask                                        |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| Encoder (self-attn)            | src_key_padding_mask $(N, S)$  <br>`sequence_mask(seq, padding_idx=0)` | src_mask $(S, S)$ <br>`torch.ones(S, S)`         |
| Decoder (cross-attn, 2nd part) | memory_key_padding_mask $(N, S)$ <br> `sequence_mask(seq, padding_idx=0)` | memory_mask $(T, S)$ <br>`look_ahead_mask(T, S)` |
| Decoder (self-attn, 1st part)  | tgt_key_padding_mask $(N, T)$ <br>`sequence_mask(tgt, padding_idx=0)` | tgt_mask $(T, T)$ <br>`look_ahead_mask(T, T)`    |

Note that cell of (Encoder - key_padding_mask) and cell of (Decoder, 2nd part - key_padding_mask) are the same acutally.  <br>
This is because Encoder outputs context vectors.  <br>
In the context vector, memory_key_padding_mask helps true sequence of context vectors to applied to decoder.

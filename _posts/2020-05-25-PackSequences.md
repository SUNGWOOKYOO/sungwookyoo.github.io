---
title: "Pack, Pad Sequence technique in pytorch "
excerpt: "Pack padded sequence or pad packed sequence."
categories:
 - tips
tags:
 - pytorch
use_math: true
last_modified_at: "2020-05-25"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# Pack, Pad Sequence 
`pack padded sequence`, `pad packed sequence`

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import torch 
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
torch.__version__
```

</div>




{:.output_data_text}

```
'1.4.0'
```



<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
V, D, H, T, B = 1000, 10, 5, 3, 2
seq = torch.randint(low=0, high=100, size=(B, T))
seqlen = torch.randint(low=1, high=T + 1, size=(B, )) # more than 1
for b in range(B):
    seq[b, seqlen[b]:] = 0
print(seq, seqlen)
```

</div>

{:.output_stream}

```
tensor([[38, 20,  4],
        [81,  0,  0]]) tensor([3, 1])

```

* objective

<!-- <img src="https://dl.dropbox.com/s/3ze3svhdz05aakk/0705img3.gif" width="600"> -->

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
E = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0)
GRU = nn.GRU(input_size=D, hidden_size=H, num_layers=1)
```

</div>

## w/o packing technique

h 는 항상 마지막, time step 의 hidden state값이 된다. <br>
따라서, seqlen에 의해 마지막 state를 구하려면 약간 불편하다. 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
hs, h = GRU(E(seq).transpose(0, 1))
hs = hs.transpose(0, 1)
hs, hs.shape, h, h.shape
```

</div>




{:.output_data_text}

```
(tensor([[[ 0.1499,  0.0979, -0.4934, -0.5707, -0.3808],
          [ 0.2610,  0.1976,  0.2972, -0.0386, -0.3643],
          [-0.4342,  0.1079, -0.8891, -0.3130, -0.0115]],
 
         [[-0.1040, -0.0475,  0.1698,  0.2809, -0.0018],
          [-0.0103,  0.0398, -0.1869,  0.2224, -0.0272],
          [ 0.0554,  0.1163, -0.3720,  0.1830, -0.0198]]],
        grad_fn=<TransposeBackward0>),
 torch.Size([2, 3, 5]),
 tensor([[[-0.4342,  0.1079, -0.8891, -0.3130, -0.0115],
          [ 0.0554,  0.1163, -0.3720,  0.1830, -0.0198]]],
        grad_fn=<StackBackward>),
 torch.Size([1, 2, 5]))
```



<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
seqlen
```

</div>




{:.output_data_text}

```
tensor([3, 1])
```



<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
def last_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1) - 1
    mask = row_vector == matrix
    mask.type(dtype)
    return mask
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
last_mask(seqlen, maxlen=hs.size(1))
```

</div>




{:.output_data_text}

```
tensor([[False, False,  True],
        [ True, False, False]])
```



<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
mask = last_mask(seqlen, maxlen=hs.size(1))
print(mask, mask.shape)
mask = mask.unsqueeze(-1).repeat(1, 1, 5)
mask
```

</div>

{:.output_stream}

```
tensor([[False, False,  True],
        [ True, False, False]]) torch.Size([2, 3])

```




{:.output_data_text}

```
tensor([[[False, False, False, False, False],
         [False, False, False, False, False],
         [ True,  True,  True,  True,  True]],

        [[ True,  True,  True,  True,  True],
         [False, False, False, False, False],
         [False, False, False, False, False]]])
```



<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
hs[mask].view(B, H)
```

</div>




{:.output_data_text}

```
tensor([[-0.4342,  0.1079, -0.8891, -0.3130, -0.0115],
        [-0.1040, -0.0475,  0.1698,  0.2809, -0.0018]], grad_fn=<ViewBackward>)
```



## pack padded sequence

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
seqlen, srtidx = seqlen.sort(dim=0, descending=True)
seqlen, srtidx
```

</div>




{:.output_data_text}

```
(tensor([3, 1]), tensor([0, 1]))
```



<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
seq = seq[srtidx]  # sorted sequence by lengths
seq, seqlen, E(seq)
```

</div>




{:.output_data_text}

```
(tensor([[38, 20,  4],
         [81,  0,  0]]),
 tensor([3, 1]),
 tensor([[[-1.0428,  0.3482, -0.7970,  2.0799,  0.4424, -0.4906, -0.9641,
           -0.3418,  0.1431,  1.3199],
          [ 0.3562,  1.6789, -0.2793,  0.7940,  0.0737,  2.5160,  0.6764,
           -0.4941, -1.2256,  0.7217],
          [-1.7787,  0.1457,  0.0709, -0.9375,  1.0674, -2.1660, -1.1485,
            0.2073, -0.6490,  1.1289]],
 
         [[ 1.2035,  0.6681,  1.2682, -0.3075, -0.9982,  0.2604,  1.4038,
           -0.9384, -0.8634,  1.0134],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000]]], grad_fn=<EmbeddingBackward>))
```



<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
print(pack_padded_sequence(seq, lengths=seqlen, batch_first=True))  # pack word label (conceptually show)
Eseq = pack_padded_sequence(E(seq), lengths=seqlen, batch_first=True)  # pack word embedding (use like this)
Eseq
```

</div>

{:.output_stream}

```
PackedSequence(data=tensor([38, 81, 20,  4]), batch_sizes=tensor([2, 1, 1]), sorted_indices=None, unsorted_indices=None)

```




{:.output_data_text}

```
PackedSequence(data=tensor([[-1.0428,  0.3482, -0.7970,  2.0799,  0.4424, -0.4906, -0.9641, -0.3418,
          0.1431,  1.3199],
        [ 1.2035,  0.6681,  1.2682, -0.3075, -0.9982,  0.2604,  1.4038, -0.9384,
         -0.8634,  1.0134],
        [ 0.3562,  1.6789, -0.2793,  0.7940,  0.0737,  2.5160,  0.6764, -0.4941,
         -1.2256,  0.7217],
        [-1.7787,  0.1457,  0.0709, -0.9375,  1.0674, -2.1660, -1.1485,  0.2073,
         -0.6490,  1.1289]], grad_fn=<PackPaddedSequenceBackward>), batch_sizes=tensor([2, 1, 1]), sorted_indices=None, unsorted_indices=None)
```



<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
hs, h = GRU(Eseq)
hs, h
```

</div>




{:.output_data_text}

```
(PackedSequence(data=tensor([[ 0.1499,  0.0979, -0.4934, -0.5707, -0.3808],
         [-0.1040, -0.0475,  0.1698,  0.2809, -0.0018],
         [ 0.2610,  0.1976,  0.2972, -0.0386, -0.3643],
         [-0.4342,  0.1079, -0.8891, -0.3130, -0.0115]], grad_fn=<CatBackward>), batch_sizes=tensor([2, 1, 1]), sorted_indices=None, unsorted_indices=None),
 tensor([[[-0.4342,  0.1079, -0.8891, -0.3130, -0.0115],
          [-0.1040, -0.0475,  0.1698,  0.2809, -0.0018]]],
        grad_fn=<StackBackward>))
```



<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
hs, blen = pad_packed_sequence(hs, batch_first=True)
h = h.transpose(0, 1)  # batch first 
hs, blen
```

</div>




{:.output_data_text}

```
(tensor([[[ 0.1499,  0.0979, -0.4934, -0.5707, -0.3808],
          [ 0.2610,  0.1976,  0.2972, -0.0386, -0.3643],
          [-0.4342,  0.1079, -0.8891, -0.3130, -0.0115]],
 
         [[-0.1040, -0.0475,  0.1698,  0.2809, -0.0018],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],
        grad_fn=<TransposeBackward0>),
 tensor([3, 1]))
```



### pad packed sequence (restore the sorting)

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
_, invidx = torch.sort(srtidx, 0, descending=False)
invidx
```

</div>




{:.output_data_text}

```
tensor([0, 1])
```



<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
hs = hs[invidx]
h = h[invidx]
seqlen = seqlen[invidx]
hs, h, seqlen
```

</div>




{:.output_data_text}

```
(tensor([[[ 0.1499,  0.0979, -0.4934, -0.5707, -0.3808],
          [ 0.2610,  0.1976,  0.2972, -0.0386, -0.3643],
          [-0.4342,  0.1079, -0.8891, -0.3130, -0.0115]],
 
         [[-0.1040, -0.0475,  0.1698,  0.2809, -0.0018],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],
        grad_fn=<IndexBackward>),
 tensor([[[-0.4342,  0.1079, -0.8891, -0.3130, -0.0115]],
 
         [[-0.1040, -0.0475,  0.1698,  0.2809, -0.0018]]],
        grad_fn=<IndexBackward>),
 tensor([3, 1]))
```



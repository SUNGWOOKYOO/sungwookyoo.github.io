---
title: "The Difference between Tensorflow and Pytorch using Embedding"
excerpt: "Compare Tensorflow and Pytorch when using Embedding."
categories:
 - tips
tags:
 - pytorch
 - tensorflow
use_math: true
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
last_modified_at: 2999-12-31 23:59
---
# Tensorflow vs Pytorch

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys, os, random
import numpy as np
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
import tensorflow as tf
import torch
msg = "tensorflow: {}, torch: {}"
print(msg.format(tf.__version__, torch.__version__))
```

</div>

{:.output_stream}

```
tensorflow: 2.0.0, torch: 0.4.1

```

there is no way to do this in pytorch. However, PyTorch doesn’t pre-occupy the GPU’s entire memory, so if your computation only uses 50% of GPU, only that much is locked by PyTorch

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
cpus = tf.config.experimental.list_physical_devices('CPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
# # GPU 메모리 제한하기
MEMORY_LIMIT_CONFIG = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
tf.config.experimental.set_virtual_device_configuration(gpus[0], MEMORY_LIMIT_CONFIG)
msg = "limit option: {}"
print(msg.format(MEMORY_LIMIT_CONFIG))
```

</div>

{:.output_stream}

```
limit option: [VirtualDeviceConfiguration(memory_limit=512)]

```

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# # only use CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
```

</div>

# Generate Dataset

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
V = 1000 # voca sizs
B, D, T, H = 2, 3, 5, 2
```

</div>

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
x = np.random.randint(0, 1000, size=(B, T), dtype=int)
# x_len = np.random.randint(0, T + 1, size=(B, ), dtype=int) # This will cause Error!!
x_len = np.random.randint(1, T + 1, size=(B, ), dtype=int)
for i in range(len(x)):
    x[i][x_len[i]:] = 0
mask = x!=0
msg = "x:\n{}\nx_len:\n{}\nmask:\n{}"
print(msg.format(x, x_len, mask))
```

</div>

{:.output_stream}

```
x:
[[359 595 629   0   0]
 [632 315 194 190   0]]
x_len:
[3 4]
mask:
[[ True  True  True False False]
 [ True  True  True  True False]]

```

## Encodeing: Embedding, LSTM

### 1. tensorflow

if `tf.test.is_gpu_available()` is executed, all gpu memories can be pre-occupied.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
# tf.test.is_gpu_available()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
import tensorflow.keras.layers as L
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
# convert to tensor
inp = tf.convert_to_tensor(x, dtype=tf.int32)
inp_len  = tf.convert_to_tensor(x_len, dtype=tf.int32)
mask = tf.convert_to_tensor(mask, dtype=tf.bool)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
inp, inp_len
```

</div>




{:.output_data_text}

```
(<tf.Tensor: id=0, shape=(2, 5), dtype=int32, numpy=
 array([[359, 595, 629,   0,   0],
        [632, 315, 194, 190,   0]], dtype=int32)>,
 <tf.Tensor: id=1, shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>)
```



<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
# embed = L.Embedding(V, D, mask_zero=True)
embed = L.Embedding(V, D)
lstm = L.LSTM(units=H, return_sequences=True, return_state=True)
blstm = L.Bidirectional(layer=lstm, merge_mode=None)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
embed(inp)
```

</div>




{:.output_data_text}

```
<tf.Tensor: id=17, shape=(2, 5, 3), dtype=float32, numpy=
array([[[ 0.00159524,  0.03665601, -0.01191108],
        [ 0.04492947,  0.01227681, -0.00458068],
        [ 0.03699413, -0.0307992 , -0.00333709],
        [-0.00643746,  0.0498703 , -0.04670119],
        [-0.00643746,  0.0498703 , -0.04670119]],

       [[-0.03196247, -0.04721764,  0.02672726],
        [ 0.04321711, -0.04162552,  0.03441907],
        [ 0.0413607 ,  0.03376241,  0.0028444 ],
        [-0.04561653, -0.03750287, -0.04467992],
        [-0.00643746,  0.0498703 , -0.04670119]]], dtype=float32)>
```



<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
#if mask_zero==True, mask values can be compute using embedding methods.
print(embed.compute_mask(inp)) 
print(embed(inp)._keras_mask) # another way.
```

</div>

{:.output_stream}

```
None
None

```

**In Tensorflow** ...

<font color=red> Please Note that :</font> Error can occurs if <mark>all sequence values are zeros in an example.</mark> Cudnn does not precess this when lstm module is used.  
The error message can be shown as follows.

<font color=red>UnknownError:</font> CUDNN_STATUS_BAD_PARAM
in tensorflow/stream_executor/cuda/cuda_dnn.cc(1424): 'cudnnSetRNNDataDescriptor( data_desc.get(), data_type, layout, max_seq_length, batch_size, data_size, seq_lengths_array, (void*)&padding_fill)' [Op:CudnnRNNV3]



<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
lstm(embed(inp)) # [h, ht, ct], automatically applied if embed.mask_zero=True.
lstm(embed(inp), mask=mask) # manully plug-in mask values.
```

</div>




{:.output_data_text}

```
[<tf.Tensor: id=294, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[-0.00712024, -0.00011674],
         [-0.01264691, -0.00656443],
         [-0.00940342, -0.01202935],
         [ 0.        ,  0.        ],
         [ 0.        ,  0.        ]],
 
        [[ 0.01308669,  0.00558831],
         [ 0.01686362,  0.00162064],
         [ 0.00492537, -0.00131158],
         [ 0.00957621, -0.00241367],
         [ 0.        ,  0.        ]]], dtype=float32)>,
 <tf.Tensor: id=298, shape=(2, 2), dtype=float32, numpy=
 array([[-0.00940342, -0.01202935],
        [ 0.00957621, -0.00241367]], dtype=float32)>,
 <tf.Tensor: id=302, shape=(2, 2), dtype=float32, numpy=
 array([[-0.01855913, -0.02394593],
        [ 0.01943874, -0.00485599]], dtype=float32)>]
```



<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
init_states = [tf.random.normal(shape=[B, H])] * 4 # [ht_fw, ht_bw, ct_fw, bt_bw]
blstm(embed(inp), mask=mask, initial_state=init_states) 
blstm(embed(inp), mask=mask) # outputs # [hf, hb, htf, htb, ctf, ctb]
```

</div>




{:.output_data_text}

```
[<tf.Tensor: id=762, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[ 0.00740621, -0.00114337],
         [ 0.00714395, -0.0095588 ],
         [-0.00048524, -0.0167635 ],
         [ 0.        ,  0.        ],
         [ 0.        ,  0.        ]],
 
        [[-0.01039943,  0.00912978],
         [-0.02368858,  0.00449102],
         [-0.01789851, -0.00223974],
         [-0.01275576, -0.00316449],
         [ 0.        ,  0.        ]]], dtype=float32)>,
 <tf.Tensor: id=903, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[-2.6652839e-03, -4.0566991e-03],
         [ 3.7362654e-04, -3.2365608e-03],
         [ 1.9953572e-03,  2.9448469e-05],
         [ 0.0000000e+00,  0.0000000e+00],
         [ 0.0000000e+00,  0.0000000e+00]],
 
        [[ 4.8977546e-03,  2.0752738e-03],
         [ 3.0945316e-03, -1.2807426e-03],
         [ 5.9818052e-04,  2.5921396e-03],
         [ 2.9544432e-03,  9.7605204e-03],
         [ 0.0000000e+00,  0.0000000e+00]]], dtype=float32)>,
 <tf.Tensor: id=766, shape=(2, 2), dtype=float32, numpy=
 array([[-0.00048524, -0.0167635 ],
        [-0.01275576, -0.00316449]], dtype=float32)>,
 <tf.Tensor: id=770, shape=(2, 2), dtype=float32, numpy=
 array([[-0.00095634, -0.03310474],
        [-0.02573843, -0.00626021]], dtype=float32)>,
 <tf.Tensor: id=896, shape=(2, 2), dtype=float32, numpy=
 array([[-0.00266528, -0.0040567 ],
        [ 0.00489775,  0.00207527]], dtype=float32)>,
 <tf.Tensor: id=900, shape=(2, 2), dtype=float32, numpy=
 array([[-0.00526954, -0.00818289],
        [ 0.0100137 ,  0.00411511]], dtype=float32)>]
```



### 2. pytorch

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
torch.cuda.is_available()
```

</div>




{:.output_data_text}

```
True
```



<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
```

</div>

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
# conver to torch.Tensor
inp = torch.LongTensor(x)
inp_len = torch.LongTensor(x_len)
inp = inp.cuda()
inp_len = inp_len.cuda()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
inp, inp_len
```

</div>




{:.output_data_text}

```
(tensor([[359, 595, 629,   0,   0],
         [632, 315, 194, 190,   0]], device='cuda:0'),
 tensor([3, 4], device='cuda:0'))
```



<div class="prompt input_prompt">
In&nbsp;[21]:
</div>

<div class="input_area" markdown="1">

```python
embed = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0).cuda()
lstm = nn.LSTM(input_size=D, hidden_size=H, num_layers=1, batch_first=True).cuda()
blstm = nn.LSTM(input_size=D, hidden_size=H, num_layers=1, batch_first=True, bidirectional=True).cuda()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
embed(inp)
```

</div>




{:.output_data_text}

```
tensor([[[ 1.6964, -0.9813,  0.0643],
         [ 0.6293,  0.6831, -0.2784],
         [ 1.6584,  0.6596, -0.1362],
         [ 0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000]],

        [[ 0.3431,  1.0262, -0.9524],
         [-0.5076, -1.7151, -0.5414],
         [ 0.1196, -0.4846,  1.1399],
         [ 0.9370, -1.1598, -1.0287],
         [ 0.0000,  0.0000,  0.0000]]],
       device='cuda:0', grad_fn=<EmbeddingBackward>)
```



<div class="prompt input_prompt">
In&nbsp;[23]:
</div>

<div class="input_area" markdown="1">

```python
# defaults initial states are all zeros.
# h0 = torch.randn(1*1, B, H) # shape: (num_layers * num_directions, batch, hidden_size)
# c0 = torch.randn(1*2, B, H)
# inp, (h0, c0) can be a input
lstm(embed(inp)) # outputs (h, (ht, ct))
```

</div>




{:.output_data_text}

```
(tensor([[[ 0.0688, -0.2996],
          [ 0.0742, -0.2835],
          [ 0.1083, -0.2933],
          [ 0.0521, -0.2850],
          [ 0.0064, -0.2976]],
 
         [[ 0.0409, -0.2272],
          [-0.1513, -0.1366],
          [-0.0482, -0.2268],
          [-0.0260, -0.2581],
          [-0.0226, -0.2709]]], device='cuda:0', grad_fn=<CudnnRnnBackward>),
 (tensor([[[ 0.0064, -0.2976],
           [-0.0226, -0.2709]]], device='cuda:0', grad_fn=<CudnnRnnBackward>),
  tensor([[[ 0.0090, -0.4694],
           [-0.0318, -0.4152]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)))
```



<div class="prompt input_prompt">
In&nbsp;[24]:
</div>

<div class="input_area" markdown="1">

```python
h0 = torch.randn(1*2, B, H).cuda() # shape: (num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(1*2, B, H).cuda()
blstm(embed(inp), (h0, c0))
```

</div>




{:.output_data_text}

```
(tensor([[[-0.4782,  0.0351, -0.1746,  0.1735],
          [-0.5423, -0.1576, -0.1632,  0.1544],
          [-0.6151, -0.1166, -0.1237,  0.1812],
          [-0.5023, -0.3070, -0.0742,  0.0486],
          [-0.4957, -0.4150, -0.0708,  0.0136]],
 
         [[ 0.1661,  0.0499, -0.0780, -0.0444],
          [ 0.0195, -0.3287, -0.0032,  0.0009],
          [ 0.1570, -0.3652, -0.1026,  0.1766],
          [ 0.1661, -0.5768, -0.3078,  0.0862],
          [-0.0333, -0.5119, -0.2880,  0.0941]]],
        device='cuda:0', grad_fn=<CudnnRnnBackward>),
 (tensor([[[-0.4957, -0.4150],
           [-0.0333, -0.5119]],
  
          [[-0.1746,  0.1735],
           [-0.0780, -0.0444]]], device='cuda:0', grad_fn=<CudnnRnnBackward>),
  tensor([[[-1.2094, -0.6821],
           [-0.0644, -1.0096]],
  
          [[-0.4359,  0.7697],
           [-0.2197, -0.1102]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)))
```



pack, unpack techniques can be used easily in pytorch. [korean blog](https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html)

<div class="prompt input_prompt">
In&nbsp;[25]:
</div>

<div class="input_area" markdown="1">

```python
sorted_inp_len, indices = torch.sort(inp_len, dim=0, descending=True)
sorted_inp_len, indices
```

</div>




{:.output_data_text}

```
(tensor([4, 3], device='cuda:0'), tensor([1, 0], device='cuda:0'))
```



<div class="prompt input_prompt">
In&nbsp;[26]:
</div>

<div class="input_area" markdown="1">

```python
embed(inp)[indices]
```

</div>




{:.output_data_text}

```
tensor([[[ 0.3431,  1.0262, -0.9524],
         [-0.5076, -1.7151, -0.5414],
         [ 0.1196, -0.4846,  1.1399],
         [ 0.9370, -1.1598, -1.0287],
         [ 0.0000,  0.0000,  0.0000]],

        [[ 1.6964, -0.9813,  0.0643],
         [ 0.6293,  0.6831, -0.2784],
         [ 1.6584,  0.6596, -0.1362],
         [ 0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000]]],
       device='cuda:0', grad_fn=<TakeBackward>)
```



if a seqeunce of an example with all zeros vectors causes `Error`.  
the message is shown as follows.  
<font color=red>ValueError</font>: Length of all samples has to be greater than 0, but found an element in 'lengths' that is <= 0

<div class="prompt input_prompt">
In&nbsp;[27]:
</div>

<div class="input_area" markdown="1">

```python
packed_embeddings = pack_padded_sequence(embed(inp)[indices], sorted_inp_len.data.tolist(), batch_first=True)
packed_embeddings
```

</div>




{:.output_data_text}

```
PackedSequence(data=tensor([[ 0.3431,  1.0262, -0.9524],
        [ 1.6964, -0.9813,  0.0643],
        [-0.5076, -1.7151, -0.5414],
        [ 0.6293,  0.6831, -0.2784],
        [ 0.1196, -0.4846,  1.1399],
        [ 1.6584,  0.6596, -0.1362],
        [ 0.9370, -1.1598, -1.0287]],
       device='cuda:0', grad_fn=<PackPaddedBackward>), batch_sizes=tensor([2, 2, 2, 1], grad_fn=<PackPaddedBackward>))
```



<div class="prompt input_prompt">
In&nbsp;[28]:
</div>

<div class="input_area" markdown="1">

```python
packed_h, (packed_hn, packed_cn) = blstm(packed_embeddings) # outputs packed results.
packed_h, (packed_hn, packed_cn)
```

</div>




{:.output_data_text}

```
(PackedSequence(data=tensor([[-0.3480, -0.1531,  0.0350, -0.0619],
         [ 0.2600, -0.2039, -0.1305,  0.1756],
         [-0.2834, -0.5071,  0.2104, -0.0072],
         [-0.1044, -0.1516, -0.1276,  0.1511],
         [-0.1307, -0.5428,  0.0541,  0.1854],
         [-0.2404, -0.0354, -0.0786,  0.1734],
         [-0.1927, -0.6621, -0.0166,  0.0799]],
        device='cuda:0', grad_fn=<CudnnRnnBackward>), batch_sizes=tensor([2, 2, 2, 1], grad_fn=<PackPaddedBackward>)),
 (tensor([[[-0.1927, -0.6621],
           [-0.2404, -0.0354]],
  
          [[ 0.0350, -0.0619],
           [-0.1305,  0.1756]]], device='cuda:0', grad_fn=<CudnnRnnBackward>),
  tensor([[[-0.2740, -1.2842],
           [-0.2947, -0.0455]],
  
          [[ 0.0900, -0.1426],
           [-0.3127,  0.7660]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)))
```



<div class="prompt input_prompt">
In&nbsp;[43]:
</div>

<div class="input_area" markdown="1">

```python
packed_hn.size()
```

</div>




{:.output_data_text}

```
torch.Size([2, 2, 2])
```



<div class="prompt input_prompt">
In&nbsp;[29]:
</div>

<div class="input_area" markdown="1">

```python
inp[indices]
```

</div>




{:.output_data_text}

```
tensor([[632, 315, 194, 190,   0],
        [359, 595, 629,   0,   0]], device='cuda:0')
```



<div class="prompt input_prompt">
In&nbsp;[30]:
</div>

<div class="input_area" markdown="1">

```python
pad_packed_sequence(packed_h, batch_first=True) # unpack the result.
```

</div>




{:.output_data_text}

```
(tensor([[[-0.3480, -0.1531,  0.0350, -0.0619],
          [-0.2834, -0.5071,  0.2104, -0.0072],
          [-0.1307, -0.5428,  0.0541,  0.1854],
          [-0.1927, -0.6621, -0.0166,  0.0799]],
 
         [[ 0.2600, -0.2039, -0.1305,  0.1756],
          [-0.1044, -0.1516, -0.1276,  0.1511],
          [-0.2404, -0.0354, -0.0786,  0.1734],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]],
        device='cuda:0', grad_fn=<TransposeBackward0>), tensor([4, 3]))
```



* Finial hidden states of Bi-LSTM:  
    * [`last forward vector`, `last backward vector = first vector of output`]: `[B, 2, H]`

<div class="prompt input_prompt">
In&nbsp;[44]:
</div>

<div class="input_area" markdown="1">

```python
packed_hn
```

</div>




{:.output_data_text}

```
tensor([[[-0.1927, -0.6621],
         [-0.2404, -0.0354]],

        [[ 0.0350, -0.0619],
         [-0.1305,  0.1756]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)
```



### Big-Difference tensorflow vs pytorch
1. **Embedding**  
    **In Tensorflow**, even though `mask_zero=True`, the outputs of embedding layer for `padding id=0` does not zero-vector.  
    <font color=red>On the other hand</font>, 
    **In Pytorch**, embedding layer's signiture `padding_idx` can determine outputs to become zero-vector.
    
2. **LSTM**   
    * **In Tensorflow**, if a seqeunce of an example with all zeros vectors causes `Error` in GPU commputing.  
      <font color=red>On the other hand</font>, **In Pytorch**, a seqeunce with all zeros vectors does not cause `Error` in GPU commputing.
    * Automatical masked outputs in LSTM can be supplied in Tensorflow, but pytorch does not have this.
    * Pytorch supplies packed, unpacked technique for efficient computation when treating LSTM sequences.  
      <font color=red>However</font>, in this technique, if a seqeunce of an example with all zeros vectors causes `Error`
      
   <font color=skyblue> Solution</font>: To prevent `Error` related to all zero vectors, all values of input_len should be larger than 0 
    
    

Create class `Embedding` in Tensorflow which operates 

<div class="prompt input_prompt">
In&nbsp;[31]:
</div>

<div class="input_area" markdown="1">

```python
class Embedding(tf.keras.layers.Layer):
  
    def __init__(self, input_dim, output_dim, padding_idx=0, **kwargs):
        """ default padding_idx=0.
        
        Call Args:
            inputs: [B, T]
        
        description:
            input_dim: V (vocabulary size)
            output_dim: D 
        """
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.padding_idx = padding_idx

    def build(self, input_shape):
        self.embeddings = self.add_weight(
          shape=(self.input_dim, self.output_dim),
          initializer='random_normal',
          dtype='float32')

    def call(self, inputs): 
        def compute_mask():
            return tf.not_equal(inputs, self.padding_idx)
        
        out = tf.nn.embedding_lookup(self.embeddings, inputs)
        masking = compute_mask() # [B, T], bool
        masking = tf.cast(tf.tile(masking[:,:, tf.newaxis], [1,1,self.output_dim]), 
                          dtype=tf.float32) # [B, T, D]
        return tf.multiply(out, masking)
  
```

</div>

<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
embed = Embedding(V, D, padding_idx=0)
```

</div>

#### regenerate dataset

<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
x = np.random.randint(0, 1000, size=(B, T), dtype=int)
# x_len = np.random.randint(0, T + 1, size=(B, ), dtype=int) # This will cause Error!!
x_len = np.random.randint(1, T + 1, size=(B, ), dtype=int)
for i in range(len(x)):
    x[i][x_len[i]:] = 0
mask = x!=0
msg = "x:\n{}\nx_len:\n{}\nmask:\n{}"
print(msg.format(x, x_len, mask))
```

</div>

{:.output_stream}

```
x:
[[965 176 491 801 149]
 [538 162 287 297 610]]
x_len:
[5 5]
mask:
[[ True  True  True  True  True]
 [ True  True  True  True  True]]

```

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
# convert to tensor
inp = tf.convert_to_tensor(x, dtype=tf.int32)
inp_len  = tf.convert_to_tensor(x_len, dtype=tf.int32)
mask = tf.convert_to_tensor(mask, dtype=tf.bool)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[35]:
</div>

<div class="input_area" markdown="1">

```python
inp, mask
```

</div>




{:.output_data_text}

```
(<tf.Tensor: id=904, shape=(2, 5), dtype=int32, numpy=
 array([[965, 176, 491, 801, 149],
        [538, 162, 287, 297, 610]], dtype=int32)>,
 <tf.Tensor: id=906, shape=(2, 5), dtype=bool, numpy=
 array([[ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True]])>)
```



<div class="prompt input_prompt">
In&nbsp;[36]:
</div>

<div class="input_area" markdown="1">

```python
# test_mask = np.array([[True, False, False, False, False],
#         [ True,  True,  True, False, False]])
# test_mask = tf.convert_to_tensor(test_mask)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[37]:
</div>

<div class="input_area" markdown="1">

```python
embed(inp)
```

</div>




{:.output_data_text}

```
<tf.Tensor: id=930, shape=(2, 5, 3), dtype=float32, numpy=
array([[[ 0.0416598 ,  0.0018823 ,  0.06277428],
        [ 0.02224987,  0.00704473,  0.02094294],
        [-0.02784282, -0.06249573,  0.00104964],
        [ 0.09029059,  0.06687873,  0.08003117],
        [-0.01774345,  0.0964142 , -0.12424164]],

       [[-0.0317932 , -0.08884034,  0.02873092],
        [-0.10553911, -0.03988774, -0.0694226 ],
        [ 0.0517051 ,  0.01909538,  0.00886771],
        [-0.04648464,  0.09031732, -0.01851485],
        [ 0.00429158, -0.02031239,  0.02710586]]], dtype=float32)>
```



<div class="prompt input_prompt">
In&nbsp;[38]:
</div>

<div class="input_area" markdown="1">

```python
lstm = L.LSTM(units=H, return_sequences=True, return_state=True)
blstm = L.Bidirectional(layer=lstm, merge_mode=None)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[39]:
</div>

<div class="input_area" markdown="1">

```python
lstm(embed(inp), mask=mask) #  [h, ht, ct]
```

</div>




{:.output_data_text}

```
[<tf.Tensor: id=1112, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[-0.01064905,  0.00394847],
         [-0.01063567,  0.00473839],
         [-0.01129253, -0.00097121],
         [-0.01781004,  0.00965991],
         [ 0.01671795,  0.00570279]],
 
        [[-0.01029179, -0.00404594],
         [ 0.00095257, -0.00189471],
         [ 0.00074197, -0.00608036],
         [ 0.00769171,  0.01725223],
         [ 0.00087247,  0.01191795]]], dtype=float32)>,
 <tf.Tensor: id=1116, shape=(2, 2), dtype=float32, numpy=
 array([[0.01671795, 0.00570279],
        [0.00087247, 0.01191795]], dtype=float32)>,
 <tf.Tensor: id=1120, shape=(2, 2), dtype=float32, numpy=
 array([[0.03161679, 0.01112464],
        [0.00175713, 0.02399201]], dtype=float32)>]
```



<div class="prompt input_prompt">
In&nbsp;[40]:
</div>

<div class="input_area" markdown="1">

```python
blstm(embed(inp), mask=mask)
```

</div>




{:.output_data_text}

```
[<tf.Tensor: id=1356, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[-0.00511874,  0.00858796],
         [-0.00808002,  0.01097147],
         [ 0.00567391,  0.00636496],
         [-0.01547743,  0.02149273],
         [-0.02562061,  0.00905941]],
 
        [[ 0.01695154, -0.00168698],
         [ 0.03395947, -0.01169712],
         [ 0.01980755, -0.0082969 ],
         [ 0.0138903 , -0.00401598],
         [ 0.01490065, -0.00126571]]], dtype=float32)>,
 <tf.Tensor: id=1497, shape=(2, 5, 2), dtype=float32, numpy=
 array([[[ 0.013806  ,  0.0142377 ],
         [ 0.00849442,  0.00643777],
         [ 0.00585771,  0.00317722],
         [ 0.02268172, -0.0008881 ],
         [ 0.00189724, -0.01832626]],
 
        [[-0.02280359, -0.00843212],
         [-0.01733721, -0.01760674],
         [ 0.00902378, -0.00283928],
         [-0.00303076, -0.01425412],
         [-0.00029973,  0.00394505]]], dtype=float32)>,
 <tf.Tensor: id=1360, shape=(2, 2), dtype=float32, numpy=
 array([[-0.02562061,  0.00905941],
        [ 0.01490065, -0.00126571]], dtype=float32)>,
 <tf.Tensor: id=1364, shape=(2, 2), dtype=float32, numpy=
 array([[-0.05175059,  0.01903152],
        [ 0.02978283, -0.00251175]], dtype=float32)>,
 <tf.Tensor: id=1490, shape=(2, 2), dtype=float32, numpy=
 array([[ 0.013806  ,  0.0142377 ],
        [-0.02280359, -0.00843212]], dtype=float32)>,
 <tf.Tensor: id=1494, shape=(2, 2), dtype=float32, numpy=
 array([[ 0.02729905,  0.02839736],
        [-0.04494345, -0.01696696]], dtype=float32)>]
```



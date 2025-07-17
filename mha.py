
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import jax 
import jax.numpy as jnp
import flax
from jax import random
from flax import linen as nnj 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MHA_pytorch(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.Wo = nn.Linear(d_model, d_model)
        self.Wqkv = nn.Linear(d_model, 3 * d_model)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        q_seq_len = q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)
        q, k, v = self.Wqkv(q).chunk(3, dim = -1)
        Q = q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = k.view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = v.view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim = -1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        output = self.Wo(attention_output)
        return output

def scaled_dot_product(q, k, v):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = nnj.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values 

class MHA_jax(nnj.Module):
    d_model : int
    num_heads : int
    
    def setup(self):
        self.Wqkv = nnj.Dense(3 * self.d_model)
        self.Wo = nnj.Dense(self.d_model)
    
    def __call__(self, x):
        batch_size, seq_length, d_model = x.shape
        qkv = self.Wqkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) 
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        values = scaled_dot_product(q, k, v)
        values = values.transpose(0, 2, 1, 3) 
        values = values.reshape(batch_size, seq_length, d_model)
        o = self.Wo(values)
        return o 

d_model = 512
num_heads = 8
batch_size = 32
seq_len = 512

attention_pytorch = MHA_pytorch(d_model, num_heads).to(device)
x_pytorch = torch.rand(batch_size, seq_len, d_model).to(device)

pytorch_times = []
with torch.no_grad():
    for _ in range(100):
        start_time = time.time()
        output_pytorch = attention_pytorch(x_pytorch, x_pytorch, x_pytorch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        pytorch_times.append((end_time - start_time) * 1e6)
print("Avg PyTorch time (microseconds):", sum(pytorch_times) / len(pytorch_times))

attention_jax = MHA_jax(d_model, num_heads)
key = random.PRNGKey(0)
x_jax = jnp.ones((batch_size, seq_len, d_model))
params = attention_jax.init(key, x_jax)

@jax.jit
def forward_jax(params, x):
    return attention_jax.apply(params, x)

forward_jax(params, x_jax).block_until_ready()

jax_times = []
for _ in range(100):
    start_time = time.time()
    output_jax = forward_jax(params, x_jax)
    output_jax.block_until_ready()
    end_time = time.time()
    jax_times.append((end_time - start_time) * 1e6)
print("Avg JAX time (microseconds):", sum(jax_times) / len(jax_times))

import os

tokens_path = os.path.expanduser('~/wiki.train.tokens')

with open(tokens_path, 'r', encoding='utf-8') as src:
    data = src.read()
    print(len(data))

with open('train.txt', 'w', encoding='utf-8') as dst:
    dst.write(data)


import json
import glob
import os


best_loss = float('inf')
save_dir = "model_checkpoints"
os.makedirs(save_dir, exist_ok=True)

def save_model(model, optimizer, scheduler, epoch, step, loss, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'model_config': {
            'num_heads': model.num_heads,
            'd_model': model.d_model,
            'd_ff': model.d_ff,
            'vocab_size': model.vocab_size,
            'num_layers': model.num_layers,
            'max_seq_len': model.max_seq_len
        }
    }


    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"New best model saved! Loss: {loss:.4f}")

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import os
import tiktoken

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda")

class Dataset:
   def __init__(self, file_path , seq_len):
       self.seq_len = seq_len
       with open(file_path, 'r', encoding='utf-8') as f:
           text = f.read()

       self.encoder = tiktoken.get_encoding("gpt2")
       self.tokens = self.encoder.encode(text)
       self.vocab_size = self.encoder.n_vocab

   def decode(self, tokens):
       return self.encoder.decode(tokens)
   def encode(self, text):
       return self.encoder.encode(text)

   def get_batch(self, batch_size):
       starts = torch.randint(0, len(self.tokens) - self.seq_len - 1, (batch_size,))
       x = torch.stack([torch.tensor(self.tokens[start:start + self.seq_len]) for start in starts]).to(device)
       y = torch.stack([torch.tensor(self.tokens[start + 1:start + self.seq_len + 1]) for start in starts]).to(device)
       return x, y

def create_future_mask(seq_len):
   return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

class MultiHeadAttention(nn.Module):
   def __init__(self, d_model, num_heads):
       super().__init__()
       self.d_model = d_model
       self.num_heads = num_heads
       self.d_k = self.d_model // self.num_heads
       self.Wq = nn.Linear(d_model, d_model)
       self.Wk = nn.Linear(d_model, d_model)
       self.Wv = nn.Linear(d_model, d_model)
       self.Wo = nn.Linear(d_model, d_model)

   def forward(self, q, k, v, mask = None):
       batch_size = q.size(0)
       q_seq_len = q.size(1)
       k_seq_len = k.size(1)
       v_seq_len = v.size(1)

       Q = self.Wq(q).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
       K = self.Wk(k).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
       V = self.Wv(v).view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)

       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
       if mask is not None:
           scores = scores.masked_fill(mask==0, -1e9)
       attention_weights = F.softmax(scores, dim = -1)

       attention_output = torch.matmul(attention_weights, V)

       attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
       output = self.Wo(attention_output)
       return output

class FF(nn.Module):
   def __init__(self, d_ff, d_model):
       super().__init__()
       self.d_ff = d_ff
       self.l1 = nn.Linear(d_model, d_ff)
       self.l2 = nn.Linear(d_ff, d_model)

   def forward(self, x):
       return self.l2(F.relu(self.l1(x)))

class Encoder(nn.Module):
   def __init__(self, num_heads, d_model, d_ff):
       super().__init__()
       self.num_heads = num_heads
       self.d_model = d_model
       self.d_ff = d_ff

       self.attention = MultiHeadAttention(d_model, num_heads)
       self.ff = FF(d_ff, d_model)
       self.norm1 = nn.LayerNorm(d_model)
       self.norm2 = nn.LayerNorm(d_model)

   def forward(self, x):
       attn_output = self.attention(x, x, x)
       x = self.norm1(x + attn_output)
       ff_out = self.ff(x)
       x = self.norm2(x + ff_out)
       return x

class Decoder(nn.Module):
   def __init__(self, num_heads, d_model, d_ff):
       super().__init__()
       self.num_heads = num_heads
       self.d_model = d_model
       self.d_ff = d_ff

       self.attention = MultiHeadAttention(d_model, num_heads)
       self.cross_attention = MultiHeadAttention(d_model, num_heads)
       self.norm1 = nn.LayerNorm(d_model)
       self.norm2 = nn.LayerNorm(d_model)
       self.norm3 = nn.LayerNorm(d_model)
       self.ff = FF(d_ff, d_model)

   def forward(self, x, encoder_output):
       seq_len = x.size(1)
       mask = create_future_mask(seq_len)
       attn_output = self.attention(x, x, x, mask)
       x = self.norm1(x + attn_output)

       cross_attn_output = self.cross_attention(x, encoder_output, encoder_output)
       x = self.norm2(x + cross_attn_output)
       ff_out = self.ff(x)
       x = self.norm3(x + ff_out)
       return x

class PositionalEncoding(nn.Module):
   def __init__(self, d_model, max_len = 5000):
       super().__init__()
       self.d_model = d_model
       self.max_len = max_len

       pe = torch.zeros(max_len, d_model)
       position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)

       div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                          (-math.log(10000.0) / d_model))

       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)

       pe = pe.unsqueeze(0)
       self.register_buffer('pe', pe)
   def forward(self, x):
       return x + self.pe[:, :x.size(1)]

class GPTDecoder(nn.Module):
    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FF(d_ff, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        mask = create_future_mask(seq_len)

        # Self-attention with causal mask
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class Transformer(nn.Module):
   def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff):
       super().__init__()
       self.src_vocab_size = src_vocab_size
       self.tgt_vocab_size = tgt_vocab_size
       self.d_model = d_model
       self.num_heads = num_heads
       self.d_ff = d_ff
       self.embedding_src = nn.Embedding(src_vocab_size, d_model)
       self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)

       self.encoder = Encoder(num_heads, d_model, d_ff)
       self.decoder = Decoder(num_heads, d_model, d_ff)
       self.linear = nn.Linear(d_model, tgt_vocab_size)

       self.pe = PositionalEncoding(d_model)

   def forward(self, src_tokens, tgt_tokens):
       src_embeddings = self.pe(self.embedding_src(src_tokens) * math.sqrt(self.d_model))
       tgt_embeddings = self.pe(self.embedding_tgt(tgt_tokens) * math.sqrt(self.d_model))

       encoder_output = self.encoder(src_embeddings)
       decoder_output = self.decoder(tgt_embeddings, encoder_output)
       return self.linear(decoder_output)

class LLM(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, vocab_size, num_layers=6, max_seq_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)

        # Stack multiple decoder layers
        self.decoder_layers = nn.ModuleList([
            GPTDecoder(num_heads, d_model, d_ff)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)  # Final layer norm
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Token embeddings + positional encoding
        embeddings = self.pe(self.embeddings(x) * math.sqrt(self.d_model))

        # Pass through all decoder layers
        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states)

        # Final layer norm and projection to vocabulary
        hidden_states = self.layer_norm(hidden_states)
        return self.linear(hidden_states)
batch_size=8
seq_len=512

file_path = "data.txt"

dataset = Dataset(file_path, 512)
print(len(dataset.tokens))
model = LLM(num_heads = 12, d_model = 768, d_ff = 3072, vocab_size = dataset.vocab_size, max_seq_len=512, num_layers = 12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
num_epochs = 50
steps_per_epoch = len(dataset.tokens) // (batch_size * seq_len)

print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
print(f"Using device: {device}")


total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

model.train()

loss_list = []

for epoch in range(num_epochs):
   epoch_start_time = time.time()
   total_loss = 0.0
   for steps in range(steps_per_epoch):
       x, y = dataset.get_batch(batch_size)
       optimizer.zero_grad()

       logits = model(x)
       logits = logits.view(-1, dataset.vocab_size)
       targets = y.view(-1)

       loss = criterion(logits, targets)

       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       optimizer.step()
       scheduler.step()

       total_loss += loss.item()


       loss_list.append(loss.item())

       if steps % 100 == 0:
               print(f"Epoch {epoch+1}/{num_epochs}, Step {steps}/{steps_per_epoch}, Loss: {loss.item():.4f}")

   epoch_end_time = time.time()
   epoch_time = epoch_end_time - epoch_start_time
   avg_loss = total_loss / steps_per_epoch
   is_best = avg_loss < best_loss
   if is_best:
       best_loss = avg_loss
       save_model(model, optimizer, scheduler, epoch, steps, avg_loss, is_best)
   print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Time taken: {epoch_time:.2f} seconds")

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()

def generate_text(model, dataset, prompt="To be or not to be", max_length=200, temperature=0.8):
    model.eval()

    device = next(model.parameters()).device

    tokens = dataset.encode(prompt)
    generated = tokens.copy()


    with torch.no_grad():
        for _ in range(max_length):
            context = generated[-model.max_seq_len:]
            context_tensor = torch.tensor(context).unsqueeze(0).to(device)  # Move to device!

            logits = model(context_tensor)

            last_logits = logits[0, -1, :] / temperature

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            new_text = dataset.decode([next_token])
            print(new_text, end="")

            if next_token == dataset.encoder.eot_token:
                break

    print("\n" + "="*50)
    return dataset.decode(generated)

def test_generation():
    checkpoint_path = os.path.join(save_dir, 'best_model.pt')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['model_config']
        model = LLM(
            num_heads=config['num_heads'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len']
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        prompt = "The future of artificial intelligence"
        generated_text = generate_text(model, dataset, prompt, max_length=100)
        print(f"Generated text: {generated_text}")
    else:
        print("No best model found!")

test_generation()

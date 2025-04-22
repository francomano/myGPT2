# Mini GPT-2 (char-level) in PyTorch

Implementazione minimale di GPT-2 in PyTorch, per scopi didattici. Include attenzione causale, MLP e blocchi Transformer. Funziona a livello di caratteri.

## Esempio rapido

```python
import gpt2
import torch, string

# Setup
chars = string.ascii_lowercase + " .,!?\'"
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = gpt2.GPT2Config(vocab_size=len(stoi))
model = gpt2.GPT2(config)

# Test input
x = torch.tensor([encode("hello my nam")])
logits = model(x)

# Predizione prossimo char
next_id = torch.argmax(logits[0][0, -1]).item()
print("Next char:", itos[next_id])

Requisiti

Python 3
torch
File principali

gpt2.py: definizione del modello
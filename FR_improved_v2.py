# Code copied from Andrei Karpathy


# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline

# Load names and convert to lowercase
words = [word.lower() for word in open('german_names.txt', 'r', encoding="utf-8").read().splitlines()]
print("Sample of words:", words[:10])  # Print first 10 words for debugging
# print("Sample words after lowercasing:", words[:10])  # Debugging print
words[:8]

len(words)

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words)) | set("éèêëàâçôùû")))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
print(f"Number of characters in vocabulary: {len(stoi)}")

# First checking the number of characters until here before running the whole code:
#import sys
#sys.exit()

itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    # Inside build_dataset function
    for ch in (w + '.').lower():  # Ensure lowercase
      try:
        ix = stoi[ch]
      except KeyError:
        pass  # Ignore characters not in stoi
      
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]  # Crop and append


  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random

# Step 1: Load the names from the file
with open("german_names.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()  # Read lines and remove newline characters

# Step 2: Split names that contain a "/"
cleaned_names = []
for name in words:
    if "/" in name:
        cleaned_names.extend(name.split("/"))  # Split at "/" and add both names separately
    else:
        cleaned_names.append(name)  # Keep names without "/"

# Step 3: Shuffle the names to avoid ordering bias
random.seed(42)
random.shuffle(cleaned_names)

# Step 4: Save cleaned names to a new file
with open("cleaned_german_names_in_french_model.txt", "w", encoding="utf-8") as f:
    for name in cleaned_names:
        f.write(name + "\n")

# print("Sample of shuffled names:", cleaned_names[:10])
print(f"Total names after cleaning: {len(cleaned_names)}")

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((len(itos), 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, len(itos)), generator=g)
b2 = torch.randn(len(itos), generator=g)
parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters) # number of parameters in total

for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(200000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 2)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  #lr = lrs[i]
  lr = 0.05 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

#print(loss.item())

plt.plot(stepi, lossi)

# training loss
emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
loss

# validation loss
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
loss



# prompt:  Tune the hyperparameters of the model to beat A. Karpathy's validation loss in 15

# further training with a smaller learning rate
for i in range(100000):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 2)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.001 # significantly smaller learning rate
  for p in parameters:
    p.data += -lr * p.grad

  # track stats (optional, but useful for monitoring)
  if i % 1000 == 0: # print loss every 1000 steps
    print(f"Step: {i}, Loss: {loss.item()}")

# validation loss after further training
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f"Validation Loss after further training: {loss.item()}")

# test loss
emb = C[Xte] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Yte)
loss

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

# Save generated names to a text file
with open("generated_german_names_in_french_model.txt", "w", encoding="utf-8") as f:
    for _ in range(20):
        out = []
        context = [0] * block_size  # initialize with all ...
        while True:
            emb = C[torch.tensor([context])]  # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        name = ''.join(itos[i] for i in out)
        print(name)  # Print to console for debugging
        f.write(name + "\n")  # Write to file


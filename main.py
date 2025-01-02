#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F 

learning_rate = 1e-3
batch_size = 64
block_size = 256
embed_size = 384
device = "cuda"
max_iters = 5000
n_heads,n_layers = 6,6

#shakespeare text
with open("./input.txt","r") as f:
    text = f.read()


#alphabet
unique_chars = sorted(list(set(text)))
vocab_size= len(unique_chars)

#encoder/decoder
encoder = {c:i for i,c in enumerate(unique_chars)}
decoder = {i:c for i,c in enumerate(unique_chars)}
encode = lambda string: [encoder[c] for c in string]
decode = lambda int_list: [decoder[i] for i in int_list]

tokenized_data = encode(text)
split = int(0.9* len(tokenized_data))
train_data = tokenized_data[:split]
test_data = tokenized_data[split:]

torch.manual_seed(19)

def get_batch(mode:str):
    data = train_data if mode == "train" else test_data
    random_vec = torch.randint(low=0,high=(len(data) - block_size),size=(batch_size,))
    #(B,T)
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in random_vec])
    y = torch.stack([torch.tensor(data[i+1:i+1+block_size] ) for i in random_vec])
    x,y = x.to(device),y.to(device)
    return x,y




class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        #Q,K,V
        self.query = nn.Linear(embed_size,head_size,bias=False)
        self.key = nn.Linear(embed_size,head_size,bias=False)
        self.value = nn.Linear(embed_size,head_size,bias=False)
    def forward(self,x):
        #x is B,T,HZ
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #B,T,HZ @ B,HZ,T => B,T,T
        tmp = (q @ k.transpose(-2,-1)) * ((k.shape[-1])**-0.5)
        #remove token's awareness of future
        tmp = torch.tril(tmp)
        tmp = tmp.masked_fill(tmp ==0, float("-inf"))
        #convert to prob dist
        tmp = F.softmax(tmp,dim=-1)

        out = tmp @ v
        return out


class MultiAttentionHead(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.lin = nn.Linear(head_size*n_heads,embed_size)
    def forward(self,x):
        #since each head will have B,T,HEAD_SIZE we concatenate on the last dimensions to get back to embed size.since HEAD_SIZE * n_heads == embed_size
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.lin(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(embed_size,embed_size),
            nn.ReLU(),
            nn.Linear(embed_size,embed_size)
        )
    def forward(self,x):
        x = self.nn(x)
        return x
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embed_size // n_heads
        self.sa_heads = MultiAttentionHead(head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    def forward(self,x):
        x = self.ln1(x + self.sa_heads(x))
        x = self.ln2(x + self.ffwd(x))
        return x 






class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        #vec and positinal embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size,embed_size)
        #final layer norm
        self.ln_f = nn.LayerNorm(embed_size)
        #multiple iterations of blocks
        self.nn_blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        #final linear transformation
        self.lm_head = nn.Linear(embed_size,vocab_size)
    def forward(self,x,targets=None):
        B,T = x.shape
        x = self.token_embedding(x)
        #add token embed + pos embed
        x = x + self.pos_embedding(torch.arange(T,device=device))
        #self attention
        x = self.nn_blocks(x)
        #normalize
        x = self.ln_f(x)
        #linear from embed to vocab size
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss


model = GPTModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()), "params")
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

def train():
    model.train()
    for iter in range(max_iters):
        x,y =get_batch("train")
        logits,loss = model(x,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iter % 1000 == 0:
           print(f"loss: {loss.item()}")


def generate(x, max_new_tokens):
    for _ in range(max_new_tokens):
        x_cond = x[:,-block_size:]
        logits, loss = model.forward(x_cond)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1) 
        x_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x,x_next), dim=1)
    return x


train()
torch.save(model.state_dict(),"./model.pth")
#model.load_state_dict(torch.load("./model.pth",weights_only=True))
context = torch.zeros((1, 1), dtype=torch.long, device=device)
answer = decode(generate(context, max_new_tokens=500)[0].tolist())
answer = "".join(answer)
with open("output.txt", "w") as f:
    f.write(answer)
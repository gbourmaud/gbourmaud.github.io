#######################################
#MultiHeadAttention
#######################################

H = self.n_head
Q = self.query(query).view(N,S,H,E//H)        
K = self.key(key).view(N,T,H,E//H) 
V = self.value(value).view(N,T,H,E//H) 

Qmove = torch.movedim(Q,2,1) #N,S,H,E//H -> N,H,S,E//H
Kmove = torch.movedim(K,1,3) #N,T,H,E//H -> N,H,E//H,T

scores = Qmove.matmul(Kmove)/math.sqrt(self.head_dim) #N, H, S, T

if attn_mask is not None:
    # Ensure small probabilities in softmax
    scores = scores.masked_fill(attn_mask==0, float("-inf")) #N, H, S, T
    
scores_SM = F.softmax(scores,dim=-1) #(N, H, S, T)
scores_SM_drop = self.attn_drop(scores_SM) #(N, H, S, T)


Vmove = torch.movedim(V,2,1) #N,T,H,E//H -> N,H,T,E//H
val = scores_SM_drop.matmul(Vmove) #(N, H, S, T) . (N, H, T, E//H) -> (N, H, S, E//H)
valmove = torch.movedim(val,2,1) # N, S, H, E//H
valview = valmove.reshape(N,S,E) # N, S, E


output = self.proj(valview)


#######################################
#Correction PositionalEncoding __init__
#######################################

# Get col idx range (i) and powers
i = torch.arange(max_len)[:, None]
pows = torch.pow(10000, -torch.arange(0, embed_dim, 2) / embed_dim)
    
# Compute positional values sin/cos
pe[0, :, 0::2] = torch.sin(i * pows)
pe[0, :, 1::2] = torch.cos(i * pows)
    
######################################
#Correction PositionalEncoding forward
######################################

output = x + self.pe[:, :S]
output = self.dropout(output)

#########################################
#Correction CaptioningTransformer forward
#########################################

N, T = captions.shape

# Embed the captions.
# shape: [N, T] -> [N, T, W]
caption_embeddings = self.embedding(captions)
caption_embeddings = self.positional_encoding(caption_embeddings)

# Project image features into the same dimension as the text embeddings.
# shape: [N, D] -> [N, W] -> [N, 1, W]
projected_features = self.visual_projection(features).unsqueeze(1)

# An additive mask for masking the future (one direction).
# shape: [T, T]
tgt_mask = torch.tril(torch.ones(T, T,
                                 device=caption_embeddings.device,
                                 dtype=caption_embeddings.dtype))

# Apply the Transformer decoder to the caption, allowing it to also
# attend to image features.
features = self.transformer(tgt=caption_embeddings,
                            memory=projected_features,
                            tgt_mask=tgt_mask)

# Project to scores per token.
# shape: [N, T, W] -> [N, T, V]
scores = self.output(features)
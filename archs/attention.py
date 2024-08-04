import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import math

class RelativePosition(nn.Module):
    """
    Relative Position Embeddings Module

    This module generates learnable relative position embeddings to enrich
    the self-attention mechanism with information about the relative distances
    between elements in input sequences.

    Args:
        d_a (int): Number of dimensions in the relative position embeddings.
        k (int): Clipping distance.

    Attributes:
        position_embeddings (nn.Parameter): Learnable parameter for relative position embeddings.

    Example:
        >>> # Create a RelativePosition instance with 16 dimensions and clipping distance of 10
        >>> relative_position = RelativePosition(d_a=16, k=10)
        >>> # Generate relative position embeddings for sequences of lengths 5 and 7
        >>> embeddings = relative_position(length_query=5, length_key=7)
    """

    def __init__(self, embed_dim: int, clip_dist: int):
        """
        Initialize the RelativePosition module.

        Args:
        - embed_dim (int): Number of dimensions in the relative position embeddings.
        - k (int): Clipping distance.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_dist = clip_dist
        self.position_embeddings = nn.Parameter(torch.empty((2 * clip_dist + 1, embed_dim)))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        """
        Compute relative position embeddings.

        Args:
        - length_query (int): Length of the query sequence.
        - length_key (int): Length of the key sequence.

        Returns:
        - embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        indices_query = torch.arange(length_query, device=self.position_embeddings.device)
        indices_key = torch.arange(length_key, device=self.position_embeddings.device)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.clip_dist, self.clip_dist)
        final_matrix = distance_matrix_clipped + self.clip_dist
        embeddings = self.position_embeddings[final_matrix.to(torch.long)]

        return embeddings


class RelationAwareAttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        feat_dim (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, feat_dim, head_dim, k_bias_matrix, v_bias_matrix):
        """
        Initializes the RelationAwareAttentionHead.

        Args:
            feat_dim (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
            k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
            v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(feat_dim, head_dim)
        self.key_weights: nn.Linear = nn.Linear(feat_dim, head_dim)
        self.value_weights: nn.Linear = nn.Linear(feat_dim, head_dim)
        self.k_bias_matrix = k_bias_matrix
        self.v_bias_matrix = v_bias_matrix

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(key) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(value) # (b_s, n_t, head_dim)

        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)

        # Relative Position Attention scores
        attn_2: torch.Tensor = torch.matmul(query.permute(1, 0, 2), self.k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)

        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)

        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)

        # Relative Position Representation for values
        values_2: torch.Tensor = torch.matmul(att_weights.permute(1, 0, 2), self.v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)

        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value


class RelationAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        feat_dim (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        k (int): Clipping distance for relative position embeddings.
        seq_len (int): Length of the input sequences.

    Attributes:
        feat_dim (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        relative_position_k (RelativePosition): Instance of RelativePosition for query-key relative positions.
        relative_position_v (RelativePosition): Instance of RelativePosition for query-value relative positions.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        attention_heads (nn.ModuleList): List of RelationAwareAttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, feat_dim, num_heads, clip_dist, seq_len):
        """
        Initializes the RelationAwareMultiHeadAttention layer.

        Args:
            feat_dim (int): Hidden size for the model (embedding dimension).
            num_heads (int): Number of attention heads.
            k (int): Clipping distance for relative position embeddings.
            seq_len (int): Length of the input sequences.
        """
        super().__init__()
        self.feat_dim: int = feat_dim
        self.num_heads: int = num_heads
        self.head_dim: int = feat_dim // num_heads
        self.relative_position_key: torch.Tensor = RelativePosition(self.head_dim, clip_dist)
        self.relative_position_val: torch.Tensor = RelativePosition(self.head_dim, clip_dist)
        self.k_bias_matrix: torch.Tensor = self.relative_position_key(seq_len, seq_len)
        self.v_bias_matrix: torch.Tensor = self.relative_position_val(seq_len, seq_len)

        #   The best hope for Bosnia is to actually throw off these decrees.
        #   Create multiple attention heads. 
        self.attention_heads: nn.ModuleList = nn.ModuleList([RelationAwareAttentionHead(self.feat_dim, self.head_dim,
                                                                           self.k_bias_matrix, self.v_bias_matrix) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(feat_dim, feat_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, feat_dim, key_dim, val_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.feat_dim = feat_dim

        self.proj_q, self.bias_q = self._get_proj_bias(key_dim)
        self.proj_k, self.bias_k = self._get_proj_bias(key_dim)
        self.proj_v, self.bias_v = self._get_proj_bias(val_dim)
        
        self.output_proj = nn.Linear(val_dim * num_heads, feat_dim, bias=False)

        self.register_buffer('scale', torch.tensor(key_dim, dtype=float).sqrt())
    
    def _get_proj_bias(self, dim_size):
        proj = nn.Parameter(torch.Tensor(self.num_heads, self.feat_dim, dim_size))
        bias = nn.Parameter(torch.Tensor(1, self.num_heads, 1, dim_size))
        nn.init.xavier_normal_(proj)
        nn.init.constant_(bias, 0.)
        return proj, bias

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
            query: (B,KPTS,D): (batch, #-keypoints, feat_dim) (1,3,1024)
            key: (B,KPTS,D): (batch, #-keypoints, feat_dim) (1,3,1024)
            value: (B,KPTS,D): (batch, #-keypoints, feat_dim) (1,3,1024)
        """
        print(f"query.shape: {query.shape} key.shape: {key.shape} value.shape: {value.shape}")
        # batch, #kpts, featdim

        query = (query.unsqueeze(1) @ self.proj_q) + self.bias_q
        key = (key.unsqueeze(1) @ self.proj_k) + self.bias_k
        value = (value.unsqueeze(1) @ self.proj_v) + self.bias_v
        # batch, head, seqlen, dk|dv
        query, key, value = query.unsqueeze(3), key.unsqueeze(2), value.unsqueeze(2)
        # q: (batch, # heads, #kpts, 1, dk)
        # k, v: (batch, #heads, 1, kvlen, dk|dv)
        logits = (query * key / self.scale).sum(-1, keepdim=True)
        # batch, head, qlen, kvlen, 1
        weighted_v = F.softmax(logits, -2) * value
        # batch, head, qlen, kvlen, dv
        heads = weighted_v.sum(-2)
        # batch, head, qlen, dv
        hid = torch.cat(heads.unbind(1), -1)
        # batch, qlen, dv * head (1, 3, 256)
        output = self.output_proj(hid)
        # batch, qlen, dmodel
        return output
    

if __name__ == "__main__":


    feat_dim = 768
    num_heads = 8
    head_dim = 64
    clip_dist = 4
    seq_len=20
    batch_size = 16

    # Set a random seed for reproducibility
    torch.manual_seed(42)

    # Create instances of RelativePosition, RelationAwareAttentionHead, and RelationAwareMultiHeadAttention
    relative_position_key = RelativePosition(embed_dim=head_dim, clip_dist=clip_dist)
    relative_position_val = RelativePosition(embed_dim=head_dim, clip_dist=clip_dist)
    embedding_key = relative_position_key(length_query=5, length_key=seq_len)
    embedding_val = relative_position_val(length_query=5, length_key=seq_len)

    attention_head = RelationAwareAttentionHead(feat_dim=feat_dim,
                                            head_dim=head_dim,
                                            k_bias_matrix=relative_position_key(seq_len, seq_len),
                                            v_bias_matrix=relative_position_val(seq_len, seq_len))

    multihead_attention = RelationAwareMultiHeadAttention(feat_dim, num_heads, clip_dist, seq_len)

    # Generate dummy input tensors
    x_input = torch.rand((batch_size, seq_len, feat_dim))    #   seq_len: # of keypoints. #  feat_dim is embedding dimension

    # Test RelativePosition
    relative_position_embeddings = relative_position_key(seq_len, seq_len)
    print("Relative Position Embeddings Shape:", relative_position_embeddings.shape)

    #   Test RelationAwareAttentionHead
    #   This is self-attention. :)
    #   
    output_attention_head = attention_head(x_input, x_input, x_input)
    print("RelationAwareAttentionHead Output Shape:", output_attention_head.shape)

    # Test RelationAwareMultiHeadAttention
    output_multihead_attention = multihead_attention(x_input, x_input, x_input)
    print("RelationAwareMultiHeadAttention Output Shape:", output_multihead_attention.shape)

    
    mha = MultiheadAttention(feat_dim=1024, key_dim=256, val_dim=256, num_heads=2)
    q = torch.rand((1,20,1024))
    k = torch.rand((1,4,1024))
    v = torch.rand((1,1,1024))

    print(f"x.shape: {q.shape}")
    o  = mha(query=q,key=k,value=v)
    print(f"o.shape: {o.shape}")
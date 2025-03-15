# self attention, value token
import torch
import torch.nn as nn
import torch.nn.functional as f


class Actor(nn.Module):
    def __init__(self, block_num, head_num):
        super(Actor, self).__init__()

        self.device = torch.device('cuda')
        self.actor = ViT(16, 1, 128, 256, head_num, block_num, 0.1, 0.1)

        self.conv1 = nn.Conv2d(4, 16, 1)
        self.conv2 = nn.Conv2d(16, 32, 1)
        self.conv3 = nn.Conv2d(32, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 1)

        self.head = nn.Sequential(nn.LayerNorm(128),
                                  nn.Linear(128, 1))

        self.fc1 = nn.Linear(4,32)
        self.fc2 = nn.Linear(32,128)

    def forward(self, attention_map, values):
        values = (values - 8) / 4.6
        values = f.relu(self.fc2(f.relu(self.fc1(values))))
        values = values.view(-1, 1, 128)

        attention_map = f.relu(self.conv4(
            f.relu(self.conv3(f.relu(self.conv2(f.relu(self.conv1(attention_map))))))))

        out, _ = self.actor(attention_map, values)
        policy = torch.mean(out, 2)

        policy = policy[:, 1:257]
        value = self.head(out[:, 0])

        return policy, value


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.Linear(mlp_num_outputs, mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(mlp_num_hiddens, mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads,
                                               dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, X, valid_lens=None):
        attn_output, attn_output_weights = self.attention(*([self.ln1(X)] * 3))
        X = X + attn_output
        return X + self.mlp(self.ln2(X)), attn_output_weights


class ViT(nn.Module):
    """Vision Transformer."""

    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))

    def forward(self, X, v):
        X = self.patch_embedding(X)
        X = torch.cat((v, X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X, attn_output_weights = blk(X)
        return X, attn_output_weights

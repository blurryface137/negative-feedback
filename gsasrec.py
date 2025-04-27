# gsasrec.py

import torch
import torch.nn as nn
from transformer_decoder import TransformerBlock

class GSASRec(nn.Module):
    def __init__(
        self,
        num_items,
        sequence_length=200,
        embedding_dim=256,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.5,
        reuse_item_embeddings=False,
    ):
        super().__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.reuse_item_embeddings = reuse_item_embeddings

        self.embeddings_dropout = nn.Dropout(dropout_rate)

        # item & position embeddings
        self.item_embedding = nn.Embedding(self.num_items + 2, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.sequence_length, self.embedding_dim)

        # action embedding: +v для positive, –v для negative, 0 для PAD
        self.action_positive = nn.Parameter(torch.randn(self.embedding_dim))

        # transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.seq_norm = nn.LayerNorm(self.embedding_dim)

        # output embeddings (можно шарить с item_embedding)
        if reuse_item_embeddings:
            self.output_embedding = None
        else:
            self.output_embedding = nn.Embedding(self.num_items + 2, self.embedding_dim)

        # head для регуляризации негативов
        self.negative_head = nn.Linear(self.embedding_dim, self.embedding_dim)


    def get_output_embeddings(self):
        return self.item_embedding if self.output_embedding is None else self.output_embedding


    def forward(self, item_ids: torch.Tensor, action_ids: torch.Tensor):
        """item_ids, action_ids — shape [B,S]"""
        bsz, seqlen = item_ids.shape

        it_emb = self.item_embedding(item_ids)  # [B,S,E]

        pos_ids = torch.arange(seqlen, device=item_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)  # [1,S,E]
        pos_emb = pos_emb.expand(bsz, -1, -1)

        # sign:  +1  (label=1)  |  -1  (label=0)  |  0  (PAD=-1)
        sign = torch.where(action_ids >= 0, (2 * action_ids) - 1, 0).unsqueeze(-1).float()
        action_emb = sign * self.action_positive  # broadcast [B,S,E]

        seq = it_emb + pos_emb + action_emb
        pad_mask = (item_ids != (self.num_items + 1)).float().unsqueeze(-1)  # [B,S,1]
        seq = self.embeddings_dropout(seq) * pad_mask

        attns = []
        for block in self.transformer_blocks:
            seq, att = block(seq, pad_mask)
            attns.append(att)
        seq = self.seq_norm(seq)

        # отрицательная голова (для регуляризации)
        neg_mask = (action_ids == 0).float().unsqueeze(-1)
        neg_out = self.negative_head(seq) * neg_mask

        return seq, neg_out, attns


    @torch.no_grad()
    def get_predictions(self, items_batch, actions_batch, topk: int, rated=None):
        seq_emb, _, _ = self.forward(items_batch, actions_batch)
        bsz, seqlen, _ = seq_emb.shape

        pos_mask = (actions_batch == 1)
        positions = torch.arange(seqlen, device=actions_batch.device).expand_as(actions_batch)
        last_pos_idx = torch.where(pos_mask, positions, torch.zeros_like(positions)).argmax(dim=1)
        last_emb = seq_emb[torch.arange(bsz, device=items_batch.device), last_pos_idx, :]

        out_emb = self.get_output_embeddings().weight
        scores  = torch.einsum("bd,nd->bn", last_emb, out_emb)
        scores[:, 0] = float("-inf")
        scores[:, self.num_items + 1:] = float("-inf")


        # last_emb = seq_emb[:, -2, :]                        # [B,E]         вот здесь исправил с -1 на -2
        # out_emb = self.get_output_embeddings().weight       # [N+2,E]
        # scores = torch.einsum("bd,nd->bn", last_emb, out_emb)

        # # запрещаем PAD + zero id
        # scores[:, 0] = float("-inf")
        # scores[:, self.num_items + 1:] = float("-inf")

        # убираем уже просмотренные
        if rated is not None:
            for i, seen in enumerate(rated):
                for itm in seen:
                    if itm <= self.num_items:
                        scores[i, itm] = float("-inf")

        return torch.topk(scores, topk, dim=1).indices, torch.topk(scores, topk, dim=1).values

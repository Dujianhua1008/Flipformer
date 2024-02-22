import torch
import random
import numpy as np
import torch.nn as nn
import logging
from performer_pytorch import SelfAttention
from performer_pytorch import Performer as performer
from transformers.models.bert.modeling_bert import  BertSelfOutput, BertIntermediate, BertOutput
class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x

# class PerformerAttention(nn.Module):
#     def __init__(self, config):
#         super(PerformerAttention, self).__init__()
#         self.config = config
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" %
#                 (config.hidden_size, config.num_attention_heads))
#         self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
#         self.num_attention_heads = config.num_attention_heads
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.input_dim= config.hidden_size
#         self.attn = SelfAttention(
#             dim=config.hidden_size,
#             heads=self.num_attention_heads,
#             causal=False,
#         )
#         self.output = BertSelfOutput(config)
        
#         self.apply(self.init_weights)

#     def init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
                
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads,
#                                        self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
    
#     def forward(self, hidden_states, attention_mask):
#         # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
#         batch_size, seq_len, _ = hidden_states.shape
#         attn_out = self.attn(hidden_states)
#         out = self.output(attn_out,hidden_states)
#         return out
# class PerformerLayer(nn.Module):
#     def __init__(self, config):
#         super(PerformerLayer, self).__init__()
#         self.attention = PerformerAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self, hidden_states, attention_mask):
#         attention_output = self.attention(hidden_states, attention_mask)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output
    
class PerformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(PerformerEncoder, self).__init__()
        self.config = config
        # self.encoders = nn.ModuleList([PerformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.encoders = performer(
            dim=config.hidden_size,
            dim_head=config.hidden_size //config.num_attention_heads,
            depth=config.num_hidden_layers,
            heads=config.num_attention_heads,
            causal=False
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print(embeddings.size())
        all_hidden_states = [embeddings]

        # for i, layer_module in enumerate(self.encoders):
        #     layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
        #     all_hidden_states.append(layer_outputs)
        out = self.encoders(all_hidden_states[-1])
        all_hidden_states.append(out)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 


class Performer(torch.nn.Module):

    def __init__(self,config):
        super(Performer, self).__init__()
        self.config = config
        self.dense_linear = nn.Linear(config.hidden_size,4)
        self.word_embedding = nn.Embedding(config.vocab_size,256,padding_idx=0)
        self.fastformer_model = PerformerEncoder(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,labels):
        mask=input_ids.bool().float()
        embds=self.word_embedding(input_ids)
        text_vec = self.fastformer_model(embds,mask)
        score = self.dense_linear(text_vec)
        loss = self.criterion(score, labels) 
        return loss, score
import torch
import random
import numpy as np
import torch.nn as nn
import logging
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

class FlipSelfAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super(FlipSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.length_flip_index_dict={}

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim
        batch_size, seq_len, _ = hidden_states.shape 
        pading_tensor_length = int(2**np.ceil(np.log2(seq_len)))
        if pading_tensor_length not in self.length_flip_index_dict:
            tempindex=np.arange(pading_tensor_length)
            #bias = [0]+[random.randint(1,pading_tensor_length-1) for _ in range(3)]
            tensors=[]
            #for j in bias:
            if self.num_attention_heads<=int(np.ceil(np.log2(seq_len))):
                for i in range(self.num_attention_heads):
                    pp=int(2**(np.ceil(np.log2(seq_len))-1-i))
                    temp=np.reshape(tempindex,(pp,pading_tensor_length//pp))
                    tensors.append(np.flip((temp)%pading_tensor_length,axis=-1).flatten())
            else:
                for i in range(self.num_attention_heads):
                    pp=int(2**(np.ceil(np.log2(seq_len))-1-i%int(np.ceil(np.log2(seq_len)))))
                    temp=np.reshape(tempindex,(pp,pading_tensor_length//pp))
                    if i<np.ceil(np.log2(seq_len)) and i<self.num_attention_heads//2:
                        bias = 0
                    else:
                        bias = random.randint(1,pading_tensor_length-1)
                    tensors.append(np.flip((temp+bias)%pading_tensor_length,axis=-1).flatten())
            self.length_flip_index_dict[pading_tensor_length] = torch.LongTensor(np.array(tensors)).to(hidden_states.device)

        mixed_query_layer = self.query(hidden_states).view(-1,seq_len,self.num_attention_heads,self.attention_head_size).transpose(1, 2)
        mixed_key_layer = self.key(hidden_states).view(-1,seq_len,self.num_attention_heads,self.attention_head_size).transpose(1, 2)
      
        newten=[]
        for i in range(self.num_attention_heads):
            newten.append(torch.index_select(mixed_key_layer[:,i], 1, self.length_flip_index_dict[pading_tensor_length][i]))
        fliper=torch.stack(newten,dim=1)
        mixed_query_layer = mixed_query_layer*fliper
        mixed_query_layer = mixed_query_layer.transpose(1, 2).reshape(batch_size,seq_len,self.num_attention_heads*self.attention_head_size)

        return mixed_query_layer 

class FlipAttention(nn.Module):
    def __init__(self, config):
        super(FlipAttention, self).__init__()
        self.self = FlipSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class FlipformerLayer(nn.Module):
    def __init__(self, config):
        super(FlipformerLayer, self).__init__()
        self.attention = FlipAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class FlipformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(FlipformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([FlipformerLayer(config) for _ in range(config.num_hidden_layers)])
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
        
        all_hidden_states = [embeddings]
        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 
class FlipFormer(torch.nn.Module):

    def __init__(self,config):
        super(FlipFormer, self).__init__()
        self.config = config
        self.dense_linear = nn.Linear(config.hidden_size,config.num_labels)
        self.word_embedding = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
        self.fastformer_model = FlipformerEncoder(config)
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
import math
import torch
import torch.nn as nn
import datasets
import logging
import numpy as np
#from .CKA import CKA, CudaCKA
from .cka import CKA_Minibatch_Grid

#device = torch.device('cuda:5')
#cuda_cka = CudaCKA(device)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.segment_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = config.hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(self.all_head_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attn_mask, is_calc_cka = False):
        #logging.info('hidden_states.shape={}'.format(hidden_states.shape))
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        #logging.info('query.shape={}'.format(query.shape))

        query = query.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        key = key.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        value = value.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        #logging.info('query.shape={}'.format(query.shape))

        attn_mask = attn_mask[:, None, None, :]
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))
        #logging.info('attn_score.shape={}'.format(attn_score.shape))

        context = torch.matmul(attn_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(value.size(0), -1, self.all_head_size)
        #print(context.shape)
        #print('cka_sum={}'.format(cka_sum))
        context = self.hidden_dropout(self.dense(context))
        output = self.layernorm(hidden_states + context)
        return output, attn_score

    def flops(self, seq_len):
        flops = 0
        flops += 3 * seq_len * self.query.in_features * self.query.out_features
        flops += seq_len * seq_len * self.all_head_size
        flops += seq_len * seq_len * self.all_head_size
        flops += self.dense.in_features * self.dense.out_features * seq_len
        return flops

class MySupernetBertAttention(nn.Module):
    def  __init__(self, config):
        super(MySupernetBertAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        #logging.info("MySupernetBertAttention, j={}, self.num_attn_heads={}".format(j, self.num_attn_heads))
        self.attn_head_size = config.hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(self.all_head_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attn_mask, is_calc_cka = False, get_cka_pair = False):
        #logging.info('hidden_states.shape={}'.format(hidden_states.shape))
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        #logging.info('query.shape={}'.format(query.shape))

        query = query.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        key = key.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        value = value.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        #logging.info('query.shape={}'.format(query.shape))

        attn_mask = attn_mask[:, None, None, :]
        #print(attn_mask)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))
        #logging.info('attn_score.shape={}'.format(attn_score.shape))

        context = torch.matmul(attn_prob, value)
        # [64, 12, 128, 44]
        c = context.permute(1, 0, 2, 3).contiguous().view(self.num_attn_heads, value.size(0), -1)
        #logging.info("c.shape={}", format(c.shape))
        #logging.info("c={}", format(c))
        context = context.permute(0, 2, 1, 3).contiguous().view(value.size(0), -1, self.all_head_size)
        #print(context.shape)
        avg_cka = 0
        if get_cka_pair:
            is_calc_cka = True
        if is_calc_cka:
            #for now_batch in range(context.shape[0]):
            #        for i in range(self.num_attn_heads):
            #            for j in range(self.num_attn_heads):
            #                cka_sum[i][j] += cuda_cka.kernel_CKA(context[now_batch,:,i*self.attn_head_size:(i+1)*self.attn_head_size],
            #                                                     context[now_batch,:,j*self.attn_head_size:(j+1)*self.attn_head_size])
            with torch.no_grad():
                cka_logger = CKA_Minibatch_Grid(self.num_attn_heads, self.num_attn_heads)
               # for data_batch in data_loader:
               #     feature_list_1 = model_1(data_batch)  # len(feature_list_1) = d1
               #     feature_list_2 = model_2(data_batch)  # len(feature_list_2) = d2
                cka_logger.update(c, c)
                cka_sum = cka_logger.compute()  # [d1, d2]
                #logging.info("cka_sum={}".format(cka_sum))
                numpy_cka = cka_sum.sum().numpy()
                avg_cka = (numpy_cka-self.num_attn_heads)/(self.num_attn_heads*self.num_attn_heads-self.num_attn_heads)
        #print('cka_sum_MySupernetBertAttention={}'.format(cka_sum))
        context = self.hidden_dropout(self.dense(context))
        output = self.layernorm(hidden_states + context)
        if get_cka_pair:
            return output, attn_score, avg_cka, cka_sum
        return output, attn_score, avg_cka

    def flops(self, seq_len):
        flops = 0
        flops += 3 * seq_len * self.query.in_features * self.query.out_features
        flops += seq_len * seq_len * self.all_head_size
        flops += seq_len * seq_len * self.all_head_size
        flops += self.dense.in_features * self.dense.out_features * seq_len
        return flops


class BertFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(BertFeedForwardNetwork, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.activation = gelu
        self.dense2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        output = self.activation(self.dense1(hidden_states))
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        return output


class BertTransformerBlock(nn.Module):
    def __init__(self, config):
        super(BertTransformerBlock, self).__init__()

        self.attention = BertAttention(config)
        self.ffn = BertFeedForwardNetwork(config)

    def forward(self, hidden_states, attn_mask):
        attn_output, attn_score = self.attention(hidden_states, attn_mask)
        output = self.ffn(attn_output)
        return output, attn_output, attn_score
        # return output, attn_output


class BertClsPooler(nn.Module):
    def __init__(self, config):
        super(BertClsPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        output = self.activation(self.dense(hidden_states))
        output = self.dropout(output)
        return output


class BertMaskedLMHead(nn.Module):
    def __init__(self, config, embedding_weights):
        super(BertMaskedLMHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = gelu
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.lm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_decoder.weight = embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.layernorm(self.activation(self.dense(hidden_states)))
        output = self.lm_decoder(hidden_states) + self.lm_bias
        return output


class BertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(BertSingle, self).__init__()

        self.use_lm = use_lm
        self.embeddings = BertEmbedding(config)
        self.encoder = nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_lm:
            self.lm_head = BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_attn_score, all_ffn_outputs = [], [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output, attn_score = layer(output, attn_mask)
            all_attn_score.append(attn_score)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)
            #logging.info(
             #   'attn_output.shape:{},output.shape:{}'.format(
              #      attn_output.shape, output.shape))

        if self.use_lm:
            output = self.lm_head(output)
        return output, all_attn_score, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class Bert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(Bert, self).__init__()

        self.task = task
        self.return_hid = return_hid
        self.embeddings = BertEmbedding(config)
        self.encoder = nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_layers)])

        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = BertClsPooler(config)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = BertClsPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)
        for layer in self.encoder:
            output, attn_output, attn_score= layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hid:
                return output, all_attn_outputs, all_ffn_outputs
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hid:
                return start_logits.squeeze(-1), end_logits.squeeze(-1), all_attn_outputs, all_ffn_outputs
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hid:
                return output, all_attn_outputs, all_ffn_outputs
            return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

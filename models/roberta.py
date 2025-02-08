import math
import torch
import torch.nn as nn
import datasets
import models
import logging

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias



class RobertaEmbedding(nn.Module):
    def __init__(self, config):
        super(RobertaEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #position_embeddings = position_embeddings.squeeze(2)
        #print("token_embeddings.shape=",token_embeddings.shape)
        #print("position_embeddings.shape=", position_embeddings.shape)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super(RobertaPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.activation(self.dense(output))
        output = self.dropout(output)
        return output


class RobertaSingle(models.BertSingle):
    def __init__(self, config):
        super(RobertaSingle, self).__init__(config)
        self.embeddings = RobertaEmbedding(config)


class Roberta(nn.Module):
    def __init__(self, config, task):
        super(Roberta, self).__init__()

        self.task = task
        self.embeddings = RobertaEmbedding(config)
        self.encoder = nn.ModuleList([models.BertTransformerBlock(config) for _ in range(config.num_layers)])

        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = RobertaPooler(config)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = RobertaPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.lm_head = RobertaLMHead(config)
        self.use_lm = False
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        output = self.embeddings(token_ids, position_ids)
        numlayer=0
        for layer in self.encoder:
            numlayer+=1
            #logging.info("layer{}".format(numlayer))
            output, _, _ = layer(output, attn_mask)

        if self.use_lm:
            output = self.lm_head(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
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

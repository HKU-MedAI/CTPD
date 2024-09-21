import torch
import torch.nn as nn


class BertForRepresentation(nn.Module):
    def __init__(self, model_name, BioBert):
        super().__init__()
        self.bert = BioBert
        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name = model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None, doc_idx_list=None):
        txt_arr = []
        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):
            if 'Longformer' in self.model_name:
                global_attention_mask = torch.clamp(
                    attention_mask.clone() - 1,
                    0, 1)
                attention_mask = torch.clamp(attention_mask, 0, 1)
                text_embeddings = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask)
            else:
                text_embeddings = self.bert(
                    input_ids, attention_mask=attention_mask)
            text_embeddings = text_embeddings[0][:, 0, :]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)
        txt_arr = torch.stack(txt_arr)

        return txt_arr
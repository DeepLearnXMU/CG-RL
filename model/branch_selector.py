# coding=utf-8

from model.attention import MultiHeadAttention
import random
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical

class BranchSelector(nn.Module):
    def __init__(self, hidden_size,action_embed_size,field_embed_size, grammar_len, field_len, production_embed, field_embed):
        super(BranchSelector, self).__init__()
        self.production_embed = nn.Embedding(grammar_len + 1, action_embed_size)
        self.field_embed = nn.Embedding(field_len, field_embed_size)
        input_size = hidden_size + action_embed_size + field_embed_size
        self.input_linear = nn.Linear(input_size, 128, bias=True)
        self.score_linear = nn.Linear(128, 1, bias=True)
        self.dropout = nn.Dropout(0.3)

    def score(self, h_t, action_ids, fields_ids, masks, select=False):
        """
        :param h_t: Variable(batch_size, hidden_size)
        :param action_emb: Variable(batch_size, action_embed_size)
        :param fields_emb: Variable(batch_size,  max_field, field_embed_size)
        :param masks: Variable(batch_size, max_field)
        :return: Variable(batch_size, max_field)
        """
        action_emb = self.production_embed(action_ids)
        fields_emb = self.field_embed(fields_ids)
        field_len = fields_emb.size(1)

        inputs = torch.cat([h_t.unsqueeze(1).expand(-1, field_len, -1),
                            action_emb.unsqueeze(1).expand(-1, field_len, -1), 
                            fields_emb], 2)
        # (batch_size, field_len, hidden_size)
        mid = torch.tanh(self.input_linear(inputs))
        if not select:
            mid = self.dropout(mid)
        mid = self.score_linear(mid).squeeze(-1)
        
        scores = mid
        scores.data.masked_fill_(~masks.bool(), -1e9)
        scores = F.softmax(scores, dim=-1)
        
        return scores
    
    def sample(self, h_t, action_ids, fields_ids, ori_masks, select=False, epsilon=0.0):
        masks = ori_masks.clone()
        batch_size, max_field = masks.size()
        pred_branch = [[] for _ in range(batch_size)]
        pred_prob = [[] for _ in range(batch_size)]
        while(True):
            if masks.bool().any():
                mask = masks.sum(-1).nonzero().squeeze(-1)
                scores = self.score(h_t[mask], action_ids[mask], fields_ids[mask], masks[mask], select)
                weights = scores
                if not select:
                    rand = random.uniform(0, 1)
                    if rand < epsilon:
                        weights = masks[mask]/masks[mask].sum(-1, True)
                    dist = Categorical(weights)
                    branch = dist.sample()
                    
                else:
                    branch = torch.argmax(weights, dim=-1)
                    
                branch_mask = F.one_hot(branch, max_field).bool()
                for mask_id, b in enumerate(branch):
                    masks[mask[mask_id]].data.masked_fill_(branch_mask[mask_id], 0)
                    pred_branch[mask[mask_id]].append(b.item()) 
                    pred_prob[mask[mask_id]].append(scores[mask_id][b.item()].squeeze()) 
                    
            else:
                break
        return pred_branch, pred_prob
    
    def select(self, h_t, action_ids, fields_ids, masks):
        max_field = masks.size(0)
        pred_branch = []
        while(True):
            if masks.bool().any():
                scores = self.score(h_t.unsqueeze(0), action_ids.unsqueeze(0), fields_ids.unsqueeze(0), masks.unsqueeze(0), True).squeeze(0)

                branch = torch.argmax(scores, dim=-1)
                pred_branch.append(branch.item()) 
                
                branch_mask = F.one_hot(branch, max_field).bool()
                masks.data.masked_fill_(branch_mask, 0)
            else:
                break
        return pred_branch
        

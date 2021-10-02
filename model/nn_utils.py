# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from six.moves import xrange

       
        
def resort_actions(t, tgt_action_infos, pred_branch, ori_id):
    # resort action info
    def resort(t, sub_action, sort_id=None):
        action_info = sub_action[t]
        result = sub_action[:t+1]

        fields = action_info.action.production.fields
        branchs = len(fields)
        childs_region = []
        current_child = 0
        fields_index = []
        region_temp = []
        end_index = None
        for i in range(t+1, len(sub_action)):
            # child node
            if sub_action[i].parent_t < action_info.t:
                end_index = i
                break
            if sub_action[i].parent_t == action_info.t:
                if i != t+1:
                    childs_region.append(region_temp)
                    region_temp = []
                # first node of current field
                if current_child != branchs and sub_action[i].frontier_field == fields[current_child]:
                    fields_index.append(len(childs_region))
                    current_child += 1
            region_temp.append(sub_action[i])
        childs_region.append(region_temp)
        assert len(fields_index) == branchs
        assert len(childs_region) >= branchs

        fields_region = []
        for i in range(branchs):
            if i == branchs-1:
                fields_region.append([fields_index[i], end_index])
            else:
                fields_region.append([fields_index[i], fields_index[i + 1]])    

        assert len(sort_id) == len(fields_region)
        if sort_id:
            fields_region = [fields_region[i] for i in sort_id]
        else:
            print("Wrong!")
        
        for index in fields_region:
            for region in childs_region[index[0]:index[1]]:
                result.extend(region)
        if end_index is not None:
            result.extend(sub_action[end_index:])
        return result
        
    def resort_id(new_tgt_action_infos, ori_id):
        index = [0] * len(new_tgt_action_infos)
        new_ori_id = [0] * len(new_tgt_action_infos)
        for i, action in enumerate(new_tgt_action_infos):
            new_ori_id[i] = ori_id[action.t]
            index[action.t] = i
            new_tgt_action_infos[i].t = i
        for i in range(t+1, len(new_tgt_action_infos)):
            new_tgt_action_infos[i].parent_t = index[new_tgt_action_infos[i].parent_t]
        return new_ori_id
    
    new_actions = []
    for batch_id, branch in enumerate(pred_branch):
        if branch:
            assert isinstance(tgt_action_infos[batch_id][t].action, ApplyRuleAction)
            assert len(branch) == len(tgt_action_infos[batch_id][t].action.production.fields)
            new_action = resort(t, tgt_action_infos[batch_id], branch)
            ori_id[batch_id] = resort_id(new_action, ori_id[batch_id])
            new_actions.append(new_action)
        else:
            new_actions.append(tgt_action_infos[batch_id])
    return new_actions, ori_id
            


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask.bool(), -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


def length_array_to_mask_tensor(length_array, cuda=False, valid_entry_has_mask_one=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        if valid_entry_has_mask_one:
            mask[i][:seq_len] = 1
        else:
            mask[i][seq_len:] = 1

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])

    return sents_t


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]


def to_input_variable(sequences, vocab, cuda=False, training=True, append_boundary_sym=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    sents_t = input_transpose(word_ids, vocab['<pad>'])
    if training:
        sents_var = Variable(torch.LongTensor(sents_t), requires_grad=False)
    else:
        with torch.no_grad():
            sents_var = Variable(torch.LongTensor(sents_t), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def variable_constr(x, v, cuda=False):
    return Variable(torch.cuda.x(v)) if cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in xrange(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       source: https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal(p.data)


def identity(x):
    return x


class LabelSmoothing(nn.Module):
    """Implement label smoothing.

    Reference: the annotated transformer
    """

    def __init__(self, smoothing, tgt_vocab_size, ignore_indices=None):
        if ignore_indices is None: ignore_indices = []

        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        smoothing_value = smoothing / float(tgt_vocab_size - 1 - len(ignore_indices))
        one_hot = torch.zeros((tgt_vocab_size,)).fill_(smoothing_value)
        for idx in ignore_indices:
            one_hot[idx] = 0.

        self.confidence = 1.0 - smoothing
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, model_prob, target):
        # (batch_size, *, tgt_vocab_size)
        dim = list(model_prob.size())[:-1] + [1]
        true_dist = Variable(self.one_hot, requires_grad=False).repeat(*dim)
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        # true_dist = model_prob.data.clone()
        # true_dist.fill_(self.smoothing / (model_prob.size(1) - 1))  # FIXME: no label smoothing for <pad> <s> and </s>
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return self.criterion(model_prob, true_dist).sum(dim=-1)


class FeedForward(nn.Module):
    """Feed forward neural network adapted from AllenNLP"""

    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(FeedForward, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        self.activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        self.linear_layers = nn.ModuleList(linear_layers)
        dropout_layers = [nn.Dropout(p=value) for value in dropout]
        self.dropout = nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, x):
        output = x
        for layer, activation, dropout in zip(self.linear_layers, self.activations, self.dropout):
            output = dropout(activation(layer(output)))
        return output

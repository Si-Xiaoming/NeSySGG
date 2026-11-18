# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved
import sys
from typing import Tuple, Dict, Optional

import numpy as np
import torch, ltn
from util import box_ops
import time
import ltn_log

import torch

class FocalAggreg:
    def __init__(self, gamma=2, stable=True, is_log=True, reduce_type="mean"):
        self.stable = stable  # kept for API parity (not used in original TF code either)
        self.gamma = gamma
        self.is_log = is_log

        if reduce_type == "sum":
            self._reduce = torch.sum
        elif reduce_type == "mean":
            self._reduce = torch.mean
        else:
            raise ValueError("reduce_type must be 'sum' or 'mean'")

    def __repr__(self):
        return f"FocalAggreg(gamma={self.gamma}, stable={self.stable})"

    def __call__(self, xs, axis=None, keepdims=False, **kwargs):
        """
        Applies the same focal-style aggregation as the TF code:
            reduce( revert_log( ((1-pt)^gamma) * log(pt) ) )
        where `xs` is either log-probabilities (is_log=True) or probabilities (is_log=False).
        """
        if self.is_log:
            # xs are log-probabilities
            pt = torch.exp(xs)
            log_pt = xs
        else:
            # xs are probabilities
            pt = xs
            log_pt = torch.log(xs)

        out = ((1 - pt) ** self.gamma) * log_pt

        # revert_log: exp if is_log else identity
        if self.is_log:
            out = torch.exp(out)

        # mimic TF reduce_sum/reduce_mean with optional axis + keepdims
        if axis is None:
            return out.sum() if self._reduce is torch.sum else out.mean()

        return self._reduce(out, dim=axis, keepdim=keepdims)

class RelInSetL(torch.nn.Module):

    def __init__(self, model: Optional[torch.nn.Module] = None, use_logit: bool = True, use_log=False):
        super(RelInSetL, self).__init__()

        if use_logit:
            self.logit_model = model

        self.use_logit = use_logit

        self.softmax = torch.nn.Softmax(dim=1)  # if not use_log else torch.nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, constr: torch.Tensor, training: bool = False, debug=False):

        # batch dim is the number of grounding to need to squeeze to done it properly
        assert mask.size()[1] == constr.size()[1], 'invalid dimension between constraint and relative mask'

        # take logit from the model
        if self.use_logit == True:
            logits = self.logit_model(x)[:, :-1]
        else:
            logits = x

        # remove background 
        probs = self.softmax(logits)

        # prune only logit related to mask
        out = torch.stack([torch.sum(probs[i] * constr[i][mask[i]]) for i in range(mask.size()[0])])


        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'


        # hack to handle float precision that could lead to 1.00000000000000000000000001
        out = torch.clamp(out, min=0.0, max=1.0)
        assert (out >= 0).all() and (out <= 1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of \
                                                     out:\n{out}\nfrom INPUTS:\nmask:{mask}\nconstr={constr}\nlogits={logits}\nprobs={probs}\n'

        return out


class RelIsAL(torch.nn.Module):

    def __init__(self, model: Optional[torch.nn.Module] = None, use_logit: bool = True, use_log=False):
        super(RelIsAL, self).__init__()

        if use_logit:
            self.logit_model = model

        self.use_logit = use_logit

        self.softmax = torch.nn.Softmax(dim=1)  # if not use_log else torch.nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, label: torch.Tensor, training: bool = False):

        # take logit from the model
        if self.use_logit == True:
            logits = self.logit_model(x)[:, :-1]
        else:
            logits = x

        probs = self.softmax(logits)

        # prune only logit related to mask
        out = torch.sum(probs * label, dim=1)

        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'
        # assert (out >=0).all() and (out <=1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of out:\n{out}\n'
        return out

class IsOfClass(torch.nn.Module):

    def __init__(self, model: Optional[torch.nn.Module] = None, use_logit: bool = True):
        super(IsOfClass, self).__init__()

        if use_logit:
            self.logit_model = model

        self.use_logit = use_logit

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, label: torch.Tensor, training: bool = False):

        # take logit from the model
        if self.use_logit == True:
            logits = self.logit_model(x)[:, :-1]
        else:
            logits = x

        probs = self.softmax(logits)

        # prune only logit related to mask
        out = torch.sum(probs * label, dim=1)

        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'
        # assert (out >=0).all() and (out <=1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of out:\n{out}\n'
        return out

class EntInSet(torch.nn.Module):

    def __init__(self, model: Optional[torch.nn.Module] = None, use_logit: bool = False, use_log=False):
        super(EntInSet, self).__init__()

        self.softmax = torch.nn.Softmax(dim=1)

        if use_logit:
            self.logit_model = model

        self.use_logit = use_logit

    def forward(self, x: torch.Tensor, mask: torch.Tensor, constr: torch.Tensor, training: bool = False, use_log=False):

        # batch dim is the number of grounding to need to squeeze to done it properly
        assert mask.size()[1] == constr.size()[1], 'invalid dimension between constraint and relative mask'
        # take logit from the model

        if self.use_logit == True:
            logits = self.logit_model(x)[:, :-1]
        else:
            logits = x

        probs = self.softmax(logits)

        # prune only logit related to mask
        out = torch.stack([torch.sum(probs[i] * constr[i][mask[i]]) for i in range(mask.size()[0])])

        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'
        out = torch.clamp(out, min=0.0, max=1)
        assert (out >= 0).all() and (out <= 1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of \
                                                     out:\n{out}\nfrom INPUTS:\nmask:{mask}\nconstr={constr}\nlogits={x}\nprobs={probs}\n'
        return out


class PredFromEmbedding(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, use_log=False):
        super(PredFromEmbedding, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)  # if not use_log else torch.nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, label: torch.Tensor, use_log=False):
        probs = self.softmax(logits)

        out = torch.sum(probs * label, dim=-1)

        return out


class PredFromLogit(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, use_log=False):
        super(PredFromLogit, self).__init__()

    def forward(self, logits: torch.Tensor):
        out = logits

        return out


def tensor_mask_hier(cls_x, hier_dict: Dict[Tuple, torch.Tensor], device: torch.device,
                     hier_list: Optional[torch.device]) -> torch.Tensor:
    """
    generate the mask of the relationship
    """
    # remove to debug
    # assert type_constr<3, f'{type_constr} not supported: it must be between 0 and  2'
    mask_list = []

    # create mask for constraint in cls_constr tensor
    dco = {k: [False] * v.shape[0] for k, v in hier_dict.items()}

    for i in range(cls_x.size(0)):
        dc = dco.copy()
        tmp = []

        k = cls_x[i].argmax().item()
        dc[k] = [True] * len(dc[k])
        for row in list(dc.values()):
            tmp += row
        mask_list.append(tmp)
        del dc

    assert len(mask_list[0]) == len(mask_list[-1]), f'all element inside the mask tensor 0 must have the same size'
    mask = torch.tensor(mask_list, device=device)
    assert mask.size(1) == hier_list.size(
        0), f'mask_idx shape {mask.size(0)} not equal to label shape{hier_list.size(0)}'

    return mask


def tensor_mask_rel(cls_ent, cls_rel: torch.Tensor, rel_dict: Dict[Tuple, torch.Tensor], device: torch.device,
                    constr_list: Optional[torch.device],
                    use_pair=False) -> torch.Tensor:
    """
    generate the mask of the relationship
    """
    # remove to debug
    # assert type_constr<3, f'{type_constr} not supported: it must be between 0 and  2'
    mask_list = []
    if not use_pair:
        cls_sub, cls_obj = cls_ent
    # 0 negative, 1 positive
    cls_constr = cls_rel

    # create mask for constraint in cls_constr tensor
    dco = {k: [False] * v.shape[0] for k, v in rel_dict.items()}

    for i in range(cls_constr.size(0)):
        dc = dco.copy()
        tmp = []

        k = (cls_sub[i].argmax().item(), cls_obj[i].argmax().item()) if not use_pair else cls_ent[i].argmax().item()

        l = len(dc[k])
        dc[k] = [True] * l
        for row in list(dc.values()):
            tmp += row

        mask_list.append(tmp)
        del dc

    assert len(mask_list[0]) == len(mask_list[-1]), f'all element inside the mask tensor 0 must have the same size'
    mask = torch.tensor(mask_list, device=device)
    assert mask.size(1) == constr_list.size(
        0), f'mask_idx shape {mask.size(0)} not equal to label shape{constr_list.size(0)}'

    return mask


def dict_tensor_mask_rel(cls_rel: torch.Tensor, rel_dict: Dict[Tuple[int, int], torch.Tensor], device: torch.device,
                         constr_list: Optional[Dict[int, torch.device]], indices: Tuple[int] = (1, 2), use_pair=False
                         ) -> Dict[int, torch.Tensor]:
    """
    generate the tensor mask of relationship in  a compact way using dictionary
    """
    # 0 negative, 1 positive
    cls_constr = cls_rel  # if type_constr == 1 else cls_rel

    # create mask for constraint in cls_constr tensor
    mask_list = {0: [], 1: []}
    for i in range(cls_constr.size(0)):
        tmp0 = []
        tmp1 = []
        for k, v in rel_dict.items():
            tmp0.extend([False] * v[0].shape[0] if cls_constr[i].argmax().item() != k else [True] * v[0].shape[0])
            tmp1.extend([False] * v[1].shape[0] if cls_constr[i].argmax().item() != k else [True] * v[1].shape[0])

        mask_list[0].append(tmp0)
        mask_list[1].append(tmp1)

    # sanity check
    assert len(mask_list[0][0]) == len(
        mask_list[0][-1]), f'all element inside the mask tensor 0 must have the same size'
    assert len(mask_list[1][0]) == len(
        mask_list[1][-1]), f'all element inside the mask tensor 1 must have the same size'

    mask = [torch.tensor(mask_list[0], device=device), torch.tensor(mask_list[1], device=device)]
    assert mask[0].size(1) == constr_list[indices[0]].size(
        0), f'mask_idx shape {mask[0].size(1)} not equal to label shape{constr_list[indices[0]].size(0)}'
    assert mask[1].size(1) == constr_list[indices[1]].size(
        0), f'mask_idx shape {mask[1].size(1)} not equal to label shape{constr_list[indices[1]].size(0)}'

    return mask


def grounding(ent_logits, ent_bbox= None,
              ltn_strategy: int = 0,
              rel_logits: Optional[torch.Tensor] = None,
              ent_cls=None,
              rel_cls: Optional[torch.Tensor] = None,
              constr_map=None,
              device: Optional[torch.device] = None,
              constr_list: Optional[Dict[int, torch.device]] = None,
              indices=None,
              constr_type: int = 0,
              use_log: bool = False,
              hier_gr = 0,
              train_ltn=False):
    """
    given the strategy it creates a regular grounding for subject, object and relationship
    """
    if constr_type != 2:
        gr_ent, gr_rel = ground_vars(ent_logits, ent_bbox, ltn_strategy, rel_logits, use_log, train_ltn=train_ltn)

        mask_idx, constr_item = ground_mask(ent_cls, rel_cls, constr_map, device, constr_list, indices,
                                            constr_type)

        return gr_ent, gr_rel, mask_idx, constr_item
    else:
        gr_x = ground_vars(ent_logits, ent_bbox, ltn_strategy, use_log=use_log, hier_gr=hier_gr, train_ltn=train_ltn)

        mask_idx, constr_item = ground_mask(ent_cls, constr_map=constr_map, device=device, constr_list=constr_list,
                                            indices=indices, constr_type=2)

        return gr_x, mask_idx, constr_item




def ground_vars(ent_logits, ent_bbox=None, ltn_strategy: int = 0, rel_logits=None, use_log=False, hier_gr=0, train_ltn=False):
    gr_ent, gr_rel = None, None

    if isinstance(ent_bbox, Tuple):
        sub_bbox, obj_bbox = ent_bbox
        sub_logits, obj_logits = ent_logits
        if ltn_strategy == 0:
            # remove background cls since already processed before invalid bbox (eos)
            tensor_sub = sub_logits[:, :-1] if not train_ltn else sub_logits
            tensor_obj = obj_logits[:, :-1] if not train_ltn else obj_logits
        elif ltn_strategy == 1 or ltn_strategy == 2:
            tensor_sub = torch.cat((sub_logits, sub_bbox), dim=-1)
            tensor_obj = torch.cat((obj_logits, obj_bbox), dim=-1)
        else:
            raise ValueError('Invalid strategy')

        if not use_log:
            gr_sub = ltn.Variable(f'gr_sub_box', tensor_sub)
            gr_obj = ltn.Variable(f'gr_obj_box', tensor_obj)
        else:
            gr_sub = ltn_log.Variable(f'gr_sub_box', tensor_sub)
            gr_obj = ltn_log.Variable(f'gr_obj_box', tensor_obj)

        gr_ent = (gr_sub, gr_obj)
    elif hier_gr != 2:
        # Choose if to use only logits (default) or logits and bbox for variable
        if ltn_strategy == 0:
            # remove background cls since already processed before invalid bbox (eos)
            tensor_ent = ent_logits[:, :-1] if not train_ltn else ent_logits
        elif ltn_strategy == 1 or ltn_strategy == 2:
            tensor_ent = torch.cat((ent_logits, ent_bbox), dim=-1)
        else:
            raise ValueError('Invalid strategy')

        if not use_log:
            gr_ent = ltn.Variable(f'gr_ent_box', tensor_ent)
        else:
            gr_ent = ltn_log.Variable(f'gr_ent_box', tensor_ent)

        if hier_gr == 1:
            return gr_ent

    if rel_logits is not None or hier_gr == 2:
        if rel_logits is None:
            rel_logits = ent_logits

        tensor_gr_rel = rel_logits[:, :-1] if not train_ltn else rel_logits

        if not use_log:
            gr_rel = ltn.Variable(f'gr_rel_box', tensor_gr_rel)
        else:
            gr_rel = ltn_log.Variable(f'gr_rel_box', tensor_gr_rel)

        if hier_gr == 2:
            return gr_rel

    return gr_ent, gr_rel


def ground_mask(ent_cls,
                rel_cls: Optional[torch.Tensor] = None, constr_map=None,
                device: Optional[torch.device] = None,
                constr_list: Optional[Dict[int, torch.device]] = None,
                indices=None,
                constr_type: int = 0):
    # check whether constraint can be applied or not
    # pruning (cls_s, cls_o)
    assert constr_map is not None, 'constraint map cannot be None'
    assert device is not None, 'device cannot be None'
    use_pair = not isinstance(ent_cls, Tuple)
    if constr_type == 0:
        mask_idx = tensor_mask_rel(ent_cls, rel_cls, constr_map, device, constr_list[indices], use_pair=use_pair)
        constr_item = constr_list[indices].unsqueeze(0).repeat(mask_idx.size(0), 1,
                                                               1)  # sanity check to avoid dumb error
        assert torch.sum(constr_item[0] - constr_item[-1]).item() == 0, 'tensor is not  correctly repeated'
    elif constr_type == 1:
        if not use_pair:
            mask_idx = dict_tensor_mask_rel(rel_cls, constr_map, device, constr_list, indices)
            constr_item = [constr_list[indices[0]].unsqueeze(0).repeat(mask_idx[0].size(0), 1, 1),
                           constr_list[indices[1]].unsqueeze(0).repeat(mask_idx[1].size(0), 1, 1)]

            assert torch.sum(
                constr_item[0][0] - constr_item[0][-1]).item() == 0, 'tensor-0 is not  correctly repeated'
            assert torch.sum(
                constr_item[1][0] - constr_item[1][-1]).item() == 0, 'tensor-1 is not  correctly repeated'
        else:
            mask_idx = tensor_mask_rel(rel_cls, ent_cls, constr_map, device, constr_list[indices], use_pair=use_pair)
            constr_item = constr_list[indices].unsqueeze(0).repeat(mask_idx.size(0), 1,
                                                                   1)  # sanity check to avoid dumb error
            assert torch.sum(constr_item[0] - constr_item[-1]).item() == 0, 'tensor is not  correctly repeated'
    else:

        mask_idx = tensor_mask_hier(ent_cls, constr_map[indices], device, constr_list[indices])
        constr_item = constr_list[indices].unsqueeze(0).repeat(mask_idx.size(0), 1,
                                                               1)  # sanity check to avoid dumb error
        assert torch.sum(constr_item[0] - constr_item[-1]).item() == 0, 'tensor is not  correctly repeated'

    return mask_idx, constr_item

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
import time

import pandas as pd
import numpy as np
from numpy import random
import json
import csv

import ltn
import ltn_log
from .fuzzy_logic import *
from torchvision.ops import batched_nms


class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """

    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False,
                 matcher=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim * 2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim * 3)
        self.so_embed = nn.Embedding(2, hidden_dim)  # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim * 2 + 128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        start_time = time.time()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, hs_t, so_masks, _ = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2], src.shape[-1])).view(hs_t.shape[0],
                                                                                              hs_t.shape[1],
                                                                                              hs_t.shape[2], -1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1], 'sub_features': hs_sub[-1], 'obj_features': hs_obj[-1],
               'ent_features': hs[-1],
               'rel_features': torch.cat((hs_sub, hs_obj, so_masks), dim=-1)[-1]
               }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel, hs_sub, hs_obj,
               hs, torch.cat((hs_sub, hs_obj, so_masks), dim=-1))
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel, hs_sub, hs_obj,
               hs, hs_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g, 'sub_features': h, 'obj_features': i, 'ent_features': j, 'rel_features': k}
                for a, b, c, d, e, f, g, h, i, j, k in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1], hs_sub[:-1], hs_obj[:-1], hs[:-1], hs_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    LTN computation has also been added.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        3) The LTN loss is computed:
            3.1) First we create a positive/negative mask map for each axiom type (except phi1)
            using the list of positive/negative  constraints from the files
            3.2) When evaluating a given axiom, we use the mask to pre-select which of the current classes in the batch
            need to be used
            3.3) We then use those classes to select the positive/negative constraints to be evaluated
    """

    def __init__(self, model, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses,
                 device='cuda',
                 sim_strategy: int = 0, sim_margin: int = 0.5,
                 kge_key_path: str = 'map2emb.txt',
                 enable_ltn: bool = False,
                 entity_fuzzy_th: int = 0.7,
                 ltn_strategy: int = 0,
                 rel_operator: int = 0,
                 aggr_p: int = 2,
                 univ_p: int = 2,
                 different_p=None,
                 neg_constr_path='constraint/prompted_positives_vg_final_neg_full.csv',
                 pos_constr_path='constraint/prompted_positives_vg_final_pos_full.csv',
                 neg_constr_path_s='constraint/domain_neg_constr4.csv',
                 pos_constr_path_s='constraint/domain_pos_constr4.csv',
                 neg_constr_path_o='constraint/range_neg_constr4.csv',
                 pos_constr_path_o='constraint/range_pos_constr4.csv',
                 increase_dim_net: bool = False,
                 p_dropout: float = 0.2,
                 reltr_grounding: float = False,
                 ec_enable: bool = False,
                 rc_enable: bool = True,
                 tgt_as_constr: bool = False,
                 use_pred_nms: bool = False,
                 sat_pred_nms: float = 0.5,
                 max_nms_bbox: int = 30,
                 pure_nesy: bool = False,
                 rel_hard_axiom_path: str = 'constraint/rel_hard_axiom.csv',
                 rel_easy_axiom_path: str = 'constraint/rel_easy_axiom.csv',
                 use_snc: bool = True,
                 use_hierarchy = False,
                 hier_path = None,
                 use_focused = 0,
                 use_log=False,
                 use_focal=False,
                 use_pair=False,
                 deactivate_constr=None,
                 trainable_ltn=True,
                 use_ltn_gt=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.train_ltn = trainable_ltn
        self.use_focal = use_focal

        self.num_rel_classes = 51 if num_classes == 151 else 31  # Using entity class numbers to adapt rel class numbers
        empty_weight_rel = torch.ones(num_rel_classes + 1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

        self.pure_nesy = pure_nesy
        self.device = torch.device(device)
        self.aggr_p = aggr_p
        self.univ_p = univ_p

        if enable_ltn:
            self.enable_ltn = enable_ltn
            self.ec_enable = ec_enable
            self.rc_enable = rc_enable
            self.use_tgt_constr = tgt_as_constr
            self.use_log = use_log
            self.use_pair = use_pair
            self.use_hierarchy = use_hierarchy
            self.use_focused_constr = use_focused
            if deactivate_constr is not None:
                self.use_neg_constr_3 = not deactivate_constr['3']
                self.use_neg_constr_4 = not deactivate_constr['4']
                self.use_pos_constr_2 = not deactivate_constr['2']
                self.use_pos_constr_5 = not deactivate_constr['5']
            else:
                self.use_neg_constr_3 = True
                self.use_neg_constr_4 = True
                self.use_pos_constr_2 = True
                self.use_pos_constr_5 = True


            if different_p == None:
                self.equal_p = True
                if not use_log:
                    self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
                    self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")

                    # for simplicity same forall for entity and relationship
                    self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
                else:
                    if not use_focal:
                        self.forall_neg = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(), quantifier="f")
                        self.forall_pos = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(), quantifier="f")

                        # for simplicity same forall for entity and relationship
                        self.forall_ent_o_rel = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(), quantifier="f")
                    else:
                        self.forall_neg = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(), quantifier="f")
                        self.forall_pos = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(), quantifier="f")

                        # for simplicity same forall for entity and relationship
                        self.forall_ent_o_rel = ltn_log.Quantifier(ltn_log.fuzzy_ops.Aggreg_LogMeanExp(),
                                                                   quantifier="f")
            else:
                self.equal_p = False
                self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[0]), quantifier="f")
                self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[0]), quantifier="f")
                self.forall_hard = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[1]), quantifier="f")

                self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[-1]),
                                                       quantifier="f")

                # lists of easy and hard relationships
                self.rel_cls_easy = torch.tensor(pd.read_csv(rel_easy_axiom_path).values, device=device)
                self.rel_cls_hard = torch.tensor(pd.read_csv(rel_hard_axiom_path).values, device=device)

            self.rel_support = False
            if not use_log:
                self.and_luk = ltn.Connective(ltn.fuzzy_ops.AndLuk())
                self.or_op = ltn.Connective(ltn.fuzzy_ops.OrMax())
                self.not_op = ltn.Connective(ltn.fuzzy_ops.NotStandard())

                self.ltn_strategy = ltn_strategy

                if rel_operator == 0:
                    self.rel_operator = ltn.Connective(ltn.fuzzy_ops.AndLuk())
                elif rel_operator == 1:
                    self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
                elif rel_operator == 2:
                    self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesGodel())
                else:
                    raise ValueError(f'rel_operator {rel_operator} is invalid')

            else:
                self.and_luk = ltn_log.Connective(ltn_log.fuzzy_ops.And_Sum())
                self.not_op = ltn_log.Connective(ltn_log.fuzzy_ops.Not_log_negation_softmax())
                self.or_op = ltn_log.Connective(ltn_log.fuzzy_ops.OR_LogMeanExp())

                self.ltn_strategy = ltn_strategy

                if rel_operator == 0:
                    self.rel_operator = ltn_log.Connective(ltn_log.fuzzy_ops.And_Sum())
                elif rel_operator == 1 or rel_operator == 2:
                    self.rel_support = True
                    self.rel_operator = ltn_log.Connective(ltn_log.fuzzy_ops.OR_LogMeanExp())
                else:
                    raise ValueError(f'rel_operator {rel_operator} is invalid')

            if not use_log:
                # Ent predicates
                self.ent_is_a = IsOfClass(use_logit=self.train_ltn)
                self.sub_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.sub_class_embed)
                self.obj_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.obj_class_embed)
                self.ent_is_a_to_p = ltn.Predicate(model=PredFromLogit())
                self.p_sub_is_a = ltn.Predicate(model=self.sub_is_a)
                self.p_obj_is_a = ltn.Predicate(model=self.obj_is_a)
                self.p_sub_in_set = ltn.Predicate(model=EntInSet(model=self.model.sub_class_embed, use_logit=self.train_ltn))
                self.p_obj_in_set = ltn.Predicate(model=EntInSet(model=self.model.obj_class_embed, use_logit=self.train_ltn))
                self.p_ent_in_set = ltn.Predicate(model=EntInSet(model=self.model.entity_class_embed, use_logit=self.train_ltn))

                self.use_regressor = False

                # Rel predicates
                self.rel_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.rel_class_embed)

                self.rel_is_a_to_p = ltn.Predicate(model=PredFromLogit())
                self.p_rel_in_set = ltn.Predicate(model=RelInSetL(model=self.model.rel_class_embed, use_logit=self.train_ltn))
                if self.use_pair:
                    self.p_rel_in_set_o = ltn.Predicate(
                        model=RelInSetL(model=self.model.rel_class_embed, use_logit=self.train_ltn, use_log=True))
                self.p_rel_is_a = ltn.Predicate(model=self.rel_is_a)

                # Final sat
                self.sat_aggr = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

                # satisfiability for debugging purpose
                self.sat_ent_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_rel_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_neg_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_pos_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

                # entity label
                self.ot_e_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_classes), self.num_classes), 1)]
                # relationship label
                self.ot_r_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_rel_classes), self.num_rel_classes),
                                               1)]
            else:
                self.ent_is_a = IsOfClass(use_logit=self.train_ltn)
                self.sub_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.sub_class_embed)
                self.obj_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.obj_class_embed)
                self.ent_is_a_to_p = ltn_log.Predicate(model=PredFromLogit())
                self.p_sub_is_a = ltn_log.Predicate(model=self.ent_is_a)
                self.p_obj_is_a = ltn_log.Predicate(model=self.ent_is_a)
                self.p_sub_in_set = ltn_log.Predicate(model=EntInSet(model=self.model.sub_class_embed, use_logit=self.train_ltn, use_log=True))
                self.p_obj_in_set = ltn_log.Predicate(model=EntInSet(model=self.model.obj_class_embed, use_logit=self.train_ltn, use_log=True))
                self.p_ent_in_set = ltn_log.Predicate(model=EntInSet(model=self.model.entity_class_embed, use_logit=self.train_ltn, use_log=True))

                self.use_regressor = False

                # Rel predicates
                self.rel_is_a = IsOfClass(use_logit=self.train_ltn, model=self.model.rel_class_embed)
                self.rel_is_a_to_p = ltn_log.Predicate(model=PredFromLogit(use_log=True))
                self.p_rel_in_set = ltn_log.Predicate(
                    model=RelInSetL(model=self.model.rel_class_embed, use_logit=self.train_ltn, use_log=True))
                if self.use_pair:
                    self.p_rel_in_set_o = ltn_log.Predicate(
                        model=RelInSetL(model=self.model.rel_class_embed, use_logit=self.train_ltn, use_log=True))
                self.p_rel_is_a = ltn_log.Predicate(model=self.rel_is_a)

                # Final sat
                self.sat_aggr = ltn_log.fuzzy_ops.SatAgg(ltn_log.fuzzy_ops.AggregMean())

                # satisfiability for debugging purpose
                self.sat_ent_axiom = ltn_log.fuzzy_ops.SatAgg(ltn_log.fuzzy_ops.AggregMean())
                self.sat_rel_axiom = ltn_log.fuzzy_ops.SatAgg(ltn_log.fuzzy_ops.AggregMean())
                self.sat_neg_axiom = ltn_log.fuzzy_ops.SatAgg(ltn_log.fuzzy_ops.AggregMean())
                self.sat_pos_axiom = ltn_log.fuzzy_ops.SatAgg(ltn_log.fuzzy_ops.AggregMean())

                # entity label
                self.ot_e_label = [ltn_log.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_classes), self.num_classes), 1)]
                # relationship label
                self.ot_r_label = [ltn_log.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_rel_classes), self.num_rel_classes),
                                               1)]

            self.entity_fuzzy_th = entity_fuzzy_th

            # Dict with triplet list for each constr type
            self.constr_lists = {}
            self.hier_lists = {}
            self.use_pair = use_pair
            self.use_ltn_gt = use_ltn_gt
            if not self.use_pair:
                self.neg_indices = [("phi4_s", "phi4_o"), "phi3"]
                self.pos_indices = [("phi2_s", "phi2_o"), "phi5"]
                self.rel_pos_const_map_2 = self._parse_r2so_constraint(pos_constr_path, self.pos_indices[0], self.use_focused_constr)
                self.rel_neg_const_map_4 = self._parse_r2so_constraint(neg_constr_path, self.neg_indices[0], self.use_focused_constr)

                self.rel_pos_const_map_5 = self._parse_so2r_constraint(pos_constr_path, self.pos_indices[1], self.use_focused_constr)
                self.rel_neg_const_map_3 = self._parse_so2r_constraint(neg_constr_path, self.neg_indices[1], self.use_focused_constr)

                self.sat_debug = {'rel_axiom_sat': -2.0, 'sub_axiom_sat': -2.0, 'obj_axiom_sat': -2.0, 'neg3_axiom_sat': -2.0, 'pos2_axiom_sat': -2.0,
                                  'neg4_axiom_sat': -2.0, 'pos5_axiom_sat': -2.0,
                                  'entity_axiom_sat': -2.0}
                self.axiom_debug = {'rel_axiom': -2.0, 'sub_axiom': -2.0, 'obj_axiom': -2.0, 'neg3_axiom': -2.0, 'pos2_axiom': -2.0, 'neg4_axiom': -2.0,
                                    'pos5_axiom': -2.0, 'neg_easy_axiom': -2.0, 'neg3_axiom_hit': 0, 'pos2_axiom_hit': 0, 'neg4_axiom_hit': 0,
                                    'pos5_axiom_hit': 0, 'neg3_axiom_hit_batch': 0, 'pos2_axiom_hit_batch': 0, 'neg4_axiom_hit_batch': 0,
                                    'pos5_axiom_hit_batch': 0, 'neg3_axiom_act': 0, 'pos2_axiom_act': 0, 'neg4_axiom_act': 0,
                                    'pos5_axiom_act': 0, 'neg_easy_axiom': -2.0,
                                    'pos_aeasy_xiom': -2.0, 'neg_hard_axiom': -2.0, 'pos_hard_axiom': -2.0,
                                    'entity_axiom': -2.0, 'tot_axiom_act': 0}
            else:
                self.neg_indices = [("phi4_s", "phi4_o"), ("phi3_s", "phi3_o")]
                self.pos_indices = [("phi2_s", "phi2_o"), ("phi5_s", "phi5_o")]
                self.rel_pos_const_map_2 = self._parse_r2so_constraint_pair((pos_constr_path_s, pos_constr_path_o), self.pos_indices[0], self.use_focused_constr)
                self.rel_neg_const_map_4 = self._parse_r2so_constraint_pair((neg_constr_path_s, neg_constr_path_o), self.neg_indices[0], self.use_focused_constr)

                self.rel_pos_const_map_5 = self._parse_so2r_constraint_pair((pos_constr_path_s, pos_constr_path_o), self.pos_indices[1], self.use_focused_constr)
                self.rel_neg_const_map_3 = self._parse_so2r_constraint_pair((neg_constr_path_s, neg_constr_path_o), self.neg_indices[1], self.use_focused_constr)

                self.sat_debug = {'rel_axiom_sat': -2.0, 'sub_axiom_sat': -2.0, 'obj_axiom_sat': -2.0, 'neg3_s_axiom_sat': -2.0, 'pos2_s_axiom_sat': -2.0,
                                  'neg4_s_axiom_sat': -2.0, 'pos5_s_axiom_sat': -2.0, 'neg3_o_axiom_sat': -2.0, 'pos2_o_axiom_sat': -2.0,
                                  'neg4_o_axiom_sat': -2.0, 'pos5_o_axiom_sat': -2.0,
                                  'entity_axiom_sat': -2.0}
                self.axiom_debug = {'rel_axiom': -2.0, 'sub_axiom': -2.0, 'obj_axiom': -2.0, 'neg3_s_axiom': -2.0, 'pos2_s_axiom': -2.0, 'neg4_s_axiom': -2.0,
                                    'pos5_s_axiom': -2.0, 'neg3_o_axiom': -2.0, 'pos2_o_axiom': -2.0, 'neg4_o_axiom': -2.0,
                                    'pos5_o_axiom': -2.0, 'neg3_s_axiom_hit': 0, 'pos2_s_axiom_hit': 0, 'neg4_s_axiom_hit': 0,
                                    'pos5_s_axiom_hit': 0, 'neg3_o_axiom_hit': 0, 'pos2_o_axiom_hit': 0, 'neg4_o_axiom_hit': 0,
                                    'pos5_o_axiom_hit': 0, 'neg3_s_axiom_hit_batch': 0, 'pos2_s_axiom_hit_batch': 0, 'neg4_s_axiom_hit_batch': 0,
                                    'pos5_s_axiom_hit_batch': 0, 'neg3_o_axiom_hit_batch': 0, 'pos2_o_axiom_hit_batch': 0, 'neg4_o_axiom_hit_batch': 0,
                                    'pos5_o_axiom_hit_batch': 0, 'neg3_s_axiom_act': 0, 'pos2_s_axiom_act': 0, 'neg4_s_axiom_act': 0,
                                    'pos5_s_axiom_act': 0, 'neg3_o_axiom_act': 0, 'pos2_o_axiom_act': 0, 'neg4_o_axiom_act': 0,
                                    'pos5_o_axiom_act': 0, 'neg_easy_axiom': -2.0,
                                    'pos_easy_axiom': -2.0, 'neg_hard_axiom': -2.0, 'pos_hard_axiom': -2.0,
                                    'entity_axiom': -2.0, 'tot_axiom_act': 0}

            if self.use_hierarchy:
                self.hier_indices = ["ent", "rel"]

                self.hier_map = self._parse_hier(hier_path)

                self.sat_debug.update({'rel_hier_axiom_sat': -2.0, 'sub_hier_axiom_sat': -2.0, 'obj_hier_axiom_sat': -2.0, 'sub_axiom_sat': -2.0, 'obj_axiom_sat': -2.0})
                self.axiom_debug.update({'rel_hier_axiom': -2.0, 'sub_hier_axiom': -2.0, 'obj_hier_axiom': -2.0, 'sub_axiom': -2.0, 'obj_axiom': -2.0,
                                         'rel_hier_axiom_hit': -2.0, 'ent_hier_axiom_hit': -2.0})

            self._save_sat_debug = True

            # NMS for predicition pruning
            self.sat_nms_enable = use_pred_nms
            self.sat_nms_threshold = sat_pred_nms

            self.max_bbox = max_nms_bbox

    @property
    def save_sat_debug(self):
        return self._save_sat_debug

    @save_sat_debug.setter
    def save_sat_debug(self, val: bool):
        self._save_sat_debug = val

    def _reset_axiom_activation_debug(self):
        if not self.use_pair:
            activation_keys = [
                'neg3_axiom_act', 'pos2_axiom_act',
                'neg4_axiom_act', 'pos5_axiom_act'
            ]
        else:
            activation_keys = [
                'neg3_s_axiom_act', 'neg3_o_axiom_act',
                'pos2_s_axiom_act', 'pos2_o_axiom_act',
                'neg4_s_axiom_act', 'neg4_o_axiom_act',
                'pos5_s_axiom_act', 'pos5_o_axiom_act'
            ]

        for key in activation_keys:
            self.axiom_debug[key] = 0
        self.axiom_debug['tot_axiom_act'] = 0

    def _parse_hier(self, hier_path: str):
        """
        generate hierarchy constraint  <if x , then x in hier(l) > (phi3, phi5) given file where they are stored
        """

        with open(hier_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize nested lists in entities (like terrain)
        entities_hier = {}
        for group, entities in data.get("entities", {}).items():
            for e in entities:
                if e not in entities_hier.keys():
                    entities_hier[e] = [self.ot_e_label[i].value.t() for i in entities]

        # only for compact version
        for k, v in entities_hier.items():
            entities_hier[k] = torch.vstack(v)

        self.hier_lists['ent'] = torch.vstack([entities_hier[k] for k in entities_hier.keys()])

        # Normalizzazione relazioni: rimozione duplicati
        relations_hier = {}
        for group, relations in data.get("relations", {}).items():
            for r in relations:
                if r not in relations_hier.keys():
                    relations_hier[r] = [self.ot_r_label[i].value.t() for i in relations]

        # only for compact version
        for k, v in relations_hier.items():
            relations_hier[k] = torch.vstack(v)

        self.hier_lists['rel'] = torch.vstack([relations_hier[k] for k in relations_hier.keys()])

        # sanity check
        correct_shape = sum([v.size(0) for v in entities_hier.values()])
        assert self.hier_lists['ent'].size(
            0) == correct_shape, f'wrong concatenation  for neg {self.hier_lists["ent"].size()} vs {correct_shape}'
        correct_shape = sum([v.size(0) for v in relations_hier.values()])
        assert self.hier_lists['rel'].size(
            0) == correct_shape, f'wrong concatenation  for neg {self.hier_lists["rel"].size()} vs {correct_shape}'

        return {'ent': entities_hier, 'rel': relations_hier}

    def _parse_so2r_constraint(self, constr_path: str, indices, focused_constr=0):
        """
        generate constraint  <if sub and obj, then rel > (phi3, phi5) given file where they are stored
        """
        constr = {}

        common_rels = [31, 20, 48, 30, 22, 29, 50, 8, 21, 1]

        with open(constr_path, 'r') as fp:
            for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines())):
                s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
                if focused_constr != 0:
                    if r in common_rels and focused_constr == 2:
                        continue
                    elif r not in common_rels and focused_constr == 1:
                        continue

                if (s, o) not in constr.keys():
                    constr[(s, o)] = []

                constr[(s, o)].append(self.ot_r_label[r].value.t())

            # only for compact version
            for k, v in constr.items():
                constr[k] = torch.vstack(v)

        self.constr_lists[indices] = torch.vstack([constr[k] for k in constr.keys()])
        # sanity check
        correct_shape = sum([v.size(0) for v in constr.values()])
        assert self.constr_lists[indices].size(
            0) == correct_shape, f'wrong concatenation  for neg {self.constr_lists[indices].size()} vs {correct_shape}'

        return constr

    def _parse_r2so_constraint(self, constr_path: str, indices: Tuple[int] = (1, 2), focused_constr=0):
        """
        generate  <if ent, then rel > (phi2, phi4) given file where they are stored
        idx index used inside the tuple all possible constraint (2 type by construction)
        """
        constr = {}
        used_label = {}

        common_rels = [31, 20, 48, 30, 22, 29, 50, 8, 21, 1]

        with open(constr_path, 'r') as fp:
            for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines())):
                # save the axiom:  s -> o
                s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
                # save all implication related to s into a map

                if focused_constr != 0:
                    if r in common_rels and focused_constr == 2:
                        continue
                    elif r not in common_rels and focused_constr == 1:
                        continue

                if r not in constr.keys():
                    constr[r] = {0: [], 1: []}
                    used_label[r] = {0: [], 1: []}

                # choose label dependens on agent constraint or action ones

                # check if label not already present in list
                if s not in used_label[r][0]:
                    used_label[r][0].append(s)
                    constr[r][0].extend([self.ot_e_label[s].value.t()])

                if o not in used_label[r][1]:
                    used_label[r][1].append(o)
                    constr[r][1].extend([self.ot_e_label[o].value.t()])

            # compact version
            for k, v in constr.items():
                constr[k][0] = torch.vstack(v[0])
                constr[k][1] = torch.vstack(v[1])

        self.constr_lists[indices[0]] = torch.vstack([constr[k][0] for k in constr.keys()])
        # sanity check
        correct_shape = sum([v[0].size(0) for v in constr.values()])
        assert self.constr_lists[indices[0]].size(
            0) == correct_shape, f'wrong concatenation  for neg {self.constr_lists[indices[0]].size()} vs {correct_shape}'

        self.constr_lists[indices[1]] = torch.vstack([constr[k][1] for k in constr.keys()])
        # sanity check
        correct_shape = sum([v[1].size(0) for v in constr.values()])
        assert self.constr_lists[indices[1]].size(
            0) == correct_shape, f'wrong concatenation  for neg {self.constr_lists[indices[1]].size()} vs {correct_shape}'

        return constr


    def _iter_constraint_rows(self, constr_path: str, expected_cols: int):
        seen_data = False
        with open(constr_path, 'r', newline='') as fp:
            for line_no, row in enumerate(csv.reader(fp), start=1):
                row = [col.strip() for col in row]
                if not row or all(col == '' for col in row):
                    continue
                if len(row) < expected_cols:
                    raise ValueError(
                        f"Invalid constraint row in {constr_path}:{line_no}: expected at least {expected_cols} columns, got {row}")
                try:
                    parsed_row = tuple(int(row[idx]) for idx in range(expected_cols))
                except ValueError as exc:
                    if not seen_data:
                        continue
                    raise ValueError(f"Invalid constraint row in {constr_path}:{line_no}: {row[:expected_cols]}") from exc
                seen_data = True
                yield parsed_row

    def _parse_r2so_constraint_pair(self, constr_path: Tuple, indices, focused_constr=0):
        """
        generate  <if rel, then ent > (phi2, phi4) given file where they are stored
        idx index used inside the tuple all possible constraint (2 type by construction)
        """
        constr_s = {}
        constr_o = {}

        constr_path_s, constr_path_o = constr_path

        common_rels = [31, 20, 48, 30, 22, 29, 50, 8, 21, 1]

        for s, r in self._iter_constraint_rows(constr_path_s, 2):
            if focused_constr != 0:
                if r in common_rels and focused_constr == 2:
                    continue
                elif r not in common_rels and focused_constr == 1:
                    continue

            if r not in constr_s.keys():
                constr_s[r] = []
            constr_s[r].append(self.ot_e_label[s].value.t())

        # only for compact version
        for k, v in constr_s.items():
            constr_s[k] = torch.vstack(v)

        for r, o in self._iter_constraint_rows(constr_path_o, 2):
            if focused_constr != 0:
                if r in common_rels and focused_constr == 2:
                    continue
                elif r not in common_rels and focused_constr == 1:
                    continue

            if r not in constr_o.keys():
                constr_o[r] = []
            constr_o[r].append(self.ot_e_label[o].value.t())

        # only for compact version
        for k, v in constr_o.items():
            constr_o[k] = torch.vstack(v)

        self.constr_lists[indices[0]] = torch.vstack([constr_s[k] for k in constr_s.keys()])
        self.constr_lists[indices[1]] = torch.vstack([constr_o[k] for k in constr_o.keys()])
        # sanity check
        correct_shape_s = sum([v.size(0) for v in constr_s.values()])
        correct_shape_o = sum([v.size(0) for v in constr_o.values()])
        assert self.constr_lists[indices[0]].size(
            0) == correct_shape_s, f'wrong concatenation for neg {self.constr_lists[indices[0]].size()} vs {correct_shape_s}'
        assert self.constr_lists[indices[1]].size(
            0) == correct_shape_o, f'wrong concatenation for neg {self.constr_lists[indices[1]].size()} vs {correct_shape_o}'

        return (constr_s, constr_o)

    def _parse_so2r_constraint_pair(self, constr_path: Tuple, indices, focused_constr=0):
        """
        generate constraint  <if sub and obj, then rel > (phi3, phi5) given file where they are stored
        """
        constr_s = {}
        constr_o = {}

        constr_path_s, constr_path_o = constr_path

        common_rels = [31, 20, 48, 30, 22, 29, 50, 8, 21, 1]

        for s, r in self._iter_constraint_rows(constr_path_s, 2):
            if focused_constr != 0:
                if r in common_rels and focused_constr == 2:
                    continue
                elif r not in common_rels and focused_constr == 1:
                    continue

            if s not in constr_s.keys():
                constr_s[s] = []
            constr_s[s].append(self.ot_r_label[r].value.t())

        # only for compact version
        for k, v in constr_s.items():
            constr_s[k] = torch.vstack(v)

        for r, o in self._iter_constraint_rows(constr_path_o, 2):
            if focused_constr != 0:
                if r in common_rels and focused_constr == 2:
                    continue
                elif r not in common_rels and focused_constr == 1:
                    continue

            if o not in constr_o.keys():
                constr_o[o] = []
            constr_o[o].append(self.ot_r_label[r].value.t())

        # only for compact version
        for k, v in constr_o.items():
            constr_o[k] = torch.vstack(v)

        self.constr_lists[indices[0]] = torch.vstack([constr_s[k] for k in constr_s.keys()])
        self.constr_lists[indices[1]] = torch.vstack([constr_o[k] for k in constr_o.keys()])
        # sanity check
        correct_shape_s = sum([v.size(0) for v in constr_s.values()])
        correct_shape_o = sum([v.size(0) for v in constr_o.values()])
        assert self.constr_lists[indices[0]].size(
            0) == correct_shape_s, f'wrong concatenation for neg {self.constr_lists[indices[0]].size()} vs {correct_shape_s}'
        assert self.constr_lists[indices[1]].size(
            0) == correct_shape_o, f'wrong concatenation for neg {self.constr_lists[indices[1]].size()} vs {correct_shape_o}'

        return (constr_s, constr_o)

    def update_aggr_p(self, aggr_p: int, univ_p: int = 0, list_p=None):
        """change aggregator norm

        Args:
            aggr_p (int): new p-norm for aggregator FolAll and SatAggr
            univ_p (int): new p-norm for aggregator FolAll and SatAggr
            aggr_p (int): new p-norm for aggregator FolAll and SatAggr
        """
        # self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p), quantifier="f")
        self.aggr_p = aggr_p
        if not self.use_log:
            self.sat_aggr = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

            if self.save_sat_debug == True:
                self.sat_ent_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_neg_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_pos_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

            if self.equal_p == True and univ_p != 0:
                self.univ_p = univ_p
                self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
                self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
                self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")

            elif self.equal_p == False and list_p is not None:
                self.univ_p = list_p[0]
                self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[0]), quantifier="f")
                self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[0]), quantifier="f")
                self.forall_hard = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[1]), quantifier="f")
                self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[-1]), quantifier="f")

    def _compute_sat_level(self, outputs: Dict[str, torch.Tensor], sub_logits: torch.Tensor, obj_logits: torch.Tensor,
                           target_sub_classes: torch.Tensor,
                           target_obj_classes: torch.Tensor, rel_target_classes: torch.Tensor,
                           ground_truth_rel: Optional[torch.Tensor] = None,
                           rel_to_verify: Optional[torch.Tensor] = None,
                           rel_feat_to_verify: Optional[torch.Tensor] = None,
                           use_log=False):
        """
            compute sat level given output, target and logits,
        """
        sat_aggr = 0.0
        self._reset_axiom_activation_debug()
        rel_logits = outputs['rel_logits']
        n_batch = rel_logits.size()[0]

        device = rel_logits.device
        sub_boxes = outputs['sub_boxes']
        obj_boxes = outputs['obj_boxes']
        # find sub and obj logits index that above a certaint threshold, then use the inclusion set to verifity
        # constraint satisfiability
        sub_scores, sub_labels = sub_logits.max(dim=-1)
        obj_scores, obj_labels = obj_logits.max(dim=-1)
        if self.sat_nms_enable:
            mask_nms_idx = torch.zeros(sub_scores.size(), device=device)
            for b in range(n_batch):
                sub_idx_pos = batched_nms(sub_boxes[b], sub_scores[b], sub_labels[b], self.sat_nms_threshold)
                obj_idx_pos = batched_nms(obj_boxes[b], obj_scores[b], obj_labels[b], self.sat_nms_threshold)

                # consider the corner case
                min_idx = min(sub_idx_pos.size(0), obj_idx_pos.size(0))

                # handlen nms retrive two different set
                if min_idx > self.max_bbox:
                    sub_idx = sub_idx_pos[:self.max_bbox]
                    obj_idx = obj_idx_pos[:self.max_bbox]

                else:
                    sub_idx = sub_idx_pos[:min_idx]
                    obj_idx = obj_idx_pos[:min_idx]

                # use intersection

                tmp_s = torch.zeros(sub_scores[b].size(), device=device).scatter_(0, sub_idx, 1).bool()
                tmp_o = torch.zeros(obj_scores[b].size(), device=device).scatter_(0, obj_idx, 1).bool()

                nms_idx = tmp_s & tmp_o
                mask_nms_idx[b][nms_idx] = 1

            mask_nms_idx = mask_nms_idx.bool()
            if torch.sum(mask_nms_idx) == 0:
                # no valid idx, then network not ready to evaluate the constraint
                if self._save_sat_debug:
                    if not self.use_pair:
                        self.sat_debug['neg3_axiom_sat'] = -2.0
                        self.sat_debug['pos2_axiom_sat'] = -2.0
                        self.sat_debug['neg4_axiom_sat'] = -2.0
                        self.sat_debug['pos5_axiom_sat'] = -2.0
                    else:
                        self.sat_debug['neg3_s_axiom_sat'] = -2.0
                        self.sat_debug['neg3_o_axiom_sat'] = -2.0
                        self.sat_debug['pos2_s_axiom_sat'] = -2.0
                        self.sat_debug['pos2_o_axiom_sat'] = -2.0
                        self.sat_debug['neg4_s_axiom_sat'] = -2.0
                        self.sat_debug['neg4_o_axiom_sat'] = -2.0
                        self.sat_debug['pos5_s_axiom_sat'] = -2.0
                        self.sat_debug['pos5_o_axiom_sat'] = -2.0

                    if self.rc_enable:
                        self.sat_debug['rel_axiom_sat'] = -2.0

                    if self.ec_enable:
                        self.sat_debug['entity_axiom_sat'] = -2.0
                        self.sat_debug['sub_axiom_sat'] = -2.0
                        self.sat_debug['obj_axiom_sat'] = -2.0

                if self.equal_p:
                    self.axiom_debug['neg3_axiom'] = -2.0
                    self.axiom_debug['pos2_axiom'] = -2.0
                    self.axiom_debug['neg4_axiom'] = -2.0
                    self.axiom_debug['pos5_axiom'] = -2.0

                else:
                    self.axiom_debug['neg_easy_axiom'] = -2.0
                    self.axiom_debug['neg_hard_axiom'] = -2.0
                    self.axiom_debug['pos_easy_axiom'] = -2.0
                    self.axiom_debug['pos_hard_axiom'] = -2.0

                if self.rc_enable:
                    self.axiom_debug['rel_axiom'] = -2.0

                if self.ec_enable:
                    self.axiom_debug['entity_axiom'] = -2.0
                    self.axiom_debug['sub_axiom'] = -2.0
                    self.axiom_debug['obj_axiom'] = -2.0

                return 0

            rel_valid_idx = (rel_target_classes < self.num_rel_classes) & (target_sub_classes < self.num_classes) & (
                    target_obj_classes < self.num_classes) & mask_nms_idx
        else:
            rel_valid_idx = (rel_target_classes < self.num_rel_classes) & (target_sub_classes < self.num_classes) & (
                    target_obj_classes < self.num_classes)

        rel_valid_cls = rel_target_classes[rel_valid_idx]
        sub_valid_cls = target_sub_classes[rel_valid_idx]
        obj_valid_cls = target_obj_classes[rel_valid_idx]

        b_sub_valid_cls = [row[mask_row] for row, mask_row in zip(target_sub_classes, rel_valid_idx)]
        b_obj_valid_cls = [row[mask_row] for row, mask_row in zip(target_obj_classes, rel_valid_idx)]
        b_rel_valid_cls = [row[mask_row] for row, mask_row in zip(rel_target_classes, rel_valid_idx)]

        # sanity check
        assert rel_valid_cls.size()[0] == sub_valid_cls.size()[0], 'number of rel and sub must be equal'
        assert rel_valid_cls.size()[0] == obj_valid_cls.size()[0], 'number of rel and obj must be equal'

        # case eval sub, obj
        if sub_valid_cls.size()[0] > 0 and obj_valid_cls.size()[0] > 0 and rel_valid_cls.size()[0] > 0:

            sub_boxes = outputs['sub_boxes'][rel_valid_idx]
            obj_boxes = outputs['obj_boxes'][rel_valid_idx]
            if not self.train_ltn:
                sub_logits = sub_logits[rel_valid_idx]
                obj_logits = obj_logits[rel_valid_idx]
                rel_logits = rel_logits[rel_valid_idx]
            else:
                sub_logits = outputs['sub_features'][rel_valid_idx]
                obj_logits = outputs['obj_features'][rel_valid_idx]
                rel_logits = outputs['rel_features'][rel_valid_idx]

            cls_s_tensor = torch.hstack([self.ot_e_label[cls].value for cls in sub_valid_cls]).t()
            cls_o_tensor = torch.hstack([self.ot_e_label[cls].value for cls in obj_valid_cls]).t()
            cls_r_tensor = torch.hstack([self.ot_r_label[cls].value for cls in rel_valid_cls]).t()

            if not self.train_ltn:
                is_s = self.sub_is_a(sub_logits[:, :-1], cls_s_tensor)
                is_o = self.obj_is_a(obj_logits[:, :-1], cls_o_tensor)
                is_r = self.rel_is_a(rel_logits[:, :-1], cls_r_tensor)
            else:
                is_s = self.sub_is_a(sub_logits, cls_s_tensor)
                is_o = self.obj_is_a(obj_logits, cls_o_tensor)
                is_r = self.rel_is_a(rel_logits, cls_r_tensor)

            final_sat = []

            # check None as sanity check
            if rel_to_verify is not None and ground_truth_rel is not None and rel_to_verify.size(
                    0) > 0 and ground_truth_rel.size(0) > 0:
                # check if the relationship must be verified

                has_rc = True
                if not self.train_ltn:
                    rel_axiom = self.eval_rel_constr(rel_to_verify, ground_truth_rel, use_log=use_log)
                else:
                    rel_axiom = self.eval_rel_constr(rel_feat_to_verify, ground_truth_rel, use_log=use_log)
                final_sat.append(rel_axiom)

                if self._save_sat_debug:
                    if not self.use_log:
                        self.sat_debug['rel_axiom_sat'] = self.sat_rel_axiom(rel_axiom)
                    else:
                        self.sat_debug['rel_axiom_sat'] = - self.sat_rel_axiom(rel_axiom)
                else:
                    self.sat_debug['rel_axiom_sat'] = -2.0

                if not self.use_log:
                    self.axiom_debug['rel_axiom'] = rel_axiom.value.item()
                else:
                    self.axiom_debug['rel_axiom'] = -rel_axiom.value.item()

            else:
                has_rc = False
                if self._save_sat_debug:
                    self.sat_debug['rel_axiom_sat'] = -2.0
                else:
                    self.sat_debug['rel_axiom_sat'] = -2.0

                self.axiom_debug['rel_axiom'] = -2.0

            if self.ec_enable:
                sub_axiom = self.eval_ent_constr(sub_logits,
                                                   cls_s_tensor,
                                                   use_log=use_log)

                if sub_axiom is not None:
                    final_sat.append(sub_axiom)
                    if self._save_sat_debug:
                        if not self.use_log:
                            self.sat_debug['sub_axiom_sat'] = self.sat_ent_axiom(sub_axiom)
                        else:
                            self.sat_debug['sub_axiom_sat'] = - self.sat_ent_axiom(sub_axiom)
                    else:
                        self.sat_debug['sub_axiom_sat'] = -2.0

                    if not self.use_log:
                        self.axiom_debug['sub_axiom'] = sub_axiom.value.item()
                    else:
                        self.axiom_debug['sub_axiom'] = - sub_axiom.value.item()
                else:
                    self.sat_debug['sub_axiom_sat'] = -2.0

                obj_axiom = self.eval_ent_constr(obj_logits,
                                                       cls_o_tensor,
                                                       use_log=use_log)
                if obj_axiom is not None:
                    final_sat.append(obj_axiom)
                    if self._save_sat_debug:
                        if not self.use_log:
                            self.sat_debug['obj_axiom_sat'] = self.sat_ent_axiom(obj_axiom)
                        else:
                            self.sat_debug['obj_axiom_sat'] = 1 + self.sat_ent_axiom(obj_axiom)
                    else:
                        self.sat_debug['obj_axiom_sat'] = -2.0

                    if not self.use_log:
                        self.axiom_debug['obj_axiom'] = obj_axiom.value.item()
                    else:
                        self.axiom_debug['obj_axiom'] = - obj_axiom.value.item()
                else:
                    self.sat_debug['obj_axiom_sat'] = -2.0


            if self.use_hierarchy:
                # check if the relationship must be verified

                if has_rc:
                    rel_hier_axiom = self.eval_hier_constr(rel_logits,
                                                           cls_r_tensor,
                                                           is_r,
                                                           rel_valid_cls,
                                                           use_log=use_log, s_r_o=1)
                    if rel_hier_axiom is not None:
                        final_sat.append(rel_hier_axiom)
                        if self._save_sat_debug:
                            if not self.use_log:
                                self.sat_debug['rel_hier_axiom_sat'] = self.sat_rel_axiom(rel_hier_axiom)
                            else:
                                self.sat_debug['rel_hier_axiom_sat'] = - self.sat_rel_axiom(rel_hier_axiom)
                        else:
                            self.sat_debug['rel_hier_axiom_sat'] = -2.0
                        if not self.use_log:
                            self.axiom_debug['rel_hier_axiom'] = rel_hier_axiom.value.item()
                        else:
                            self.axiom_debug['rel_hier_axiom'] = - rel_hier_axiom.value.item()
                    else:
                        self.sat_debug['rel_hier_axiom_sat'] = -2.0

                sub_hier_axiom = self.eval_hier_constr(sub_logits,
                                                       cls_s_tensor,
                                                       is_s,
                                                       sub_valid_cls,
                                                       sub_boxes,
                                                       use_log=use_log, s_r_o=0)
                if sub_hier_axiom is not None:
                    final_sat.append(sub_hier_axiom)
                    if self._save_sat_debug:
                        if not self.use_log:
                            self.sat_debug['sub_hier_axiom_sat'] = self.sat_rel_axiom(sub_hier_axiom)
                        else:
                            self.sat_debug['sub_hier_axiom_sat'] = - self.sat_rel_axiom(sub_hier_axiom)
                    else:
                        self.sat_debug['sub_hier_axiom_sat'] = -2.0

                    if not self.use_log:
                        self.axiom_debug['sub_hier_axiom'] = sub_hier_axiom.value.item()
                    else:
                        self.axiom_debug['sub_hier_axiom'] = - sub_hier_axiom.value.item()
                else:
                    self.sat_debug['sub_hier_axiom_sat'] = -2.0

                obj_hier_axiom = self.eval_hier_constr(obj_logits,
                                                       cls_o_tensor,
                                                       is_o,
                                                       obj_valid_cls,
                                                       obj_boxes,
                                                       use_log=use_log, s_r_o=2)
                if obj_hier_axiom is not None:
                    final_sat.append(obj_hier_axiom)
                    if self._save_sat_debug:
                        if not self.use_log:
                            self.sat_debug['obj_hier_axiom_sat'] = self.sat_rel_axiom(obj_hier_axiom)
                        else:
                            self.sat_debug['obj_hier_axiom_sat'] = - self.sat_rel_axiom(obj_hier_axiom)
                    else:
                        self.sat_debug['obj_hier_axiom_sat'] = -2.0

                    if not self.use_log:
                        self.axiom_debug['obj_hier_axiom'] = obj_hier_axiom.value.item()
                    else:
                        self.axiom_debug['obj_hier_axiom'] = - obj_hier_axiom.value.item()
                else:
                    self.sat_debug['obj_hier_axiom_sat'] = -2.0

            if not self.use_pair:
                neg_constr_hit_3 = len(list(
                    filter(lambda so: (so[0].item(), so[1].item()) in self.rel_neg_const_map_3.keys(),
                           zip(sub_valid_cls, obj_valid_cls))))

                pos_constr_hit_5 = len(list(
                    filter(lambda so: (so[0].item(), so[1].item()) in self.rel_pos_const_map_5.keys(),
                           zip(sub_valid_cls, obj_valid_cls))))

                neg_constr_hit_4 = len(
                    list(filter(lambda s: s.item() in self.rel_neg_const_map_4.keys(), rel_valid_cls)))

                pos_constr_hit_2 = len(
                    list(filter(lambda s: s.item() in self.rel_pos_const_map_2.keys(), rel_valid_cls)))

                self.axiom_debug['neg3_axiom_hit'] = neg_constr_hit_3
                self.axiom_debug['neg4_axiom_hit'] = neg_constr_hit_4
                self.axiom_debug['pos2_axiom_hit'] = pos_constr_hit_2
                self.axiom_debug['pos5_axiom_hit'] = pos_constr_hit_5

                b_neg_constr_hit_3 = 0

                b_pos_constr_hit_5 = 0

                b_neg_constr_hit_4 = 0

                b_pos_constr_hit_2 = 0


                for i in range(len(b_sub_valid_cls)):
                    b_neg_constr_hit_3 = b_neg_constr_hit_3 + len(list(
                        filter(lambda so: (so[0].item(), so[1].item()) in self.rel_neg_const_map_3.keys(),
                               zip(b_sub_valid_cls[i], b_obj_valid_cls[i]))))

                    b_pos_constr_hit_5 = b_pos_constr_hit_5 + len(list(
                        filter(lambda so: (so[0].item(), so[1].item()) in self.rel_pos_const_map_5.keys(),
                               zip(b_sub_valid_cls[i], b_obj_valid_cls[i]))))

                    b_neg_constr_hit_4 = b_neg_constr_hit_4 + len(
                        list(filter(lambda s: s.item() in self.rel_neg_const_map_4.keys(), b_rel_valid_cls[i])))

                    b_pos_constr_hit_2 = b_pos_constr_hit_2 + len(
                        list(filter(lambda s: s.item() in self.rel_pos_const_map_2.keys(), b_rel_valid_cls[i])))

                self.axiom_debug['neg3_axiom_hit_batch'] = b_neg_constr_hit_3 / len(b_sub_valid_cls)
                self.axiom_debug['neg4_axiom_hit_batch'] = b_neg_constr_hit_4 / len(b_sub_valid_cls)
                self.axiom_debug['pos2_axiom_hit_batch'] = b_pos_constr_hit_2 / len(b_sub_valid_cls)
                self.axiom_debug['pos5_axiom_hit_batch'] = b_pos_constr_hit_5 / len(b_sub_valid_cls)
            else:
                neg_constr_hit_3_s = len(list(
                    filter(lambda s: s.item() in self.rel_neg_const_map_3[0].keys(),
                           sub_valid_cls)))
                neg_constr_hit_3_o = len(list(
                    filter(lambda o: o.item() in self.rel_neg_const_map_3[1].keys(),
                           obj_valid_cls)))
                neg_constr_hit_3 = neg_constr_hit_3_s + neg_constr_hit_3_o

                pos_constr_hit_5_s = len(list(
                    filter(lambda s: s.item() in self.rel_pos_const_map_5[0].keys(),
                           sub_valid_cls)))
                pos_constr_hit_5_o = len(list(
                    filter(lambda o: o.item() in self.rel_pos_const_map_5[1].keys(),
                           obj_valid_cls)))
                pos_constr_hit_5 = pos_constr_hit_5_s + pos_constr_hit_5_o

                neg_constr_hit_4_s = len(list(
                    filter(lambda s: s.item() in self.rel_neg_const_map_4[0].keys(),
                           rel_valid_cls)))
                neg_constr_hit_4_o = len(list(
                    filter(lambda o: o.item() in self.rel_neg_const_map_4[1].keys(),
                           rel_valid_cls)))
                neg_constr_hit_4 = neg_constr_hit_4_s + neg_constr_hit_4_o

                pos_constr_hit_2_s = len(list(
                    filter(lambda s: s.item() in self.rel_pos_const_map_2[0].keys(),
                           rel_valid_cls)))
                pos_constr_hit_2_o = len(list(
                    filter(lambda o: o.item() in self.rel_pos_const_map_2[1].keys(),
                           rel_valid_cls)))
                pos_constr_hit_2 = pos_constr_hit_2_s + pos_constr_hit_2_o

                self.axiom_debug['neg3_s_axiom_hit'] = neg_constr_hit_3_s
                self.axiom_debug['neg4_s_axiom_hit'] = neg_constr_hit_4_s
                self.axiom_debug['pos2_s_axiom_hit'] = pos_constr_hit_2_s
                self.axiom_debug['pos5_s_axiom_hit'] = pos_constr_hit_5_s
                self.axiom_debug['neg3_o_axiom_hit'] = neg_constr_hit_3_o
                self.axiom_debug['neg4_o_axiom_hit'] = neg_constr_hit_4_o
                self.axiom_debug['pos2_o_axiom_hit'] = pos_constr_hit_2_o
                self.axiom_debug['pos5_o_axiom_hit'] = pos_constr_hit_5_o

                b_neg_constr_hit_3_s = 0
                b_neg_constr_hit_3_o = 0

                b_pos_constr_hit_5_s = 0
                b_pos_constr_hit_5_o = 0

                b_neg_constr_hit_4_s = 0
                b_neg_constr_hit_4_o = 0

                b_pos_constr_hit_2_s = 0
                b_pos_constr_hit_2_o = 0

                for i in range(len(b_sub_valid_cls)):
                    b_neg_constr_hit_3_s = b_neg_constr_hit_3_s + len(list(
                        filter(lambda s: s.item() in self.rel_neg_const_map_3[0].keys(),
                               b_sub_valid_cls[i])))
                    b_neg_constr_hit_3_o = b_neg_constr_hit_3_o + len(list(
                        filter(lambda o: o.item() in self.rel_neg_const_map_3[1].keys(),
                               b_obj_valid_cls[i])))

                    b_pos_constr_hit_5_s = b_pos_constr_hit_5_s + len(list(
                        filter(lambda s: s.item() in self.rel_pos_const_map_5[0].keys(),
                               b_sub_valid_cls[i])))
                    b_pos_constr_hit_5_o = b_pos_constr_hit_5_o + len(list(
                        filter(lambda o: o.item() in self.rel_pos_const_map_5[1].keys(),
                               b_obj_valid_cls[i])))

                    b_neg_constr_hit_4_s = b_neg_constr_hit_4_s + len(list(
                        filter(lambda s: s.item() in self.rel_neg_const_map_4[0].keys(),
                               b_rel_valid_cls[i])))
                    b_neg_constr_hit_4_o = b_neg_constr_hit_4_o + len(list(
                        filter(lambda o: o.item() in self.rel_neg_const_map_4[1].keys(),
                               b_rel_valid_cls[i])))

                    b_pos_constr_hit_2_s = b_pos_constr_hit_2_s + len(list(
                        filter(lambda s: s.item() in self.rel_pos_const_map_2[0].keys(),
                               b_rel_valid_cls[i])))
                    b_pos_constr_hit_2_o = b_pos_constr_hit_2_o + len(list(
                        filter(lambda o: o.item() in self.rel_pos_const_map_2[1].keys(),
                               b_rel_valid_cls[i])))

                self.axiom_debug['neg3_s_axiom_hit_batch'] = b_neg_constr_hit_3_s / len(b_sub_valid_cls)
                self.axiom_debug['neg4_s_axiom_hit_batch'] = b_neg_constr_hit_4_s / len(b_sub_valid_cls)
                self.axiom_debug['pos2_s_axiom_hit_batch'] = b_pos_constr_hit_2_s / len(b_sub_valid_cls)
                self.axiom_debug['pos5_s_axiom_hit_batch'] = b_pos_constr_hit_5_s / len(b_sub_valid_cls)
                self.axiom_debug['neg3_o_axiom_hit_batch'] = b_neg_constr_hit_3_o / len(b_sub_valid_cls)
                self.axiom_debug['neg4_o_axiom_hit_batch'] = b_neg_constr_hit_4_o / len(b_sub_valid_cls)
                self.axiom_debug['pos2_o_axiom_hit_batch'] = b_pos_constr_hit_2_o / len(b_sub_valid_cls)
                self.axiom_debug['pos5_o_axiom_hit_batch'] = b_pos_constr_hit_5_o / len(b_sub_valid_cls)

            neg_easy_axioms = []
            neg_hard_axioms = []
            pos_easy_axioms = []
            pos_hard_axioms = []

            if neg_constr_hit_3 > 0 and self.use_neg_constr_3:
                if self.equal_p:
                    # print(neg_axiom_3)
                    if not self.use_pair:
                        neg_axiom_3 = self.eval_neg_constr_v3(sub_boxes, obj_boxes,
                                                              sub_logits, obj_logits, rel_logits,
                                                              sub_valid_cls, obj_valid_cls,
                                                              cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                              is_s, is_o,
                                                              use_log=use_log)

                        final_sat.append(neg_axiom_3)
                        if self._save_sat_debug:
                            if not self.use_log:
                                self.sat_debug['neg3_axiom_sat'] = self.sat_neg_axiom(neg_axiom_3)
                            else:
                                self.sat_debug['neg3_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_3)
                        else:
                            self.sat_debug['neg3_axiom_sat'] = -2.0

                        if not self.use_log:
                            self.axiom_debug['neg3_axiom'] = neg_axiom_3.value.item()
                        else:
                            self.axiom_debug['neg3_axiom'] = - neg_axiom_3.value.item()
                    else:
                        if neg_constr_hit_3_s > 0:
                            neg_axiom_3_s = self.eval_neg_constr_v3_pair(sub_boxes,
                                                                         sub_logits, rel_logits,
                                                                         sub_valid_cls,
                                                                         cls_s_tensor, cls_r_tensor,
                                                                         is_s,
                                                                         use_log=use_log, s_o=0)
                            final_sat.append(neg_axiom_3_s)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['neg3_s_axiom_sat'] = self.sat_neg_axiom(neg_axiom_3_s)
                                else:
                                    self.sat_debug['neg3_s_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_3_s)
                            else:
                                self.sat_debug['neg3_s_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['neg3_s_axiom'] = neg_axiom_3_s.value.item()
                            else:
                                self.axiom_debug['neg3_s_axiom'] = - neg_axiom_3_s.value.item()
                        if neg_constr_hit_3_o > 0:
                            neg_axiom_3_o = self.eval_neg_constr_v3_pair(obj_boxes,
                                                                         obj_logits, rel_logits,
                                                                         obj_valid_cls,
                                                                         cls_o_tensor, cls_r_tensor,
                                                                         is_o,
                                                                         use_log=use_log, s_o=1)
                            final_sat.append(neg_axiom_3_o)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['neg3_o_axiom_sat'] = self.sat_neg_axiom(neg_axiom_3_o)
                                else:
                                    self.sat_debug['neg3_o_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_3_o)
                            else:
                                self.sat_debug['neg3_o_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['neg3_o_axiom'] = neg_axiom_3_o.value.item()
                            else:
                                self.axiom_debug['neg3_o_axiom'] = - neg_axiom_3_o.value.item()
                else:
                    if not self.use_pair:
                        raise ValueError('SNC cannot use constraint differentiation!')

                    neg3_s_axioms = []
                    neg3_o_axioms = []
                    for difficulty, debug_axioms in [(0, neg_easy_axioms), (1, neg_hard_axioms)]:
                        rel_idx = self.get_rel4axiom_type(rel_valid_cls, difficulty)
                        if torch.sum(rel_idx) == 0:
                            continue

                        if self._has_labels_in_constraint_map(sub_valid_cls[rel_idx], self.rel_neg_const_map_3[0]):
                            neg_axiom_3_s = self.eval_neg_constr_v3_pair(
                                sub_boxes[rel_idx], sub_logits[rel_idx], rel_logits[rel_idx],
                                sub_valid_cls[rel_idx], cls_s_tensor[rel_idx], cls_r_tensor[rel_idx],
                                is_s[rel_idx], use_log=use_log, s_o=0, forall_type=difficulty)
                            final_sat.append(neg_axiom_3_s)
                            debug_axioms.append(neg_axiom_3_s)
                            neg3_s_axioms.append(neg_axiom_3_s)

                        if self._has_labels_in_constraint_map(obj_valid_cls[rel_idx], self.rel_neg_const_map_3[1]):
                            neg_axiom_3_o = self.eval_neg_constr_v3_pair(
                                obj_boxes[rel_idx], obj_logits[rel_idx], rel_logits[rel_idx],
                                obj_valid_cls[rel_idx], cls_o_tensor[rel_idx], cls_r_tensor[rel_idx],
                                is_o[rel_idx], use_log=use_log, s_o=1, forall_type=difficulty)
                            final_sat.append(neg_axiom_3_o)
                            debug_axioms.append(neg_axiom_3_o)
                            neg3_o_axioms.append(neg_axiom_3_o)

                    if self._save_sat_debug:
                        self.sat_debug['neg3_s_axiom_sat'] = self.sat_neg_axiom(neg3_s_axioms) if neg3_s_axioms else -2.0
                        self.sat_debug['neg3_o_axiom_sat'] = self.sat_neg_axiom(neg3_o_axioms) if neg3_o_axioms else -2.0

                    self.axiom_debug['neg3_s_axiom'] = self._axiom_debug_value(neg3_s_axioms, self.sat_neg_axiom)
                    self.axiom_debug['neg3_o_axiom'] = self._axiom_debug_value(neg3_o_axioms, self.sat_neg_axiom)

            if neg_constr_hit_4 > 0 and self.use_neg_constr_4:
                if self.equal_p:
                    if not self.use_pair:
                        neg_axiom_4 = self.eval_neg_constr_v4(sub_boxes, obj_boxes,
                                                              sub_logits, obj_logits, rel_logits,
                                                              rel_valid_cls, cls_s_tensor,
                                                              cls_o_tensor, cls_r_tensor,
                                                              is_r, forall_type=0,
                                                              use_log=use_log)
                        final_sat.append(neg_axiom_4)

                        if self._save_sat_debug:
                            if not self.use_log:
                                self.sat_debug['neg4_axiom_sat'] = self.sat_neg_axiom(neg_axiom_4)
                            else:
                                self.sat_debug['neg4_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_4)
                        else:
                            self.sat_debug['neg4_axiom_sat'] = -2.0

                        if not self.use_log:
                            self.axiom_debug['neg4_axiom'] = neg_axiom_4.value.item()
                        else:
                            self.axiom_debug['neg4_axiom'] = - neg_axiom_4.value.item()
                    else:
                        if neg_constr_hit_4_s > 0:
                            neg_axiom_4_s = self.eval_neg_constr_v4_pair(sub_boxes,
                                                                         sub_logits, rel_logits,
                                                                         rel_valid_cls,
                                                                         cls_s_tensor, cls_r_tensor,
                                                                         is_r, forall_type=0,
                                                                         use_log=use_log, s_o=0)
                            final_sat.append(neg_axiom_4_s)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['neg4_s_axiom_sat'] = self.sat_neg_axiom(neg_axiom_4_s)
                                else:
                                    self.sat_debug['neg4_s_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_4_s)
                            else:
                                self.sat_debug['neg4_s_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['neg4_s_axiom'] = neg_axiom_4_s.value.item()
                            else:
                                self.axiom_debug['neg4_s_axiom'] = - neg_axiom_4_s.value.item()
                        if neg_constr_hit_4_o > 0:
                            neg_axiom_4_o = self.eval_neg_constr_v4_pair(obj_boxes,
                                                                         obj_logits, rel_logits, rel_valid_cls,
                                                                         cls_s_tensor, cls_r_tensor,
                                                                         is_r, forall_type=0,
                                                                         use_log=use_log, s_o=1)
                            final_sat.append(neg_axiom_4_o)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['neg4_o_axiom_sat'] = self.sat_neg_axiom(neg_axiom_4_o)
                                else:
                                    self.sat_debug['neg4_o_axiom_sat'] = - self.sat_neg_axiom(neg_axiom_4_o)
                            else:
                                self.sat_debug['neg4_o_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['neg4_o_axiom'] = neg_axiom_4_o.value.item()
                            else:
                                self.axiom_debug['neg4_o_axiom'] = - neg_axiom_4_o.value.item()

                else:
                    if not self.use_pair:
                        rel_easy_idx = self.get_rel4axiom_type(rel_valid_cls, 0)
                        if torch.sum(rel_easy_idx) > 0:
                            rel_easy_n_valid_cls = rel_valid_cls[rel_easy_idx]
                            neg_easy_axiom = self.eval_neg_constr_v4(sub_boxes, obj_boxes,
                                                                     sub_logits, obj_logits, rel_logits,
                                                                     sub_valid_cls, obj_valid_cls, rel_easy_n_valid_cls,
                                                                     forall_type=0, use_log=use_log)
                            final_sat.append(neg_easy_axiom)
                            neg_easy_axioms.append(neg_easy_axiom)
                            n_have_easy = True
                        else:
                            n_have_easy = False

                        rel_hard_idx = self.get_rel4axiom_type(rel_valid_cls, 1)

                        if torch.sum(rel_hard_idx) > 0:
                            rel_hard_n_valid_cls = rel_valid_cls[rel_hard_idx]
                            neg_hard_axiom = self.eval_neg_constr_v4(sub_boxes, obj_boxes,
                                                                     sub_logits, obj_logits, rel_logits,
                                                                     sub_valid_cls, obj_valid_cls, rel_hard_n_valid_cls,
                                                                     forall_type=1, use_log=use_log)
                            final_sat.append(neg_hard_axiom)
                            neg_hard_axioms.append(neg_hard_axiom)
                            n_have_hard = True
                        else:
                            n_have_hard = False

                        if self._save_sat_debug:
                            if n_have_hard and n_have_easy:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_easy_axiom, neg_hard_axiom)

                            elif n_have_hard:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_hard_axiom)

                            elif n_have_easy:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_easy_axiom)
                            else:
                                self.sat_debug['neg_axiom_sat'] = -2.0
                    else:
                        neg4_s_axioms = []
                        neg4_o_axioms = []
                        for difficulty, debug_axioms in [(0, neg_easy_axioms), (1, neg_hard_axioms)]:
                            rel_idx = self.get_rel4axiom_type(rel_valid_cls, difficulty)
                            if torch.sum(rel_idx) == 0:
                                continue

                            rel_cls = rel_valid_cls[rel_idx]
                            if self._has_labels_in_constraint_map(rel_cls, self.rel_neg_const_map_4[0]):
                                neg_axiom_4_s = self.eval_neg_constr_v4_pair(
                                    sub_boxes[rel_idx], sub_logits[rel_idx], rel_logits[rel_idx],
                                    rel_cls, cls_s_tensor[rel_idx], cls_r_tensor[rel_idx], is_r[rel_idx],
                                    forall_type=difficulty, use_log=use_log, s_o=0)
                                final_sat.append(neg_axiom_4_s)
                                debug_axioms.append(neg_axiom_4_s)
                                neg4_s_axioms.append(neg_axiom_4_s)

                            if self._has_labels_in_constraint_map(rel_cls, self.rel_neg_const_map_4[1]):
                                neg_axiom_4_o = self.eval_neg_constr_v4_pair(
                                    obj_boxes[rel_idx], obj_logits[rel_idx], rel_logits[rel_idx],
                                    rel_cls, cls_o_tensor[rel_idx], cls_r_tensor[rel_idx], is_r[rel_idx],
                                    forall_type=difficulty, use_log=use_log, s_o=1)
                                final_sat.append(neg_axiom_4_o)
                                debug_axioms.append(neg_axiom_4_o)
                                neg4_o_axioms.append(neg_axiom_4_o)

                        if self._save_sat_debug:
                            self.sat_debug['neg4_s_axiom_sat'] = self.sat_neg_axiom(neg4_s_axioms) if neg4_s_axioms else -2.0
                            self.sat_debug['neg4_o_axiom_sat'] = self.sat_neg_axiom(neg4_o_axioms) if neg4_o_axioms else -2.0

                        self.axiom_debug['neg4_s_axiom'] = self._axiom_debug_value(neg4_s_axioms, self.sat_neg_axiom)
                        self.axiom_debug['neg4_o_axiom'] = self._axiom_debug_value(neg4_o_axioms, self.sat_neg_axiom)

            if pos_constr_hit_5 > 0 and self.use_pos_constr_5:
                if self.equal_p:
                    if not self.use_pair:
                        pos_axiom_5 = self.eval_pos_constr_v5(sub_boxes, obj_boxes,
                                                              sub_logits, obj_logits, rel_logits,
                                                              sub_valid_cls, obj_valid_cls,
                                                              cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                              is_s, is_o,
                                                              use_log=use_log)

                        final_sat.append(pos_axiom_5)
                        if self._save_sat_debug:
                            if not self.use_log:
                                self.sat_debug['pos5_axiom_sat'] = self.sat_pos_axiom(pos_axiom_5)
                            else:
                                self.sat_debug['pos5_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_5)
                        else:
                            self.sat_debug['pos5_axiom_sat'] = -2.0

                        if not self.use_log:
                            self.axiom_debug['pos5_axiom'] = pos_axiom_5.value.item()
                        else:
                            self.axiom_debug['pos5_axiom'] = 1+pos_axiom_5.value.item()
                    else:
                        if pos_constr_hit_5_s > 0:
                            pos_axiom_5_s = self.eval_pos_constr_v5_pair(sub_boxes,
                                                                         sub_logits, rel_logits,
                                                                         sub_valid_cls,
                                                                         cls_s_tensor, cls_r_tensor,
                                                                         is_s,
                                                                         use_log=use_log, s_o=0)
                            final_sat.append(pos_axiom_5_s)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['pos5_s_axiom_sat'] = self.sat_pos_axiom(pos_axiom_5_s)
                                else:
                                    self.sat_debug['pos5_s_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_5_s)
                            else:
                                self.sat_debug['pos5_s_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['pos5_s_axiom'] = pos_axiom_5_s.value.item()
                            else:
                                self.axiom_debug['pos5_s_axiom'] = 1+ pos_axiom_5_s.value.item()
                        if pos_constr_hit_5_o > 0:
                            pos_axiom_5_o = self.eval_pos_constr_v5_pair(obj_boxes,
                                                                         obj_logits, rel_logits,
                                                                         obj_valid_cls,
                                                                         cls_o_tensor, cls_r_tensor,
                                                                         is_o,
                                                                         use_log=use_log, s_o=1)
                            final_sat.append(pos_axiom_5_o)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['pos5_o_axiom_sat'] = self.sat_pos_axiom(pos_axiom_5_o)
                                else:
                                    self.sat_debug['pos5_o_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_5_o)
                            else:
                                self.sat_debug['pos5_o_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['pos5_o_axiom'] = pos_axiom_5_o.value.item()
                            else:
                                self.axiom_debug['pos5_o_axiom'] = - pos_axiom_5_o.value.item()
                else:
                    if self.use_pair:
                        pos5_s_axioms = []
                        pos5_o_axioms = []
                        for difficulty, debug_axioms in [(0, pos_easy_axioms), (1, pos_hard_axioms)]:
                            rel_idx = self.get_rel4axiom_type(rel_valid_cls, difficulty)
                            if torch.sum(rel_idx) == 0:
                                continue

                            if self._has_labels_in_constraint_map(sub_valid_cls[rel_idx], self.rel_pos_const_map_5[0]):
                                pos_axiom_5_s = self.eval_pos_constr_v5_pair(
                                    sub_boxes[rel_idx], sub_logits[rel_idx], rel_logits[rel_idx],
                                    sub_valid_cls[rel_idx], cls_s_tensor[rel_idx], cls_r_tensor[rel_idx],
                                    is_s[rel_idx], use_log=use_log, s_o=0, forall_type=difficulty)
                                final_sat.append(pos_axiom_5_s)
                                debug_axioms.append(pos_axiom_5_s)
                                pos5_s_axioms.append(pos_axiom_5_s)

                            if self._has_labels_in_constraint_map(obj_valid_cls[rel_idx], self.rel_pos_const_map_5[1]):
                                pos_axiom_5_o = self.eval_pos_constr_v5_pair(
                                    obj_boxes[rel_idx], obj_logits[rel_idx], rel_logits[rel_idx],
                                    obj_valid_cls[rel_idx], cls_o_tensor[rel_idx], cls_r_tensor[rel_idx],
                                    is_o[rel_idx], use_log=use_log, s_o=1, forall_type=difficulty)
                                final_sat.append(pos_axiom_5_o)
                                debug_axioms.append(pos_axiom_5_o)
                                pos5_o_axioms.append(pos_axiom_5_o)

                        if self._save_sat_debug:
                            self.sat_debug['pos5_s_axiom_sat'] = self.sat_pos_axiom(pos5_s_axioms) if pos5_s_axioms else -2.0
                            self.sat_debug['pos5_o_axiom_sat'] = self.sat_pos_axiom(pos5_o_axioms) if pos5_o_axioms else -2.0

                        self.axiom_debug['pos5_s_axiom'] = self._axiom_debug_value(pos5_s_axioms, self.sat_pos_axiom)
                        self.axiom_debug['pos5_o_axiom'] = self._axiom_debug_value(pos5_o_axioms, self.sat_pos_axiom)

            if pos_constr_hit_2 > 0 and self.use_pos_constr_2:
                if self.equal_p:
                    if not self.use_pair:
                        pos_axiom_2 = self.eval_pos_constr_v2(sub_boxes, obj_boxes,
                                                              sub_logits, obj_logits, rel_logits,
                                                              rel_valid_cls,
                                                              cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                              is_r,
                                                              forall_type=0,
                                                              use_log=use_log)

                        final_sat.append(pos_axiom_2)
                        if self._save_sat_debug:
                            if not self.use_log:
                                self.sat_debug['pos2_axiom_sat'] = self.sat_pos_axiom(pos_axiom_2)
                            else:
                                self.sat_debug['pos2_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_2)
                        else:
                            self.sat_debug['pos2_axiom_sat'] = -2.0

                        if not self.use_log:
                            self.axiom_debug['pos2_axiom'] = pos_axiom_2.value.item()
                        else:
                            self.axiom_debug['pos2_axiom'] = - pos_axiom_2.value.item()
                    else:
                        if pos_constr_hit_2_s > 0:
                            pos_axiom_2_s = self.eval_pos_constr_v2_pair(sub_boxes,
                                                                         sub_logits, rel_logits,
                                                                         rel_valid_cls,
                                                                         cls_s_tensor, cls_r_tensor,
                                                                         is_r,
                                                                         forall_type=0,
                                                                         use_log=use_log, s_o=0)
                            final_sat.append(pos_axiom_2_s)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['pos2_s_axiom_sat'] = self.sat_pos_axiom(pos_axiom_2_s)
                                else:
                                    self.sat_debug['pos2_s_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_2_s)
                            else:
                                self.sat_debug['pos2_s_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['pos2_s_axiom'] = pos_axiom_2_s.value.item()
                            else:
                                self.axiom_debug['pos2_s_axiom'] = - pos_axiom_2_s.value.item()

                        if pos_constr_hit_2_o > 0:
                            pos_axiom_2_o = self.eval_pos_constr_v2_pair(obj_boxes,
                                                                         obj_logits, rel_logits,
                                                                         rel_valid_cls,
                                                                         cls_o_tensor, cls_r_tensor,
                                                                         is_r,
                                                                         forall_type=0,
                                                                         use_log=use_log, s_o=1)
                            final_sat.append(pos_axiom_2_o)
                            if self._save_sat_debug:
                                if not self.use_log:
                                    self.sat_debug['pos2_o_axiom_sat'] = self.sat_pos_axiom(pos_axiom_2_o)
                                else:
                                    self.sat_debug['pos2_o_axiom_sat'] = - self.sat_pos_axiom(pos_axiom_2_o)

                            else:
                                self.sat_debug['pos2_o_axiom_sat'] = -2.0

                            if not self.use_log:
                                self.axiom_debug['pos2_o_axiom'] = pos_axiom_2_o.value.item()
                            else:
                                self.axiom_debug['pos2_o_axiom'] = - pos_axiom_2_o.value.item()

                else:
                    if not self.use_pair:
                        rel_easy_idx = self.get_rel4axiom_type(rel_valid_cls, 0)
                        if torch.sum(rel_easy_idx) > 0:
                            rel_easy_p_valid_cls = rel_valid_cls[rel_easy_idx]
                            pos_easy_axiom = self.eval_pos_constr(sub_boxes, obj_boxes,
                                                                  sub_logits, obj_logits, rel_logits,
                                                                  sub_valid_cls, obj_valid_cls, rel_easy_p_valid_cls,
                                                                  forall_type=0, use_log=use_log)
                            final_sat.append(pos_easy_axiom)
                            pos_easy_axioms.append(pos_easy_axiom)
                            p_have_easy = True
                        else:
                            p_have_easy = False

                        rel_hard_idx = self.get_rel4axiom_type(rel_valid_cls, 1)

                        if torch.sum(rel_hard_idx) > 0:
                            rel_hard_p_valid_cls = rel_valid_cls[rel_hard_idx]

                            pos_hard_axiom = self.eval_pos_constr(sub_boxes, obj_boxes,
                                                                  sub_logits, obj_logits, rel_logits,
                                                                  sub_valid_cls, obj_valid_cls, rel_hard_p_valid_cls,
                                                                  forall_type=1, use_log=use_log)
                            final_sat.append(pos_hard_axiom)
                            pos_hard_axioms.append(pos_hard_axiom)
                            p_have_hard = True
                        else:
                            p_have_hard = False

                        if self._save_sat_debug:
                            if p_have_hard and p_have_easy:
                                self.sat_debug['pos_axiom_sat'] = self.sat_pos_axiom(pos_easy_axiom, pos_hard_axiom)

                            elif p_have_hard:
                                self.sat_debug['pos_axiom_sat'] = self.sat_neg_axiom(pos_hard_axiom)

                            elif p_have_easy:
                                self.sat_debug['pos_axiom_sat'] = self.sat_neg_axiom(pos_easy_axiom)
                            else:
                                self.sat_debug['pos_axiom_sat'] = -2.0
                    else:
                        pos2_s_axioms = []
                        pos2_o_axioms = []
                        for difficulty, debug_axioms in [(0, pos_easy_axioms), (1, pos_hard_axioms)]:
                            rel_idx = self.get_rel4axiom_type(rel_valid_cls, difficulty)
                            if torch.sum(rel_idx) == 0:
                                continue

                            rel_cls = rel_valid_cls[rel_idx]
                            if self._has_labels_in_constraint_map(rel_cls, self.rel_pos_const_map_2[0]):
                                pos_axiom_2_s = self.eval_pos_constr_v2_pair(
                                    sub_boxes[rel_idx], sub_logits[rel_idx], rel_logits[rel_idx],
                                    rel_cls, cls_s_tensor[rel_idx], cls_r_tensor[rel_idx], is_r[rel_idx],
                                    forall_type=difficulty, use_log=use_log, s_o=0)
                                final_sat.append(pos_axiom_2_s)
                                debug_axioms.append(pos_axiom_2_s)
                                pos2_s_axioms.append(pos_axiom_2_s)

                            if self._has_labels_in_constraint_map(rel_cls, self.rel_pos_const_map_2[1]):
                                pos_axiom_2_o = self.eval_pos_constr_v2_pair(
                                    obj_boxes[rel_idx], obj_logits[rel_idx], rel_logits[rel_idx],
                                    rel_cls, cls_o_tensor[rel_idx], cls_r_tensor[rel_idx], is_r[rel_idx],
                                    forall_type=difficulty, use_log=use_log, s_o=1)
                                final_sat.append(pos_axiom_2_o)
                                debug_axioms.append(pos_axiom_2_o)
                                pos2_o_axioms.append(pos_axiom_2_o)

                        if self._save_sat_debug:
                            self.sat_debug['pos2_s_axiom_sat'] = self.sat_pos_axiom(pos2_s_axioms) if pos2_s_axioms else -2.0
                            self.sat_debug['pos2_o_axiom_sat'] = self.sat_pos_axiom(pos2_o_axioms) if pos2_o_axioms else -2.0

                        self.axiom_debug['pos2_s_axiom'] = self._axiom_debug_value(pos2_s_axioms, self.sat_pos_axiom)
                        self.axiom_debug['pos2_o_axiom'] = self._axiom_debug_value(pos2_o_axioms, self.sat_pos_axiom)

            if not self.equal_p:
                self.axiom_debug['neg_easy_axiom'] = self._axiom_debug_value(neg_easy_axioms, self.sat_neg_axiom)
                self.axiom_debug['neg_hard_axiom'] = self._axiom_debug_value(neg_hard_axioms, self.sat_neg_axiom)
                self.axiom_debug['pos_easy_axiom'] = self._axiom_debug_value(pos_easy_axioms, self.sat_pos_axiom)
                self.axiom_debug['pos_hard_axiom'] = self._axiom_debug_value(pos_hard_axioms, self.sat_pos_axiom)

            if final_sat:
                sat_aggr += self.sat_aggr(final_sat)

            # sat aggregation on constraint' predicates
            if not (neg_constr_hit_3 > 0 or neg_constr_hit_4 > 0 or pos_constr_hit_2 > 0 or pos_constr_hit_5 > 0):
                if self._save_sat_debug:
                    if not self.use_pair:
                        self.sat_debug['neg3_axiom_sat'] = -2.0
                        self.sat_debug['pos2_axiom_sat'] = -2.0
                        self.sat_debug['neg4_axiom_sat'] = -2.0
                        self.sat_debug['pos5_axiom_sat'] = -2.0
                    else:
                        self.sat_debug['neg3_s_axiom_sat'] = -2.0
                        self.sat_debug['neg3_o_axiom_sat'] = -2.0
                        self.sat_debug['pos2_s_axiom_sat'] = -2.0
                        self.sat_debug['pos2_o_axiom_sat'] = -2.0
                        self.sat_debug['neg4_s_axiom_sat'] = -2.0
                        self.sat_debug['neg4_o_axiom_sat'] = -2.0
                        self.sat_debug['pos5_s_axiom_sat'] = -2.0
                        self.sat_debug['pos5_o_axiom_sat'] = -2.0

                if self.equal_p:
                    if not self.use_pair:
                        self.axiom_debug['neg3_axiom'] = -2.0
                        self.axiom_debug['pos2_axiom'] = -2.0
                        self.axiom_debug['neg4_axiom'] = -2.0
                        self.axiom_debug['pos5_axiom'] = -2.0
                    else:
                        self.axiom_debug['neg3_s_axiom'] = -2.0
                        self.axiom_debug['neg3_o_axiom'] = -2.0
                        self.axiom_debug['pos2_s_axiom'] = -2.0
                        self.axiom_debug['pos2_o_axiom'] = -2.0
                        self.axiom_debug['neg4_s_axiom'] = -2.0
                        self.axiom_debug['neg4_o_axiom'] = -2.0
                        self.axiom_debug['pos5_s_axiom'] = -2.0
                        self.axiom_debug['pos5_o_axiom'] = -2.0

                else:
                    self.axiom_debug['neg_easy_axiom'] = -2.0
                    self.axiom_debug['neg_hard_axiom'] = -2.0
                    self.axiom_debug['pos_easy_axiom'] = -2.0
                    self.axiom_debug['pos_hard_axiom'] = -2.0

            if not self.use_pair:
                self.axiom_debug['neg3_axiom_act'] = neg_constr_hit_3 / sub_logits.size()[0]
                self.axiom_debug['pos2_axiom_act'] = pos_constr_hit_2 / sub_logits.size()[0]
                self.axiom_debug['neg4_axiom_act'] = neg_constr_hit_4 / sub_logits.size()[0]
                self.axiom_debug['pos5_axiom_act'] = pos_constr_hit_5 / sub_logits.size()[0]
                self.axiom_debug['tot_axiom_act'] = (self.axiom_debug['neg3_axiom_act'] + self.axiom_debug[
                    'neg4_axiom_act'] +
                                                     self.axiom_debug['pos2_axiom_act'] + self.axiom_debug[
                                                         'pos5_axiom_act']) / 4
            else:
                self.axiom_debug['neg3_s_axiom_act'] = neg_constr_hit_3_s / sub_logits.size()[0]
                self.axiom_debug['neg3_o_axiom_act'] = neg_constr_hit_3_o / sub_logits.size()[0]
                self.axiom_debug['pos2_s_axiom_act'] = pos_constr_hit_2_s / sub_logits.size()[0]
                self.axiom_debug['pos2_o_axiom_act'] = pos_constr_hit_2_o / sub_logits.size()[0]
                self.axiom_debug['neg4_s_axiom_act'] = neg_constr_hit_4_s / sub_logits.size()[0]
                self.axiom_debug['neg4_o_axiom_act'] = neg_constr_hit_4_o / sub_logits.size()[0]
                self.axiom_debug['pos5_s_axiom_act'] = pos_constr_hit_5_s / sub_logits.size()[0]
                self.axiom_debug['pos5_o_axiom_act'] = pos_constr_hit_5_o / sub_logits.size()[0]
                self.axiom_debug['tot_axiom_act'] = (self.axiom_debug['neg3_s_axiom_act'] + self.axiom_debug[
                    'neg4_s_axiom_act'] +
                                                     self.axiom_debug['pos2_s_axiom_act'] + self.axiom_debug[
                                                         'pos5_s_axiom_act'] + self.axiom_debug['neg3_o_axiom_act'] +
                                                     self.axiom_debug['neg4_o_axiom_act'] +
                                                     self.axiom_debug['pos2_o_axiom_act'] + self.axiom_debug[
                                                         'pos5_o_axiom_act']) / 8

        else:
            if self._save_sat_debug:
                self.sat_debug['rel_axiom_sat'] = -2.0
                self.sat_debug['entity_axiom_sat'] = -2.0
                self.sat_debug['sub_axiom_sat'] = -2.0
                self.sat_debug['obj_axiom_sat'] = -2.0
                if not self.use_pair:
                    self.sat_debug['neg3_axiom_sat'] = -2.0
                    self.sat_debug['pos2_axiom_sat'] = -2.0
                    self.sat_debug['neg4_axiom_sat'] = -2.0
                    self.sat_debug['pos5_axiom_sat'] = -2.0
                else:
                    self.sat_debug['neg3_s_axiom_sat'] = -2.0
                    self.sat_debug['neg3_o_axiom_sat'] = -2.0
                    self.sat_debug['pos2_s_axiom_sat'] = -2.0
                    self.sat_debug['pos2_o_axiom_sat'] = -2.0
                    self.sat_debug['neg4_s_axiom_sat'] = -2.0
                    self.sat_debug['neg4_o_axiom_sat'] = -2.0
                    self.sat_debug['pos5_s_axiom_sat'] = -2.0
                    self.sat_debug['pos5_o_axiom_sat'] = -2.0

            self.axiom_debug['rel_axiom'] = -2.0
            self.axiom_debug['entity_axiom'] = -2.0
            self.axiom_debug['sub_axiom'] = -2.0
            self.axiom_debug['obj_axiom'] = -2.0

            if not self.use_pair:
                if self.equal_p:
                    self.axiom_debug['neg3_axiom'] = -2.0
                    self.axiom_debug['pos2_axiom'] = -2.0
                    self.axiom_debug['neg4_axiom'] = -2.0
                    self.axiom_debug['pos5_axiom'] = -2.0

                else:
                    self.axiom_debug['neg_easy_axiom'] = -2.0
                    self.axiom_debug['neg_hard_axiom'] = -2.0
                    self.axiom_debug['pos_easy_axiom'] = -2.0
                    self.axiom_debug['pos_hard_axiom'] = -2.0
            else:
                self.axiom_debug['neg3_s_axiom'] = -2.0
                self.axiom_debug['neg3_o_axiom'] = -2.0

                self.axiom_debug['pos2_s_axiom'] = -2.0
                self.axiom_debug['pos2_o_axiom'] = -2.0

                self.axiom_debug['neg4_s_axiom'] = -2.0
                self.axiom_debug['neg4_o_axiom'] = -2.0

                self.axiom_debug['pos5_s_axiom'] = -2.0
                self.axiom_debug['pos5_o_axiom'] = -2.0

        return sat_aggr

    def get_rel4axiom_type(self, rel_cls: torch.Tensor, difficulty: int):
        """
        constructs lists of common (easy) or tailed (hard) relationships
        return an index to select the corresponding classes to keep in the list relationships

        rel_cls(torch.Tensor): tensor of classes to be pruned
        difficulty(int): the difficulty of class to retrieve
                    0 -> easy class to respect axioms
                    1 -> hard class to respect axioms
        """

        # dummy initalization, just for code compactness
        valid_idx = torch.zeros(rel_cls.size(), dtype=bool, device=rel_cls.device)

        if difficulty == 0:
            for i, r in enumerate(rel_cls):
                if r in self.rel_cls_easy:
                    valid_idx[i] = True
        elif difficulty == 1:
            for i, r in enumerate(rel_cls):
                if r in self.rel_cls_hard:
                    valid_idx[i] = True

        else:
            raise ValueError(f'relationship\'s axiom difficulty=={difficulty}: it must be either 0 or 1')

        return valid_idx

    @staticmethod
    def _has_labels_in_constraint_map(labels: torch.Tensor, constr_map):
        return any(label.item() in constr_map.keys() for label in labels)

    def _axiom_debug_value(self, axioms, sat_aggr):
        if len(axioms) == 0:
            return -2.0
        if len(axioms) == 1:
            return axioms[0].value.item()
        return sat_aggr(axioms).item()

    def eval_ent_constr(self, ent_logits, ent_gt_cls, use_log=False):
        """
        evaluated relationships constraint phi1 wrt to ground-truth
        """
        if not self.train_ltn:
            if not self.use_log:
                gr_ent = ltn.Variable(f'gr_ent_box', ent_logits[:, :-1])
            else:
                gr_ent = ltn_log.Variable(f'gr_ent_box', ent_logits[:, :-1])
        else:
            if not self.use_log:
                gr_ent = ltn.Variable(f'gr_ent_box', ent_logits)
            else:
                gr_ent = ltn_log.Variable(f'gr_ent_box', ent_logits)
        try:
            if not self.use_log:
                cls_e = ltn.Variable(f'cls_e', torch.hstack([self.ot_e_label[cls].value for cls in ent_gt_cls]).t())
            else:
                cls_e = ltn_log.Variable(f'cls_e', torch.hstack([self.ot_e_label[cls].value for cls in ent_gt_cls]).t())

            if not self.use_log:
                axiom_ent = self.forall_ent_o_rel(
                    ltn.diag(gr_ent, cls_e),
                    self.p_ent_is_a(gr_ent, cls_e)
                )
            else:
                axiom_ent = self.forall_ent_o_rel(
                    ltn_log.diag(gr_ent, cls_e),
                    self.p_ent_is_a(gr_ent, cls_e)
                )
        except Exception as e:
            print(f'EXCEPTION:{e}')

            print(f'rel_gt_cls:{ent_gt_cls}')

            print(f'pytorch seed:{torch.get_rng_state()}')
            raise e

        return axiom_ent

    def eval_rel_constr(self, rel_logits, rel_gt_cls, use_log=False):
        """
        evaluated relationships constraint phi1 wrt to ground-truth
        """
        if not self.train_ltn:
            if not self.use_log:
                gr_rel = ltn.Variable(f'gr_rel_box', rel_logits[:, :-1])
            else:
                gr_rel = ltn_log.Variable(f'gr_rel_box', rel_logits[:, :-1])
        else:
            if not self.use_log:
                gr_rel = ltn.Variable(f'gr_rel_box', rel_logits)
            else:
                gr_rel = ltn_log.Variable(f'gr_rel_box', rel_logits)
        try:
            if not self.use_log:
                cls_r = ltn.Variable(f'cls_r', torch.hstack([self.ot_r_label[cls].value for cls in rel_gt_cls]).t())
            else:
                cls_r = ltn_log.Variable(f'cls_r', torch.hstack([self.ot_r_label[cls].value for cls in rel_gt_cls]).t())

            if not self.use_log:
                axiom_rel = self.forall_ent_o_rel(
                    ltn.diag(gr_rel, cls_r),
                    self.p_rel_is_a(gr_rel, cls_r)
                )
            else:
                axiom_rel = self.forall_ent_o_rel(
                    ltn_log.diag(gr_rel, cls_r),
                    self.p_rel_is_a(gr_rel, cls_r)
                )
        except Exception as e:
            print(f'EXCEPTION:{e}')

            print(f'rel_gt_cls:{rel_gt_cls}')

            print(f'pytorch seed:{torch.get_rng_state()}')
            raise e

        return axiom_rel

    def eval_hier_constr(self, logits, cls_x_tensor, is_x, x_valid_cls, out_ent_box=None, use_log=False, s_r_o=2):
        """
        evaluated the negative constraint axiom phi3 in a fuzzy-fashion
        """

        hier_gr = 2 if s_r_o == 1 else 1

        idx_with_constr = tuple(map(lambda x: torch.tensor(x[0]), filter(
            lambda x: x[1].item() in self.hier_map[self.hier_indices[hier_gr-1]].keys(),
            enumerate(x_valid_cls))))

        if not idx_with_constr:
            return None

        idx_with_constr = torch.stack(idx_with_constr)

        cls_x_tensor = cls_x_tensor[idx_with_constr]

        logits_valid = logits[idx_with_constr]

        if out_ent_box is not None:
            out_ent_box = out_ent_box[idx_with_constr]

        gr_x, mask_idx_tensor, constr_tensor = grounding(logits_valid,
                                                          out_ent_box,
                                                          self.ltn_strategy,
                                                          ent_cls=cls_x_tensor,
                                                          constr_map=self.hier_map, device=self.device,
                                                          constr_list=self.hier_lists, indices=self.hier_indices[hier_gr-1],
                                                          constr_type=2,
                                                          use_log=use_log, hier_gr=hier_gr,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            hier_map = ltn.Variable(f'hier_map', constr_tensor)

            is_x_v = ltn.Variable(f'is_x', is_x[idx_with_constr])
        else:
            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            hier_map = ltn_log.Variable(f'hier_map', constr_tensor)

            is_x_v = ltn_log.Variable(f'is_x', is_x[idx_with_constr])

        if s_r_o == 1:
            if not self.use_log:
                axiom_fuzzy = self.forall_neg(ltn.diag(gr_x, mask_idx, hier_map),
                                              self.p_rel_in_set(gr_x, mask_idx, hier_map, training=True)
                                              )

            else:
                axiom_fuzzy = self.forall_neg(ltn_log.diag(gr_x, mask_idx, hier_map),
                                              self.p_rel_in_set(gr_x, mask_idx, hier_map, training=True)
                                            )
        else:
            if s_r_o == 0:
                if not self.use_log:
                    axiom_fuzzy = self.forall_neg(ltn.diag(gr_x, mask_idx, hier_map),
                                                  self.p_sub_in_set(gr_x, mask_idx, hier_map, training=True)
                                                  )

                else:
                    axiom_fuzzy = self.forall_neg(ltn_log.diag(gr_x, mask_idx, hier_map),
                                                  self.p_ent_in_set(gr_x, mask_idx, hier_map, training=True)
                                              )
            else:
                if not self.use_log:
                    axiom_fuzzy = self.forall_neg(ltn.diag(gr_x, mask_idx, hier_map),
                                                  self.p_obj_in_set(gr_x, mask_idx, hier_map, training=True)
                                                  )

                else:
                    axiom_fuzzy = self.forall_neg(ltn_log.diag(gr_x, mask_idx, hier_map),
                                                  self.p_obj_in_set(gr_x, mask_idx, hier_map, training=True)
                                                  )

        return axiom_fuzzy

    def eval_neg_constr_v3(self, sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits, sub_valid_cls, obj_valid_cls,
                           cls_s_tensor, cls_o_tensor, cls_r_tensor, is_s, is_o, use_log=False):
        """
        evaluated the negative constraint axiom phi3 in a fuzzy-fashion
        """

        idx_with_constr = torch.stack(tuple(map(lambda o: torch.tensor(o[0]), filter(
            lambda so: (so[1][0].item(), so[1][1].item()) in self.rel_neg_const_map_3.keys(),
            enumerate(zip(sub_valid_cls, obj_valid_cls))))))

        cls_s_tensor = cls_s_tensor[idx_with_constr]
        cls_o_tensor = cls_o_tensor[idx_with_constr]
        cls_r_tensor = cls_r_tensor[idx_with_constr]

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]

        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]

        assert cls_s_tensor.size(0) == cls_o_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        _, gr_rel, mask_idx_tensor, constr_tensor = grounding((sub_logits_valid, obj_logits_valid),
                                                              (out_sub_box, out_obj_box),
                                                              self.ltn_strategy, rel_logits_valid,
                                                              (cls_s_tensor, cls_o_tensor), cls_r_tensor,
                                                              self.rel_neg_const_map_3, self.device,
                                                              self.constr_lists, (self.neg_indices[1]),
                                                              constr_type=0,
                                                              use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn.Variable(f'neg_constr', constr_tensor)

            is_s_v = ltn.Variable(f'is_s', is_s[idx_with_constr])
            is_o_v = ltn.Variable(f'is_o', is_o[idx_with_constr])
        else:
            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn_log.Variable(f'neg_constr', constr_tensor)

            is_s_v = ltn_log.Variable(f'is_s', is_s[idx_with_constr])
            is_o_v = ltn_log.Variable(f'is_o', is_o[idx_with_constr])

        if not self.rel_support:
            if not self.use_log:
                axiom_fuzzy = self.forall_neg(ltn.diag(is_s_v, is_o_v, gr_rel, mask_idx, neg_constr),
                                              self.rel_operator(
                                                  self.and_luk(is_s_v, is_o_v),
                                                  self.not_op(
                                                      self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True))
                                              )
                                              )
            else:
                axiom_fuzzy = self.forall_neg(ltn_log.diag(is_s_v, is_o_v, gr_rel, mask_idx, neg_constr),
                                              self.rel_operator(
                                                  self.and_luk(is_s_v, is_o_v),
                                                  self.not_op(
                                                      self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True))
                                              )
                                              )
        else:
            axiom_fuzzy = self.forall_neg(ltn_log.diag(is_s_v, is_o_v, gr_rel, mask_idx, neg_constr),
                                          self.rel_operator(
                                              self.or_op(self.not_op(is_s_v), self.not_op(is_o_v)),
                                              self.not_op(
                                                  self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True))
                                          )
                                          )

        return axiom_fuzzy

    def eval_pos_constr_v5(self, sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits, sub_valid_cls, obj_valid_cls,
                           cls_s_tensor, cls_o_tensor, cls_r_tensor, is_s, is_o, use_log=False):
        """
        evaluated the negative constraint axiom phi5 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda o: torch.tensor(o[0]), filter(
            lambda so: (so[1][0].item(), so[1][1].item()) in self.rel_pos_const_map_5.keys(),
            enumerate(zip(sub_valid_cls, obj_valid_cls))))))

        cls_s_tensor = cls_s_tensor[idx_with_constr]
        cls_o_tensor = cls_o_tensor[idx_with_constr]
        cls_r_tensor = cls_r_tensor[idx_with_constr]

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]

        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]

        assert cls_s_tensor.size(0) == cls_o_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        _, gr_rel, mask_idx_tensor, constr_tensor = grounding((sub_logits_valid, obj_logits_valid),
                                                            (out_sub_box, out_obj_box),
                                                                   self.ltn_strategy, rel_logits_valid,
                                                                   (cls_s_tensor, cls_o_tensor), cls_r_tensor,
                                                                   self.rel_pos_const_map_5, self.device,
                                                                   self.constr_lists, (self.pos_indices[1]),
                                                                   constr_type=0,
                                                                   use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn.Variable(f'pos_constr', constr_tensor)

            is_s_v = ltn.Variable(f'is_s', is_s[idx_with_constr])
            is_o_v = ltn.Variable(f'is_o', is_o[idx_with_constr])
        else:
            cls_r = ltn_log.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn_log.Variable(f'pos_constr', constr_tensor)

            is_s_v = ltn_log.Variable(f'is_s', is_s[idx_with_constr])
            is_o_v = ltn_log.Variable(f'is_o', is_o[idx_with_constr])

        if not self.rel_support:
            if not self.use_log:
                axiom_fuzzy = self.forall_pos(ltn.diag(is_s_v, is_o_v, gr_rel, cls_r, mask_idx, pos_constr),
                                              self.rel_operator(
                                                  self.and_luk(is_s_v, is_o_v),
                                                  self.not_op(
                                                      self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True))
                                              )
                                              )
            else:
                axiom_fuzzy = self.forall_pos(ltn_log.diag(is_s_v, is_o_v, gr_rel, cls_r, mask_idx, pos_constr),
                                              self.rel_operator(
                                                  self.and_luk(is_s_v, is_o_v),
                                                  self.not_op(
                                                      self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True))
                                              )
                                              )
        else:
            axiom_fuzzy = self.forall_pos(ltn_log.diag(is_s_v, is_o_v, gr_rel, cls_r, mask_idx, pos_constr),
                                          self.rel_operator(
                                              self.or_op(self.not_op(is_s_v), self.not_op(is_o_v)),
                                              self.not_op(
                                                  self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True))
                                          )
                                          )

        return axiom_fuzzy

    def eval_pos_constr_v2(self, sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits,
                           rel_valid_cls, cls_s_tensor, cls_o_tensor, cls_r_tensor, is_r, forall_type, use_log=False):
        """
        evaluated the negative constraint axiom phi2 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda o: torch.tensor(o[0]),
                                                filter(lambda r: r[1].item() in self.rel_pos_const_map_2.keys(),
                                                       enumerate(rel_valid_cls)))))

        cls_s_tensor = cls_s_tensor[idx_with_constr]
        cls_o_tensor = cls_o_tensor[idx_with_constr]
        cls_r_tensor = cls_r_tensor[idx_with_constr]

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]

        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]

        assert cls_s_tensor.size(0) == cls_o_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        gr_ent, gr_rel, mask_idx_tensor, pos_contr_tensor = grounding((sub_logits_valid, obj_logits_valid),
                                                                      (out_sub_box, out_obj_box),
                                                                      self.ltn_strategy, rel_logits_valid,
                                                                      (cls_s_tensor, cls_o_tensor), cls_r_tensor,
                                                                      self.rel_pos_const_map_2, self.device,
                                                                      self.constr_lists, self.pos_indices[0],
                                                                      constr_type=1,
                                                                      use_log=use_log,
                                                              train_ltn=self.train_ltn)

        gr_sub, gr_obj = gr_ent

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

            mask_idx_s = ltn.Variable(f'mask_id', mask_idx_tensor[0])
            pos_constr_s = ltn.Variable(f'pos_constr', pos_contr_tensor[0])
            mask_idx_o = ltn.Variable(f'mask_id', mask_idx_tensor[1])
            pos_constr_o = ltn.Variable(f'pos_constr', pos_contr_tensor[1])

            is_r_v = ltn.Variable(f'is_r', is_r[idx_with_constr])
        else:
            cls_r = ltn_log.Variable(f'cls_r', cls_r_tensor)

            mask_idx_s = ltn_log.Variable(f'mask_id', mask_idx_tensor[0])
            pos_constr_s = ltn_log.Variable(f'pos_constr', pos_contr_tensor[0])
            mask_idx_o = ltn_log.Variable(f'mask_id', mask_idx_tensor[1])
            pos_constr_o = ltn_log.Variable(f'pos_constr', pos_contr_tensor[1])

            is_r_v = ltn_log.Variable(f'is_r', is_r[idx_with_constr])

        #
        if forall_type == 0:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = self.forall_pos(ltn.diag(gr_sub, gr_obj, is_r_v,
                                                           mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o),
                                                  self.rel_operator(
                                                      self.rel_is_a_to_p(is_r_v),
                                                      self.and_luk(self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                                                   self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o))
                                                  )
                                                  )
                else:
                    axiom_fuzzy = self.forall_pos(ltn_log.diag(gr_sub, gr_obj, is_r_v,
                                                               mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o),
                                                  self.rel_operator(
                                                      self.rel_is_a_to_p(is_r_v),
                                                      self.and_luk(self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                                                   self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o))
                                                  )
                                                  )
            else:
                axiom_fuzzy = self.forall_pos(ltn_log.diag(gr_sub, gr_obj, is_r_v,
                                                           mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o),
                                              self.rel_operator(
                                                  self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                  self.and_luk(self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                                               self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o))
                                              )
                                              )
        elif forall_type == 1:
            axiom_fuzzy = self.forall_hard(ltn.diag(gr_sub, gr_obj, gr_rel, cls_r,
                                                    mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o),
                                           self.rel_operator(
                                               self.p_rel_is_a(gr_rel, cls_r),
                                               self.and_luk(self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                                            self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o))

                                           )
                                           )
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        return axiom_fuzzy

    def eval_neg_constr_v4(self, sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits,
                           rel_valid_cls, cls_s_tensor, cls_o_tensor, cls_r_tensor, is_r, forall_type, use_log=False):
        """
        evaluated the negative constraint axiom phi4 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda o: torch.tensor(o[0]),
                                                filter(lambda r: r[1].item() in self.rel_neg_const_map_4.keys(),
                                                       enumerate(rel_valid_cls)))))

        cls_s_tensor = cls_s_tensor[idx_with_constr]
        cls_o_tensor = cls_o_tensor[idx_with_constr]
        cls_r_tensor = cls_r_tensor[idx_with_constr]

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]

        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]

        assert cls_s_tensor.size(0) == cls_o_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        gr_ent, gr_rel, mask_idx_tensor, neg_contr_tensor = grounding((sub_logits_valid, obj_logits_valid),
                                                                      (out_sub_box, out_obj_box),
                                                                              self.ltn_strategy, rel_logits_valid,
                                                                              (cls_s_tensor, cls_o_tensor),
                                                                              cls_r_tensor,
                                                                              self.rel_neg_const_map_4, self.device,
                                                                              self.constr_lists, self.neg_indices[0],
                                                                              constr_type=1,
                                                                              use_log=use_log,
                                                              train_ltn=self.train_ltn)
        gr_sub, gr_obj = gr_ent

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'
        if not use_log:
            cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

            mask_idx_s = ltn.Variable(f'mask_id', mask_idx_tensor[0])
            neg_constr_s = ltn.Variable(f'neg_constr', neg_contr_tensor[0])
            mask_idx_o = ltn.Variable(f'mask_id', mask_idx_tensor[1])
            neg_constr_o = ltn.Variable(f'neg_constr', neg_contr_tensor[1])

            is_r_v = ltn.Variable(f'is_r', is_r[idx_with_constr])
        else:
            cls_r = ltn_log.Variable(f'cls_r', cls_r_tensor)

            mask_idx_s = ltn_log.Variable(f'mask_id', mask_idx_tensor[0])
            neg_constr_s = ltn_log.Variable(f'neg_constr', neg_contr_tensor[0])
            mask_idx_o = ltn_log.Variable(f'mask_id', mask_idx_tensor[1])
            neg_constr_o = ltn_log.Variable(f'neg_constr', neg_contr_tensor[1])

            is_r_v = ltn_log.Variable(f'is_r', is_r[idx_with_constr])

        if forall_type == 0:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = self.forall_neg(ltn.diag(gr_sub, gr_obj, is_r_v,
                                                           mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o),
                                                  self.rel_operator(
                                                      self.rel_is_a_to_p(is_r_v),
                                                      self.and_luk(
                                                          self.not_op(
                                                              self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s)),
                                                          self.not_op(
                                                              self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o))
                                                      )
                                                  )
                                                  )
                else:
                    axiom_fuzzy = self.forall_neg(ltn_log.diag(gr_sub, gr_obj, is_r_v,
                                                               mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o),
                                                  self.rel_operator(
                                                      self.rel_is_a_to_p(is_r_v),
                                                      self.and_luk(
                                                          self.not_op(
                                                              self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s)),
                                                          self.not_op(
                                                              self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o))
                                                      )
                                                  )
                                                  )
            else:
                axiom_fuzzy = self.forall_neg(ltn_log.diag(gr_sub, gr_obj, is_r_v,
                                                           mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o),
                                              self.rel_operator(
                                                  self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                  self.and_luk(
                                                      self.not_op(self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s)),
                                                      self.not_op(self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o))
                                                  )
                                              )
                                              )
        elif forall_type == 1:
            axiom_fuzzy = self.forall_hard(ltn.diag(gr_sub, gr_obj, gr_rel, cls_r,
                                                    mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o),
                                           self.rel_operator(
                                               self.p_rel_is_a(gr_rel, cls_r),
                                               self.and_luk(
                                                   self.not_op(self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s)),
                                                   self.not_op(self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o))
                                               )

                                           )
                                           )
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,2]')

        return axiom_fuzzy

    def eval_neg_constr_v3_pair(self, ent_boxes, ent_logits, rel_logits,
                                ent_valid_cls, cls_e_tensor, cls_r_tensor, is_e, use_log=False, s_o=0,
                                forall_type=0):
        """
        evaluated the negative constraint axiom phi3 in a fuzzy-fashion
        """

        idx_with_constr = torch.stack(tuple(map(lambda e: torch.tensor(e[0]),
                                                filter(lambda r: r[1].item() in self.rel_neg_const_map_3[s_o].keys(),
                                                       enumerate(ent_valid_cls)))))

        ent_logits_valid = ent_logits[idx_with_constr]
        cls_e_tensor = cls_e_tensor[idx_with_constr]
        out_ent_box = ent_boxes[idx_with_constr]

        cls_r_tensor = cls_r_tensor[idx_with_constr]

        rel_logits_valid = rel_logits[idx_with_constr]

        assert cls_e_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_e_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        _, gr_rel, mask_idx_tensor, constr_tensor = grounding(ent_logits_valid,
                                                              out_ent_box,
                                                              self.ltn_strategy, rel_logits_valid,
                                                              cls_e_tensor, cls_r_tensor,
                                                              self.rel_neg_const_map_3[s_o], self.device,
                                                              self.constr_lists, self.neg_indices[1][s_o],
                                                              constr_type=0,
                                                              use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn.Variable(f'neg_constr', constr_tensor)

            is_e_v = ltn.Variable(f'is_e', is_e[idx_with_constr])
        else:
            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn_log.Variable(f'neg_constr', constr_tensor)

            is_e_v = ltn_log.Variable(f'is_e', is_e[idx_with_constr])

        if forall_type == 0:
            forall = self.forall_neg
        elif forall_type == 1:
            forall = self.forall_hard
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        if s_o == 0:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = forall(ltn.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                      self.not_op(
                                                          self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True, debug=True)))
                                                  )
                else:
                    axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                      self.not_op(
                                                          self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True)))
                                                  )
            else:
                axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                              self.rel_operator(
                                                  self.not_op(self.ent_is_a_to_p(is_e_v)),
                                                  self.not_op(
                                                      self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True))
                                              )
                                              )
        else:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = forall(ltn.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                      self.not_op(
                                                          self.p_rel_in_set_o(gr_rel, mask_idx, neg_constr, training=True)))
                                                  )
                else:
                    axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                      self.not_op(
                                                          self.p_rel_in_set_o(gr_rel, mask_idx, neg_constr, training=True)))
                                                  )
            else:
                axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, neg_constr),
                                              self.rel_operator(
                                                  self.not_op(self.ent_is_a_to_p(is_e_v)),
                                                  self.not_op(
                                                      self.p_rel_in_set_o(gr_rel, mask_idx, neg_constr, training=True))
                                              )
                                              )

        return axiom_fuzzy

    def eval_pos_constr_v5_pair(self, ent_boxes, ent_logits, rel_logits, ent_valid_cls, cls_e_tensor, cls_r_tensor, is_e, use_log=False,
                                s_o=0, forall_type=0):
        """
        evaluated the negative constraint axiom phi5 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda e: torch.tensor(e[0]),
                                                filter(lambda r: r[1].item() in self.rel_pos_const_map_5[s_o].keys(),
                                                       enumerate(ent_valid_cls)))))

        ent_logits_valid = ent_logits[idx_with_constr]
        cls_e_tensor = cls_e_tensor[idx_with_constr]
        out_ent_box = ent_boxes[idx_with_constr]

        cls_r_tensor = cls_r_tensor[idx_with_constr]

        rel_logits_valid = rel_logits[idx_with_constr]

        assert cls_e_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_e_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        _, gr_rel, mask_idx_tensor, constr_tensor = grounding(ent_logits_valid,
                                                              out_ent_box,
                                                              self.ltn_strategy, rel_logits_valid,
                                                              cls_e_tensor, cls_r_tensor,
                                                              self.rel_pos_const_map_5[s_o], self.device,
                                                              self.constr_lists, self.pos_indices[1][s_o],
                                                              constr_type=0,
                                                              use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn.Variable(f'pos_constr', constr_tensor)

            is_e_v = ltn.Variable(f'is_e', is_e[idx_with_constr])
        else:
            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn_log.Variable(f'pos_constr', constr_tensor)

            is_e_v = ltn_log.Variable(f'is_e', is_e[idx_with_constr])

        if forall_type == 0:
            forall = self.forall_pos
        elif forall_type == 1:
            forall = self.forall_hard
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        if self.use_pair:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = forall(ltn.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                          self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True))
                                                  )
                else:
                    axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                          self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True))
                                                  )
            else:
                axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                              self.rel_operator(
                                                  self.not_op(self.ent_is_a_to_p(is_e_v)),
                                                      self.p_rel_in_set(gr_rel, mask_idx, pos_constr, training=True)
                                              )
                                              )
        else:
            if not self.rel_support:
                if not self.use_log:
                    axiom_fuzzy = forall(ltn.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                          self.p_rel_in_set_o(gr_rel, mask_idx, pos_constr, training=True))
                                                  )
                else:
                    axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.ent_is_a_to_p(is_e_v),
                                                          self.p_rel_in_set_o(gr_rel, mask_idx, pos_constr, training=True))
                                                  )
            else:
                axiom_fuzzy = forall(ltn_log.diag(is_e_v, gr_rel, mask_idx, pos_constr),
                                              self.rel_operator(
                                                  self.not_op(self.ent_is_a_to_p(is_e_v)),
                                                      self.p_rel_in_set_o(gr_rel, mask_idx, pos_constr, training=True)
                                              )
                                              )

        return axiom_fuzzy

    def eval_pos_constr_v2_pair(self, ent_boxes, ent_logits, rel_logits,
                                rel_valid_cls, cls_e_tensor, cls_r_tensor, is_r, forall_type,
                                use_log=False, s_o=0):
        """
        evaluated the negative constraint axiom phi2 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda r: torch.tensor(r[0]),
                                                filter(lambda e: e[1].item() in self.rel_pos_const_map_2[s_o].keys(),
                                                       enumerate(rel_valid_cls)))))

        ent_logits_valid = ent_logits[idx_with_constr]
        cls_e_tensor = cls_e_tensor[idx_with_constr]
        out_ent_box = ent_boxes[idx_with_constr]

        cls_r_tensor = cls_r_tensor[idx_with_constr]

        rel_logits_valid = rel_logits[idx_with_constr]

        assert cls_e_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_e_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        gr_ent, gr_rel, mask_idx_tensor, constr_tensor = grounding(ent_logits_valid,
                                                                   out_ent_box,
                                                              self.ltn_strategy, rel_logits_valid,
                                                              cls_e_tensor, cls_r_tensor,
                                                              self.rel_pos_const_map_2[s_o], self.device,
                                                              self.constr_lists, self.pos_indices[0][s_o],
                                                              constr_type=1,
                                                              use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn.Variable(f'pos_constr', constr_tensor)

            is_r_v = ltn.Variable(f'is_r', is_r[idx_with_constr])
        else:
            cls_r = ltn_log.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            pos_constr = ltn_log.Variable(f'pos_constr', constr_tensor)

            is_r_v = ltn_log.Variable(f'is_r', is_r[idx_with_constr])

        if forall_type == 0:
            forall = self.forall_pos
        elif forall_type == 1:
            forall = self.forall_hard
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        if forall_type in [0, 1]:
            if not self.use_pair:
                if not self.rel_support:
                    if not self.use_log:
                        axiom_fuzzy = forall(ltn.diag(gr_ent, is_r_v,
                                                               mask_idx, pos_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.p_sub_in_set(gr_ent, mask_idx, pos_constr)
                                                      )
                                                      )
                    else:
                        axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                                   mask_idx, pos_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.p_sub_in_set(gr_ent, mask_idx, pos_constr))
                                                      )

                else:
                    axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                               mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                      self.and_luk(self.p_sub_in_set(gr_ent, mask_idx, pos_constr)
                                                  )))
            else:
                if not self.rel_support:
                    if not self.use_log:
                        axiom_fuzzy = forall(ltn.diag(gr_ent, is_r_v,
                                                               mask_idx, pos_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.p_obj_in_set(gr_ent, mask_idx, pos_constr)
                                                      )
                                                      )
                    else:
                        axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                                   mask_idx, pos_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.p_obj_in_set(gr_ent, mask_idx, pos_constr))
                                                      )

                else:
                    axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                               mask_idx, pos_constr),
                                                  self.rel_operator(
                                                      self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                      self.p_obj_in_set(gr_ent, mask_idx, pos_constr)
                                                                   ))

        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        return axiom_fuzzy

    def eval_neg_constr_v4_pair(self, ent_boxes, ent_logits, rel_logits,
                                rel_valid_cls, cls_e_tensor, cls_r_tensor, is_r, forall_type,
                                use_log=False, s_o=0):
        """
        evaluated the negative constraint axiom phi4 in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty,
                           1 hard constraint
        """

        idx_with_constr = torch.stack(tuple(map(lambda r: torch.tensor(r[0]),
                                                filter(lambda e: e[1].item() in self.rel_neg_const_map_4[s_o].keys(),
                                                       enumerate(rel_valid_cls)))))

        ent_logits_valid = ent_logits[idx_with_constr]
        cls_e_tensor = cls_e_tensor[idx_with_constr]
        out_ent_box = ent_boxes[idx_with_constr]

        cls_r_tensor = cls_r_tensor[idx_with_constr]

        rel_logits_valid = rel_logits[idx_with_constr]

        assert cls_e_tensor.size(0) == cls_r_tensor.size(
            0), f'cls_s_tensor size({cls_e_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'

        gr_ent, gr_rel, mask_idx_tensor, constr_tensor = grounding(ent_logits_valid,
                                                                   out_ent_box,
                                                                   self.ltn_strategy, rel_logits_valid,
                                                                   cls_e_tensor, cls_r_tensor,
                                                                   self.rel_neg_const_map_4[s_o], self.device,
                                                                   self.constr_lists, self.neg_indices[0][s_o],
                                                                   constr_type=1,
                                                                   use_log=use_log,
                                                              train_ltn=self.train_ltn)

        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        if not use_log:
            cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn.Variable(f'neg_constr', constr_tensor)

            is_r_v = ltn.Variable(f'is_r', is_r[idx_with_constr])
        else:
            cls_r = ltn_log.Variable(f'cls_r', cls_r_tensor)

            mask_idx = ltn_log.Variable(f'mask_id', mask_idx_tensor)
            neg_constr = ltn_log.Variable(f'neg_constr', constr_tensor)

            is_r_v = ltn_log.Variable(f'is_r', is_r[idx_with_constr])

        if forall_type == 0:
            forall = self.forall_neg
        elif forall_type == 1:
            forall = self.forall_hard
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        if forall_type in [0, 1]:
            if not self.use_pair:
                if not self.rel_support:
                    if not self.use_log:
                        axiom_fuzzy = forall(ltn.diag(gr_ent, is_r_v,
                                                               mask_idx, neg_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.not_op(self.not_op(self.p_sub_in_set(gr_ent, mask_idx, neg_constr)))
                                                      )
                                                      )
                    else:
                        axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                                   mask_idx, neg_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.not_op(self.p_sub_in_set(gr_ent, mask_idx, neg_constr)))
                                                      )

                else:
                    axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                               mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                      self.not_op(self.p_sub_in_set(gr_ent, mask_idx, neg_constr))
                                                                   ))
            else:
                if not self.rel_support:
                    if not self.use_log:
                        axiom_fuzzy = forall(ltn.diag(gr_ent, is_r_v,
                                                               mask_idx, neg_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.not_op(
                                                              self.p_obj_in_set(gr_ent, mask_idx, neg_constr))
                                                      )
                                                      )
                    else:
                        axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                                   mask_idx, neg_constr),
                                                      self.rel_operator(
                                                          self.rel_is_a_to_p(is_r_v),
                                                          self.not_op(self.p_obj_in_set(gr_ent, mask_idx, neg_constr)))
                                                      )

                else:
                    axiom_fuzzy = forall(ltn_log.diag(gr_ent, is_r_v,
                                                               mask_idx, neg_constr),
                                                  self.rel_operator(
                                                      self.not_op(self.rel_is_a_to_p(is_r_v)),
                                                      self.not_op(self.p_obj_in_set(gr_ent, mask_idx, neg_constr))
                                                  ))


        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')

        return axiom_fuzzy

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs
        start_time = time.time()
        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=pred_logits.device)

        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        if not self.enable_ltn or not self.ec_enable:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

            loss_weight = torch.cat(
                (torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2] * 0.5, indices[3] * 0.5), dim=-1)
            losses = {'loss_ce': (loss_ce * loss_weight).sum() / self.empty_weight[target_classes].sum()}
        # print(time.time() - start_time)
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]

        if self.enable_ltn == True:

            rel_logits = outputs['rel_logits']

            rel_target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices[1])])

            idx_2 = rel_target_classes_o < self.num_rel_classes

            if not self.use_ltn_gt:
                sub_class = sub_logits.sigmoid().argmax(-1).squeeze(0)
                obj_class = obj_logits.sigmoid().argmax(-1).squeeze(0)
                rel_class = rel_logits.sigmoid().argmax(-1).squeeze(0)
            else:
                sub_class = target_sub_classes
                obj_class = target_obj_classes

                idx_3 = self._get_src_permutation_idx(indices[1])
                rel_target_classes = torch.full(rel_logits.shape[:2], self.num_rel_classes, dtype=torch.int64,
                                    device=rel_logits.device)
                rel_target_classes[idx_3] = rel_target_classes_o
                rel_class = rel_target_classes

            if not self.use_log:
                losses['sat_agg_error'] = (1 -
                                           self._compute_sat_level(outputs, sub_logits, obj_logits, sub_class,
                                                                   obj_class, rel_class,
                                                                   rel_target_classes_o[idx_2],
                                                                   rel_logits[rel_idx][idx_2], outputs['rel_features'][rel_idx],
                                                                   use_log=self.use_log))
            else:
                losses['sat_agg_error'] = (-
                                           self._compute_sat_level(outputs, sub_logits, obj_logits, sub_class,
                                                                   obj_class, rel_class,
                                                                   rel_target_classes_o[idx_2],
                                                                   rel_logits[rel_idx][idx_2], outputs['rel_features'][rel_idx],
                                                                   use_log=self.use_log))
            # print(time.time() - start_time)
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat(
            [t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat(
            [t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 151 if args.dataset != 'oi' else 289  # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes=num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    if args.enable_ltn:
        # losses.append('constraints_satisfaction')

        weight_dict['sat_agg_error'] = args.sat_loss_coef
        criterion = SetCriterion(model, num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses,
                                 enable_ltn=args.enable_ltn, ltn_strategy=args.prior_strategy,
                                 rel_operator=args.rel_operator,
                                 neg_constr_path=args.neg_constr_path,
                                 pos_constr_path=args.pos_constr_path,
                                 neg_constr_path_s=args.neg_constr_path_s,
                                 pos_constr_path_s=args.pos_constr_path_s,
                                neg_constr_path_o = args.neg_constr_path_o,
                                pos_constr_path_o = args.pos_constr_path_o,
                                 increase_dim_net=args.incr_gr,
                                 p_dropout=args.dropout_gr,
                                 aggr_p=args.aggr_p,
                                 univ_p=args.univ_p,
                                 different_p=None if args.different_p == False else [args.p_easy_constr,
                                                                                     args.p_hard_constr, args.rel_p],
                                 reltr_grounding=args.reltr_grounding,
                                 tgt_as_constr=args.tgt_as_constr,
                                 ec_enable=args.ec_enable,
                                 rc_enable=args.rc_enable,
                                 use_pred_nms=args.nms_sat_enable,
                                 sat_pred_nms=args.nms_sat,
                                 pure_nesy=args.pure_nesy,
                                 use_log=args.use_log,
                                 use_pair=args.use_pair,
                                use_hierarchy = args.use_hierarchy,
                                hier_path = args.hier_path,
                                deactivate_constr={'3': args.constr_3_off, '4': args.constr_4_off,
                                                   '2': args.constr_2_off, '5': args.constr_5_off},
                                 trainable_ltn=args.train_ltn,
                                use_ltn_gt = args.use_ltn_gt,
                                 use_focused=args.use_focused
                                 )
    else:
        criterion = SetCriterion(model, num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses, device=device,
                                 reltr_grounding=args.reltr_grounding)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

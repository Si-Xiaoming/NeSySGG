# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

import torch
import torchvision.transforms as T
from models import build_model

from util.common import (
    get_args_parser,
    constr_version
)

# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--dataset', default='vg')
#
#     # image path
#     parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
#                         help="Path of the test image")
#
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet50', type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")
#
#     # * Transformer
#     parser.add_argument('--enc_layers', default=6, type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=2048, type=int,
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=256, type=int,
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int,
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_entities', default=100, type=int,
#                         help="Number of query slots")
#     parser.add_argument('--num_triplets', default=200, type=int,
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')
#
#     # Loss
#     parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
#                         help="Disables auxiliary decoding losses (loss at each layer)")
#
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth', help='resume from checkpoint')
#     parser.add_argument('--set_cost_class', default=1, type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox', default=5, type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou', default=2, type=float,
#                         help="giou box coefficient in the matching cost")
#     parser.add_argument('--set_iou_threshold', default=0.7, type=float,
#                         help="giou box coefficient in the matching cost")
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument('--rel_loss_coef', default=1, type=float)
#     parser.add_argument('--eos_coef', default=0.1, type=float,
#                         help="Relative classification weight of the no-object class")
#
#
#     # distributed training parameters
#     parser.add_argument('--return_interm_layers', action='store_true',
#                         help="Return the fpn if there is the tag")
#     return parser


def main(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    neg_constr_path = 'constraint/prompted_positives_vg_final_neg_full.csv'
    pos_constr_path = 'constraint/prompted_positives_vg_final_pos_full.csv'
    neg_constr_path_s = 'constraint/domain_neg_constr4.csv'
    pos_constr_path_s = 'constraint/domain_pos_constr4.csv'
    neg_constr_path_o = 'constraint/range_neg_constr4.csv'
    pos_constr_path_o = 'constraint/range_pos_constr4.csv'

    if 'pair' not in args.resume:
        neg_indices = [("phi4_s", "phi4_o"), "phi3"]
        pos_indices = [("phi2_s", "phi2_o"), "phi5"]
        rel_pos_const_map_2 = _parse_r2so_constraint(pos_constr_path, pos_indices[0])
        rel_neg_const_map_4 = _parse_r2so_constraint(neg_constr_path, neg_indices[0])

        rel_pos_const_map_5 = _parse_so2r_constraint(pos_constr_path, pos_indices[1])
        rel_neg_const_map_3 = _parse_so2r_constraint(neg_constr_path, neg_indices[1])

    else:
        neg_indices = [("phi4_s", "phi4_o"), ("phi3_s", "phi3_o")]
        pos_indices = [("phi2_s", "phi2_o"), ("phi5_s", "phi5_o")]
        rel_pos_const_map_2 = _parse_r2so_constraint_pair((pos_constr_path_s, pos_constr_path_o),
                                                                    pos_indices[0])
        rel_neg_const_map_4 = _parse_r2so_constraint_pair((neg_constr_path_s, neg_constr_path_o),
                                                                    neg_indices[0])

        rel_pos_const_map_5 = _parse_so2r_constraint_pair((pos_constr_path_s, pos_constr_path_o),
                                                                    pos_indices[1])
        rel_neg_const_map_3 = _parse_so2r_constraint_pair((neg_constr_path_s, neg_constr_path_o),
                                                                    neg_indices[1])

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.cpu()
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)

    img_set = 'val'

    ann_path = '/home/odin/PycharmProjects/RelTR/data/vg/' + img_set + '.json'

    with open('/'.join(ann_path.split('/')[:-1]) + '/rel.json', 'r') as f:
        all_rels = json.load(f)

    with open(ann_path, 'r') as f:
        annotations = json.load(f)["annotations"]

    im_id = int(img_path.split('/')[-1].split('.')[0])

    obj_labels = []

    f = False
    for ann in annotations:
        if ann['image_id'] == im_id:
            f = True
            if 'iscrowd' not in ann or ann['iscrowd'] == 0:
                obj_labels.append(ann)
        elif f:
            break

    rels = all_rels[img_set][str(im_id)]

    gt_triplets = [CLASSES[obj_labels[r[0]]['category_id']] + '\n' + REL_CLASSES[r[2]] + '\n' + CLASSES[obj_labels[r[1]]['category_id']] for r in rels]
    gt_bbox_sub = [obj_labels[r[0]]['bbox'] for r in rels]
    gt_bbox_obj = [obj_labels[r[1]]['bbox'] for r in rels]
    gt_triplets_found = [False] * len(gt_triplets)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    img = img

    # propagate through the model
    outputs = model(img)

    threshold = 0.3
    # keep only predictions with 0.+ confidence
    rel_valid_cls = outputs['rel_logits'].sigmoid().argmax(-1).squeeze(0)
    sub_valid_cls = outputs['sub_logits'].sigmoid().argmax(-1).squeeze(0)
    obj_valid_cls = outputs['obj_logits'].sigmoid().argmax(-1).squeeze(0)
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]

    keep = torch.logical_and(probas.max(-1).values > threshold, torch.logical_and(probas_sub.max(-1).values > threshold,
                                                                            probas_obj.max(-1).values > threshold))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]

    keep_queries = keep_queries[indices]

    if 'pair' not in args.resume:
        neg_constr_hit_3 = ['yes' if (so[0].item(), so[1].item()) in rel_neg_const_map_3.keys() else 'no'
                   for so in zip(sub_valid_cls, obj_valid_cls)]

        pos_constr_hit_5 = ['yes' if (so[0].item(), so[1].item()) in rel_pos_const_map_5.keys() else 'no'
                   for so in zip(sub_valid_cls, obj_valid_cls)]

        neg_constr_hit_4 = ['yes' if s.item() in rel_neg_const_map_4.keys() else 'no' for s in  rel_valid_cls]

        pos_constr_hit_2 = ['yes' if s.item() in rel_pos_const_map_2.keys() else 'no' for s in  rel_valid_cls]
    else:
        neg_constr_hit_3_s = ['yes' if s.item() in rel_neg_const_map_3[0].keys() else 'no' for s in
                   sub_valid_cls]
        neg_constr_hit_3_o = ['yes' if o.item() in rel_neg_const_map_3[1].keys() else 'no' for o in
                   obj_valid_cls]
        neg_constr_hit_3 = neg_constr_hit_3_s + neg_constr_hit_3_o

        pos_constr_hit_5_s = ['yes' if s.item() in rel_pos_const_map_5[0].keys() else 'no' for s in
                   sub_valid_cls]
        pos_constr_hit_5_o = ['yes' if o.item() in rel_pos_const_map_5[1].keys() else 'no' for o in
                   obj_valid_cls]
        pos_constr_hit_5 = pos_constr_hit_5_s + pos_constr_hit_5_o

        neg_constr_hit_4_s = ['yes' if s.item() in rel_neg_const_map_4[0].keys() else 'no' for s in
                   rel_valid_cls]
        neg_constr_hit_4_o = ['yes' if o.item() in rel_neg_const_map_4[1].keys() else 'no' for o in
                   rel_valid_cls]
        neg_constr_hit_4 = neg_constr_hit_4_s + neg_constr_hit_4_o

        pos_constr_hit_2_s = ['yes' if s.item() in rel_pos_const_map_2[0].keys() else 'no' for s in
                   rel_valid_cls]
        pos_constr_hit_2_o = ['yes' if o.item() in rel_pos_const_map_2[1].keys() else 'no' for o in
                   rel_valid_cls]
        pos_constr_hit_2 = pos_constr_hit_2_s + pos_constr_hit_2_o

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]

    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        fig1, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(im)

            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            triplet = CLASSES[probas_sub[idx].argmax()] + '\n' + REL_CLASSES[probas[idx].argmax()] + '\n' + CLASSES[
                        probas_obj[idx].argmax()]
            is_gt, gt_idx = check_gt(triplet, (sxmin, symin, sxmax - sxmin, symax - symin),
                     (oxmin, oymin, oxmax - oxmin, oymax - oymin), gt_triplets, gt_bbox_sub, gt_bbox_obj)

            if 'pair' not in args.resume and 'triplet' not in args.resume:
                ax.set_title(
                    triplet,
                    fontsize=10, color="blue" if is_gt else ("red" if gt_idx == -1 else ("yellow" if gt_idx == 1 else "orange")))
            elif 'pair' not in args.resume:
                ax.set_title(triplet+'\nPhi2: '+pos_constr_hit_2[idx]+'\nPhi3: '+neg_constr_hit_3[idx]+'\nPhi4: '+neg_constr_hit_4[idx]+'\nPhi5: '+pos_constr_hit_5[idx]+'', fontsize=10, color="blue" if is_gt else ("red" if gt_idx == -1 else ("yellow" if gt_idx == 1 else "orange")))
            else:
                ax.set_title(
                    triplet + '\nPhi2: s: ' + pos_constr_hit_2_s[idx] + ', o: ' + pos_constr_hit_2_o[idx] + '\nPhi3: s: ' +
                        neg_constr_hit_3_s[idx] + ', o: ' + neg_constr_hit_3_o[idx] + '\nPhi4: s: ' + neg_constr_hit_4_s[idx] + ', o: ' + neg_constr_hit_4_o[idx] + '\nPhi5: s: ' +
                        pos_constr_hit_5_s[idx] + ', o: ' + pos_constr_hit_5_o[idx] + '', fontsize=10, color="blue" if is_gt else ("red" if gt_idx == -1 else ("yellow" if gt_idx == 1 else "orange")))

            if is_gt:
                gt_triplets_found[gt_idx] = True
        fig1.tight_layout()

        # fig1.suptitle('\t\t\t'.join(gt_triplets))


        fig2, axs2 = plt.subplots(ncols=len(gt_triplets), nrows=1, figsize=(22, 7))

        for triplet, found, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(gt_triplets, gt_triplets_found, axs2.T, gt_bbox_sub, gt_bbox_obj):
            ax = ax_i
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax, symax,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax, oymax,
                                       fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            ax.set_title(
                triplet,
                fontsize=10, color="blue" if found else "purple")

        fig2.tight_layout()

        fig2.suptitle('GT')
        plt.show()

def _parse_so2r_constraint(constr_path: str, indices):
    """
    generate constraint  <if sub and obj, then rel > (phi3, phi5) given file where they are stored
    """
    constr = {}

    with open(constr_path, 'r') as fp:
        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines())):
            s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
            if (s, o) not in constr.keys():
                constr[(s, o)] = []

            constr[(s, o)].append(torch.zeros(1))

        # only for compact version
        for k, v in constr.items():
            constr[k] = torch.vstack(v)

    return constr

def _parse_r2so_constraint(constr_path: str, indices):
    """
    generate  <if ent, then rel > (phi2, phi4) given file where they are stored
    idx index used inside the tuple all possible constraint (2 type by construction)
    """
    constr = {}
    used_label = {}

    with open(constr_path, 'r') as fp:
        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines())):
            # save the axiom:  s -> o
            s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
            # save all implication related to s into a map

            if r not in constr.keys():
                constr[r] = {0: [], 1: []}
                used_label[r] = {0: [], 1: []}

            # choose label dependens on agent constraint or action ones

            # check if label not already present in list
            if s not in used_label[r][0]:
                used_label[r][0].append(s)
                constr[r][0].extend([torch.zeros(1)])

            if o not in used_label[r][1]:
                used_label[r][1].append(o)
                constr[r][1].extend([torch.zeros(1)])

        # compact version
        for k, v in constr.items():
            constr[k][0] = torch.vstack(v[0])

    return constr

def _parse_r2so_constraint_pair(constr_path, indices):
    """
    generate  <if rel, then ent > (phi2, phi4) given file where they are stored
    idx index used inside the tuple all possible constraint (2 type by construction)
    """
    constr_s = {}
    constr_o = {}

    constr_path_s, constr_path_o = constr_path

    with open(constr_path_s, 'r') as sr, open(constr_path_o, 'r') as ro:
        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), sr.readlines())):
            s, r = int(tr[0]), int(tr[1])
            if r not in constr_s.keys():
                constr_s[r] = []
            constr_s[r].append(torch.zeros(1))

        # only for compact version
        for k, v in constr_s.items():
            constr_s[k] = torch.vstack(v)

        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), ro.readlines())):
            r, o = int(tr[0]), int(tr[1])
            if r not in constr_o.keys():
                constr_o[r] = []
            constr_o[r].append(torch.zeros(1))

        # only for compact version
        for k, v in constr_o.items():
            constr_o[k] = torch.vstack(v)

    return (constr_s, constr_o)

def _parse_so2r_constraint_pair(constr_path, indices):
    """
    generate constraint  <if sub and obj, then rel > (phi3, phi5) given file where they are stored
    """
    constr_s = {}
    constr_o = {}

    constr_path_s, constr_path_o = constr_path

    with open(constr_path_s, 'r') as sr, open(constr_path_o, 'r') as ro:
        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), sr.readlines())):
            s, r = int(tr[0]), int(tr[1])
            if s not in constr_s.keys():
                constr_s[s] = []
            constr_s[s].append(torch.zeros(1))

        # only for compact version
        for k, v in constr_s.items():
            constr_s[k] = torch.vstack(v)

        for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), ro.readlines())):
            r, o = int(tr[0]), int(tr[1])
            if o not in constr_o.keys():
                constr_o[o] = []
            constr_o[o].append(torch.zeros(1))

        # only for compact version
        for k, v in constr_o.items():
            constr_o[k] = torch.vstack(v)
    return (constr_s, constr_o)

def xywh_to_xyxy(box):
    """(x, y, w, h) -> (xmin, ymin, xmax, ymax)"""
    x, y, w, h = box
    return (x, y, x + w, y + h)

def intersection_area_xyxy(a, b):
    """
    a, b: (xmin, ymin, xmax, ymax)
    returns: intersection area (float)
    """
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)

    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)
    print(a)
    print(b)
    print(f"[{xi1}, {yi1}, {xi2}, {yi2}]")
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    area = inter_w * inter_h
    print(area)
    return area

def check_gt(triplet, sub_bbox, obj_bbox, gt_triplets, gt_sub_bboxes, gt_obj_bboxes):
    """
    a, b: (xmin, ymin, xmax, ymax)
    returns: intersection area (float)
    """
    out = (False, -1)

    if triplet in gt_triplets:
        print(triplet + " found in gt")
        keep = [i for i, t in enumerate(gt_triplets) if t == triplet]
        keep_sub = np.array(gt_sub_bboxes)[keep]
        keep_obj = np.array(gt_obj_bboxes)[keep]
        for gt_s_bbox, gt_o_bbox, i in zip(keep_sub, keep_obj, keep):
           if intersection_area_xyxy(sub_bbox, gt_s_bbox) > 0.5 and intersection_area_xyxy(obj_bbox, gt_o_bbox) > 0.5:
               return (True, i)
           elif intersection_area_xyxy(sub_bbox, gt_s_bbox) > 0.5:
               out = (False, 1)
           elif intersection_area_xyxy(obj_bbox, gt_o_bbox) > 0.5:
               out = (False, 2)
        if out[1] == -1:
            out = (False, 3)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

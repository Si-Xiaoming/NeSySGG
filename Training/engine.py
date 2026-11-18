# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable
import numpy as np

import torch
import wandb

from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def verify_ltn_integrity(criterion, current_expected_p, device):
    print(f"\n======== LTN Integrity Check (p={current_expected_p}) ========")
    
    # 我们主要检查 sat_aggr (因为它是 Loss 的核心)
    # 同时也检查一个 Quantifier (如 forall_neg)
    targets = [
        ("sat_aggr", criterion.sat_aggr),
        ("forall_neg", getattr(criterion, "forall_neg", None))
    ]

    # 创建一个固定的测试数据 (在 GPU 上)
    # 使用 [0.1, 0.9] 这样的差异数据，不同 p 值的计算结果差异明显
    # p=2: ((0.1^2 + 0.9^2)/2)^(1/2) ≈ 0.6403
    # p=4: ((0.1^4 + 0.9^4)/2)^(1/4) ≈ 0.7600
    dummy_input = torch.tensor([[0.1, 0.9]], device=device, dtype=torch.float32, requires_grad=True)

    for name, obj in targets:
        if obj is None: continue
        
        print(f"--- Checking {name} ---")
        
        # 1. 深入检查内部 p 属性的存储位置
        internal_op = obj.agg_op if hasattr(obj, 'agg_op') else obj
        if hasattr(internal_op, 'p'):
            real_p = internal_op.p
            if isinstance(real_p, torch.Tensor):
                print(f"  [Attr Check] internal.p is Tensor on: {real_p.device}")
                print(f"  [Attr Check] internal.p value: {real_p.item()}")
                
                # 严重错误检查：p 值不在 GPU 上
                if "cuda" not in str(real_p.device):
                    print("  ❌ CRITICAL: Parameter p is NOT on GPU!")
            else:
                print(f"  [Attr Check] internal.p is Python Scalar: {real_p}")
        
        # 2. 运行一次实际计算 (Forward Check)
        try:
            # SatAgg 和 Quantifier 调用方式可能略有不同，这里做个适配
            if name == "sat_aggr":
                # SatAgg 期望输入逻辑值的列表或张量
                output = obj(dummy_input)
            else:
                # Quantifier 通常期望 (input, cond)，这里我们只测试其内部 agg_op 以简化
                output = internal_op(dummy_input, dim=1)

            # 3. 验证输出设备 (GPU Check)
            print(f"  [Output Device] Result is on: {output.device}")
            if "cuda" not in str(output.device):
                print("  ❌ CRITICAL: Computation result fell back to CPU!")
            else:
                print("  ✅ Device OK: Computation stayed on GPU.")

            # 4. 验证数值是否随 p 变化 (Math Check)
            # 手动计算预期值 (AggregPMeanError logic: (mean(x^p))^(1/p))
            # 注意：具体公式取决于你用的是 AggregPMean 还是 AggregPMeanError
            # 这里我们只看它是否大致符合预期趋势：p 越大，结果应该越接近 max (对于 PMean) 或 特定值
            val = output.item()
            print(f"  [Math Check] Output Value: {val:.4f}")
            
            # 简单的数学验证：p=2 时约为 0.64，p=4 时约为 0.76 (针对 PMean)
            # 如果是 PMeanError (1 - PMean(1-x))，逻辑反过来
            # 只要这里的值和你预期的一致，就说明新 p 被用上了
            
            # 5. 验证梯度连接 (Grad Check)
            # 这一步证明计算图没有断！
            loss = output.sum()
            loss.backward()
            if dummy_input.grad is not None:
                # 检查梯度是否全 0 (有些算子在特定值梯度为0，但通常不会)
                grad_norm = dummy_input.grad.norm().item()
                if grad_norm > 0:
                    print(f"  ✅ Gradient OK: Input received grad {grad_norm:.4f}")
                else:
                    print(f"  ⚠️ Gradient Warning: Grad is zero (could be normal math, or broken)")
            else:
                print("  ❌ CRITICAL: Gradient broken! (No grad on input)")
            
            # 清理梯度
            dummy_input.grad = None

        except Exception as e:
            print(f"  ❌ Error during probe: {e}")

    print("==============================================\n")


def _get_univ_p_for_debug(criterion):
    forall_neg = getattr(criterion, 'forall_neg', None)
    internal_op = getattr(forall_neg, 'agg_op', None)
    real_p = getattr(internal_op, 'p', None)

    if isinstance(real_p, torch.Tensor):
        return real_p.item()
    if real_p is not None:
        return real_p
    return getattr(criterion, 'univ_p', None)



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, use_pair=False, args=None, wandb_run=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    if args.enable_ltn:
        metric_logger.add_meter('sat_agg_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sat_agg', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

        if args.ec_enable == True:
            metric_logger.add_meter('sat_entity_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('entity_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('sat_sub_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sub_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('sat_obj_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('obj_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))


        if args.rc_enable == True:
            metric_logger.add_meter('sat_relationship_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('relationship_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

        if args.use_hierarchy == True:
            metric_logger.add_meter('sat_hierarchy_sub_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('hierarchy_sub_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('sat_hierarchy_rel_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('hierarchy_rel_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('sat_hierarchy_obj_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('hierarchy_obj_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

        if not use_pair:
            metric_logger.add_meter('sat_neg3_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_pos2_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_neg4_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_pos5_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))

            metric_logger.add_meter('neg3_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg3_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg3_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            if args.different_p:
                metric_logger.add_meter('neg_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('neg_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

                metric_logger.add_meter('pos_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('pos_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            else:
                metric_logger.add_meter('neg3_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('pos2_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('neg4_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('pos5_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        else:
            metric_logger.add_meter('sat_neg3_s_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_neg3_o_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))

            metric_logger.add_meter('sat_pos2_s_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_pos2_o_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))

            metric_logger.add_meter('sat_neg4_s_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_neg4_o_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))

            metric_logger.add_meter('sat_pos5_s_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))
            metric_logger.add_meter('sat_pos5_o_axiom', utils.SmoothedValue(window_size=1, fmt='{valid_value:.2f}'))

            metric_logger.add_meter('neg3_s_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg3_o_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos2_s_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_o_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg4_s_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_o_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos5_s_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_o_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))


            metric_logger.add_meter('neg3_s_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('neg3_o_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('pos2_s_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('pos2_o_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('neg4_s_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('neg4_o_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('pos5_s_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('pos5_o_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            if args.different_p:
                metric_logger.add_meter('neg_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('neg_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

                metric_logger.add_meter('pos_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
                metric_logger.add_meter('pos_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            metric_logger.add_meter('neg3_s_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg3_o_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos2_s_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_o_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg4_s_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_o_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos5_s_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_o_axiom_hit', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg3_s_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg3_o_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos2_s_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos2_o_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('neg4_s_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('neg4_o_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

            metric_logger.add_meter('pos5_s_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
            metric_logger.add_meter('pos5_o_axiom_hit_batch', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

        metric_logger.add_meter('tot_axiom_activation', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

        p_schedule_list = getattr(args, 'p_schedule_list', None)
        if p_schedule_list is not None:
            # 2. 根据总 Epoch 数自动计算当前应该处于的列表索引
            # 公式：(当前 epoch / 总 epochs) * 列表长度
            stage_index = int((epoch / args.epochs) * len(p_schedule_list))
            
            # 3. 防止索引越界
            stage_index = min(stage_index, len(p_schedule_list) - 1)
            
            # 4. 获取当前阶段的目标 P 值
            target_p = p_schedule_list[stage_index]
            # add epoch 0
            if epoch == 0:
                target_p = p_schedule_list[0]
                
                criterion.update_aggr_p(aggr_p=args.aggr_p, univ_p=target_p)

                # 验证更新是否生效
                # 使用 target_p 进行校验，确保逻辑闭环
                verify_ltn_integrity(criterion, current_expected_p=target_p, device=device)

            # 5. 【关键检测】只在 P 值发生跳变时执行更新和验证
            # 检查 criterion 当前记录的 p 是否与目标 p 不同
            current_univ_p = getattr(criterion, 'univ_p', args.univ_p)
            if current_univ_p != target_p:
                print(f"\n[P-Scheduler] Epoch {epoch}: Switching p-value from {current_univ_p} to {target_p}")
                
                # 执行更新 (务必配合之前提供的 In-place 原地更新版 update_aggr_p)
                criterion.update_aggr_p(aggr_p=args.aggr_p, univ_p=target_p)

                # 验证更新是否生效
                # 使用 target_p 进行校验，确保逻辑闭环
                # verify_ltn_integrity(criterion, current_expected_p=target_p, device=device)
            
           







        criterion.save_sat_debug = True
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if args.enable_ltn and getattr(args, 'p_schedule_list', None) is not None:
        #     print(f"[P-Scheduler] Epoch {epoch} Step {i}: univ_p={_get_univ_p_for_debug(criterion)}")

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start_time = time.time()
        outputs = model(samples)
        # print(time.time() - start_time)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict, device=device)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        if not args.rc_enable:
            metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.enable_ltn:
            if not args.rc_enable:
                metric_logger.update(rel_error=1-criterion.sat_debug['rel_axiom_sat'])
            if args.use_log:
                metric_logger.update(sat_agg_error=loss_dict_reduced['sat_agg_error'])
                metric_logger.update(sat_agg=1-loss_dict_reduced['sat_agg_error'])
            else:
                metric_logger.update(sat_agg_error=loss_dict_reduced['sat_agg_error'])
                metric_logger.update(sat_agg=-loss_dict_reduced['sat_agg_error'])

            if args.ec_enable:
                metric_logger.update(sat_entity_axiom=criterion.sat_debug['entity_axiom_sat'])
                metric_logger.update(entity_axiom_fuzzy=criterion.axiom_debug['entity_axiom'])
                metric_logger.update(sat_sub_axiom=criterion.sat_debug['sub_axiom_sat'])
                metric_logger.update(sub_axiom_fuzzy=criterion.axiom_debug['sub_axiom'])
                metric_logger.update(sat_obj_axiom=criterion.sat_debug['obj_axiom_sat'])
                metric_logger.update(obj_axiom_fuzzy=criterion.axiom_debug['obj_axiom'])

            if args.rc_enable:
                metric_logger.update(sat_relationship_axiom=criterion.sat_debug['rel_axiom_sat'])
                metric_logger.update(relationship_axiom_fuzzy=criterion.axiom_debug['rel_axiom'])

            if args.use_hierarchy == True:
                metric_logger.update(sat_hierarchy_sub_axiom=criterion.sat_debug['sub_hier_axiom_sat'])
                metric_logger.update(hierarchy_sub_axiom_fuzzy=criterion.axiom_debug['sub_hier_axiom'])

                metric_logger.update(sat_hierarchy_rel_axiom=criterion.sat_debug['rel_hier_axiom_sat'])
                metric_logger.update(hierarchy_rel_axiom_fuzzy=criterion.axiom_debug['rel_hier_axiom'])

                metric_logger.update(sat_hierarchy_obj_axiom=criterion.sat_debug['obj_hier_axiom_sat'])
                metric_logger.update(hierarchy_obj_axiom_fuzzy=criterion.axiom_debug['obj_hier_axiom'])

            if not use_pair:
                metric_logger.update(sat_neg3_axiom=criterion.sat_debug['neg3_axiom_sat'])
                metric_logger.update(sat_pos2_axiom=criterion.sat_debug['pos2_axiom_sat'])
                metric_logger.update(sat_neg4_axiom=criterion.sat_debug['neg4_axiom_sat'])
                metric_logger.update(sat_pos5_axiom=criterion.sat_debug['pos5_axiom_sat'])

                metric_logger.update(neg3_axiom_activation=criterion.axiom_debug['neg3_axiom_act'])
                metric_logger.update(pos2_axiom_activation=criterion.axiom_debug['pos2_axiom_act'])
                metric_logger.update(neg4_axiom_activation=criterion.axiom_debug['neg4_axiom_act'])
                metric_logger.update(pos5_axiom_activation=criterion.axiom_debug['pos5_axiom_act'])

                metric_logger.update(neg3_axiom_hit=criterion.axiom_debug['neg3_axiom_hit'])
                metric_logger.update(pos2_axiom_hit=criterion.axiom_debug['pos2_axiom_hit'])
                metric_logger.update(neg4_axiom_hit=criterion.axiom_debug['neg4_axiom_hit'])
                metric_logger.update(pos5_axiom_hit=criterion.axiom_debug['pos5_axiom_hit'])

                metric_logger.update(neg3_axiom_hit_batch=criterion.axiom_debug['neg3_axiom_hit_batch'])
                metric_logger.update(pos2_axiom_hit_batch=criterion.axiom_debug['pos2_axiom_hit_batch'])
                metric_logger.update(neg4_axiom_hit_batch=criterion.axiom_debug['neg4_axiom_hit_batch'])
                metric_logger.update(pos5_axiom_hit_batch=criterion.axiom_debug['pos5_axiom_hit_batch'])

                metric_logger.update(tot_axiom_activation=criterion.axiom_debug['tot_axiom_act'])

                if args.different_p:
                    metric_logger.update(neg_easy_axiom_fuzzy=criterion.axiom_debug['neg_easy_axiom'])
                    metric_logger.update(neg_hard_axiom_fuzzy=criterion.axiom_debug['neg_hard_axiom'])

                    metric_logger.update(pos_easy_axiom_fuzzy=criterion.axiom_debug['pos_easy_axiom'])
                    metric_logger.update(pos_hard_axiom_fuzzy=criterion.axiom_debug['pos_hard_axiom'])

                else:
                    metric_logger.update(neg3_axiom_fuzzy=criterion.axiom_debug['neg3_axiom'])
                    metric_logger.update(pos2_axiom_fuzzy=criterion.axiom_debug['pos2_axiom'])
                    metric_logger.update(neg4_axiom_fuzzy=criterion.axiom_debug['neg4_axiom'])
                    metric_logger.update(pos5_axiom_fuzzy=criterion.axiom_debug['pos5_axiom'])
            else:
                metric_logger.update(sat_neg3_s_axiom=criterion.sat_debug['neg3_s_axiom_sat'])
                metric_logger.update(sat_neg3_o_axiom=criterion.sat_debug['neg3_o_axiom_sat'])

                metric_logger.update(sat_pos2_s_axiom=criterion.sat_debug['pos2_s_axiom_sat'])
                metric_logger.update(sat_pos2_o_axiom=criterion.sat_debug['pos2_o_axiom_sat'])

                metric_logger.update(sat_neg4_s_axiom=criterion.sat_debug['neg4_s_axiom_sat'])
                metric_logger.update(sat_neg4_o_axiom=criterion.sat_debug['neg4_o_axiom_sat'])

                metric_logger.update(sat_pos5_s_axiom=criterion.sat_debug['pos5_s_axiom_sat'])
                metric_logger.update(sat_pos5_o_axiom=criterion.sat_debug['pos5_o_axiom_sat'])

                metric_logger.update(neg3_s_axiom_activation=criterion.axiom_debug['neg3_s_axiom_act'])
                metric_logger.update(neg3_o_axiom_activation=criterion.axiom_debug['neg3_o_axiom_act'])

                metric_logger.update(pos2_s_axiom_activation=criterion.axiom_debug['pos2_s_axiom_act'])
                metric_logger.update(pos2_o_axiom_activation=criterion.axiom_debug['pos2_o_axiom_act'])

                metric_logger.update(neg4_s_axiom_activation=criterion.axiom_debug['neg4_s_axiom_act'])
                metric_logger.update(neg4_o_axiom_activation=criterion.axiom_debug['neg4_o_axiom_act'])

                metric_logger.update(pos5_s_axiom_activation=criterion.axiom_debug['pos5_s_axiom_act'])
                metric_logger.update(pos5_o_axiom_activation=criterion.axiom_debug['pos5_o_axiom_act'])

                metric_logger.update(neg3_s_axiom_fuzzy=criterion.axiom_debug['neg3_s_axiom'])
                metric_logger.update(neg3_o_axiom_fuzzy=criterion.axiom_debug['neg3_o_axiom'])

                metric_logger.update(pos2_s_axiom_fuzzy=criterion.axiom_debug['pos2_s_axiom'])
                metric_logger.update(pos2_o_axiom_fuzzy=criterion.axiom_debug['pos2_o_axiom'])

                metric_logger.update(neg4_s_axiom_fuzzy=criterion.axiom_debug['neg4_s_axiom'])
                metric_logger.update(neg4_o_axiom_fuzzy=criterion.axiom_debug['neg4_o_axiom'])

                metric_logger.update(pos5_s_axiom_fuzzy=criterion.axiom_debug['pos5_s_axiom'])
                metric_logger.update(pos5_o_axiom_fuzzy=criterion.axiom_debug['pos5_o_axiom'])

                if args.different_p:
                    metric_logger.update(neg_easy_axiom_fuzzy=criterion.axiom_debug['neg_easy_axiom'])
                    metric_logger.update(neg_hard_axiom_fuzzy=criterion.axiom_debug['neg_hard_axiom'])

                    metric_logger.update(pos_easy_axiom_fuzzy=criterion.axiom_debug['pos_easy_axiom'])
                    metric_logger.update(pos_hard_axiom_fuzzy=criterion.axiom_debug['pos_hard_axiom'])

                metric_logger.update(neg3_s_axiom_hit=criterion.axiom_debug['neg3_s_axiom_hit'])
                metric_logger.update(neg3_o_axiom_hit=criterion.axiom_debug['neg3_o_axiom_hit'])

                metric_logger.update(pos2_s_axiom_hit=criterion.axiom_debug['pos2_s_axiom_hit'])
                metric_logger.update(pos2_o_axiom_hit=criterion.axiom_debug['pos2_o_axiom_hit'])

                metric_logger.update(neg4_s_axiom_hit=criterion.axiom_debug['neg4_s_axiom_hit'])
                metric_logger.update(neg4_o_axiom_hit=criterion.axiom_debug['neg4_o_axiom_hit'])

                metric_logger.update(pos5_s_axiom_hit=criterion.axiom_debug['pos5_s_axiom_hit'])
                metric_logger.update(pos5_o_axiom_hit=criterion.axiom_debug['pos5_o_axiom_hit'])

                metric_logger.update(neg3_s_axiom_hit_batch=criterion.axiom_debug['neg3_s_axiom_hit_batch'])
                metric_logger.update(neg3_o_axiom_hit_batch=criterion.axiom_debug['neg3_o_axiom_hit_batch'])

                metric_logger.update(pos2_s_axiom_hit_batch=criterion.axiom_debug['pos2_s_axiom_hit_batch'])
                metric_logger.update(pos2_o_axiom_hit_batch=criterion.axiom_debug['pos2_o_axiom_hit_batch'])

                metric_logger.update(neg4_s_axiom_hit_batch=criterion.axiom_debug['neg4_s_axiom_hit_batch'])
                metric_logger.update(neg4_o_axiom_hit_batch=criterion.axiom_debug['neg4_o_axiom_hit_batch'])

                metric_logger.update(pos5_s_axiom_hit_batch=criterion.axiom_debug['pos5_s_axiom_hit_batch'])
                metric_logger.update(pos5_o_axiom_hit_batch=criterion.axiom_debug['pos5_o_axiom_hit_batch'])

            metric_logger.update(tot_axiom_activation=criterion.axiom_debug['tot_axiom_act'])

                # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    average_stats = {k: meter.global_avg_valid for k, meter in metric_logger.meters.items()}
    # wandb_run.log(average_stats)
    print(f"Averaged stats: {', '.join('%s : %.4f' % (k,v) for k, v in average_stats.items())}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args, silent=False, eval=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # initilize evaluator
    # TODO merge evaluation programs
    if args.dataset == 'vg':
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
        if args.eval:
            evaluator_list = []
            for index, name in enumerate(data_loader.dataset.rel_categories):
                if index == 0:
                    continue
                evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        else:
            evaluator_list = None
    else:
        all_results = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict, device=device)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        if args.dataset == 'vg':
            evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)
        else:
            evaluate_rel_batch_oi(outputs, targets, all_results)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if args.dataset == 'vg':
        evaluator['sgdet'].print_stats()
    else:
        task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)

    if args.eval and args.dataset == 'vg':
        calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    average_stats = {k: meter.global_avg_valid for k, meter in metric_logger.meters.items()}
    print(f"Averaged stats: {', '.join('%s : %.4f' % (k,v) for k, v in average_stats.items())}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator, None

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list, background_first=False,  task='sgdet'):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
        if background_first:
            rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)
        else:
            rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry, background_first)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry, background_first)


def evaluate_rel_batch_oi(outputs, targets, all_results):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {'sbj_boxes': sub_bboxes_scaled,
                           'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
                           'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
                           'obj_boxes': obj_bboxes_scaled,
                           'obj_labels': pred_obj_classes.cpu().clone().numpy(),
                           'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                           'prd_scores': rel_scores.cpu().clone().numpy(),
                           'image': str(target['image_id'].item())+'.jpg',
                           'gt_sbj_boxes': gt_sub_boxes,
                           'gt_sbj_labels': gt_sub_labels,
                           'gt_obj_boxes': gt_obj_boxes,
                           'gt_obj_labels': gt_obj_labels,
                           'gt_prd_labels': relation_idx[:, 2]
                           }
        all_results.append(img_result_dict)

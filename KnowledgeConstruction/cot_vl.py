import os
import random
import re
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
TP_SIZE = 2
IMAGE_BASE_DIR = "relation_image_exports"
FEW_SHOT_COUNT = 10 # 每条Prompt的图片数量
MAX_MODEL_LEN = 4096 # 视觉模型需要较大上下文
CHUNK_SIZE = 10000 #  10000   100

# [42, 43, 44, 45, 46]
SEEDS = [42] 

print(f"Loading {MODEL_PATH} ... ")

# 初始化 VLM
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TP_SIZE,
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    limit_mm_per_prompt={"image": FEW_SHOT_COUNT} 
)

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"Loaded VLM {MODEL_PATH} successfully")

# ================= 实体与关系列表 =================
entities = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird',
            'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet',
            'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog',
            'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food',
            'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head',
            'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf',
            'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number',
            'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate',
            'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep',
            'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker',
            'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave',
            'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

relationships = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between",
    "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from",
    "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near",
    "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"
]

# ================= 辅助函数 =================
def parse_info_from_filename(filename):
    sub_match = re.search(r'_sub-[^_]+-([^_]+)_', filename)
    obj_match = re.search(r'_obj-[^_]+-([^_]+)_', filename)
    
    sub = sub_match.group(1).replace('_', ' ') if sub_match else "subject"
    obj = obj_match.group(1).replace('_', ' ') if obj_match else "object"
    return sub, obj

def get_folder_for_relation(rel_name, base_dir):
    safe_rel = rel_name.replace(" ", "_")
    if not os.path.exists(base_dir):
        return None
        
    for folder in os.listdir(base_dir):
        if folder.endswith(f"_{safe_rel}"):
            return os.path.join(base_dir, folder)
    return None

def format_question(sub, rel, obj):
    sub_f = f'a group of {sub}' if sub in ['men', 'people'] else f'a {sub}'
    obj_f = f'a group of {obj}' if obj in ['men', 'people'] else f'a {obj}'
    
    rel_formatted = rel
    if rel != 'and':
        if rel == 'wears': rel_formatted = 'wearing'
        elif rel == 'says': rel_formatted = 'saying'
        elif rel == 'has': rel_formatted = 'have'
        elif rel == 'riding': rel_formatted = 'riding on'
        
        rel_f = 'have' if rel_formatted == 'have' else f'be {rel_formatted}'
        return f"Can {sub_f} {rel_f} {obj_f}?"
    else:
        return f"Can {sub_f} {rel} {obj_f} appear together?"

def generate_auto_cot(sub, rel, obj, is_positive_example=True):
    if is_positive_example:
        return (f"Reasoning: 1) Based on real-world knowledge, a '{sub}' and a '{obj}' are common physical entities. "
                f"2) According to physical laws and spatial possibilities, it is completely normal and realistic for a '{sub}' to be {rel} a '{obj}'. "
                f"3) This is a visually plausible scenario that easily appears in real-world photographs.\nFinal Answer: Yes")
    else:
        return (f"Reasoning: 1) Consider the physical properties, scale, and biological nature of a '{sub}' and a '{obj}'. "
                f"2) In the real world, a '{sub}' cannot logically or physically be {rel} a '{obj}'. "
                f"3) This situation violates basic commonsense and would only appear in CGI, fantasy, or metaphorical contexts.\nFinal Answer: No")

def get_dynamic_few_shots(rel_name, base_dir, num_shots=1):
    folder_path = get_folder_for_relation(rel_name, base_dir)
    if not folder_path:
        return [] 
        
    boxed_imgs = [f for f in os.listdir(folder_path) if f.endswith('_boxed.jpg') or f.endswith('_boxed.png')]
    if not boxed_imgs:
        return []
        
    sampled_imgs = random.sample(boxed_imgs, min(num_shots, len(boxed_imgs)))
    
    shots = []
    for img_name in sampled_imgs:
        img_path = os.path.join(folder_path, img_name)
        sub, obj = parse_info_from_filename(img_name)
        question = format_question(sub, rel_name, obj)
        cot_text = generate_auto_cot(sub, rel_name, obj, is_positive_example=True)
        
        shots.append({
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(img_path)},
                {"type": "text", "text": question}
            ]
        })
        shots.append({
            "role": "assistant",
            "content": cot_text
        })
    return shots


def execute():
    base_messages = """You are an expert in visual commonsense reasoning. 
You will be provided with examples consisting of an image and a question in the form: 'Can a <Subject> be <Relationship> on a <Object>?'.
(Note: In the example images, bounding boxes may be present to help you identify the entities, but you must NOT rely on them for your final logical reasoning).

Your task is to evaluate this question according to commonsense knowledge and visual plausibility.
Determine whether the situation is logically possible in the real world according to commonsense knowledge, and whether this situation could realistically appear in a real photograph or video (i.e., no CGI, no fantasy, no metaphorical interpretations)..
Ignore fictional, metaphorical, symbolic, microscopic, or extremely contrived edge cases unless explicitly stated.
Base your judgment purely on general physical laws, scales, and typical real-world conditions.
Answer step-by-step and end your response strictly with 'Final Answer: Yes' or 'Final Answer: No'."""

    sampling_params = SamplingParams(temperature=0.000, max_tokens=500)
    all_prompts = []
    all_metadata = []
    all_results = [] 

    def process_chunk(prompts, metadata):
        print(f"Submitting {len(prompts)} prompts to the VLM for concurrent inference...")
        outputs = llm.generate(prompts, sampling_params)

        for meta, out in zip(metadata, outputs):
            answer = out.outputs[0].text.strip().lower()
            if 'final answer: yes' in answer:
                meta['Answer'] = 'Yes'
            elif 'final answer: no' in answer:
                meta['Answer'] = 'No'
            else:
                meta['Answer'] = 'Error'

            all_results.append(meta)
        
        prompts.clear()
        metadata.clear()
    
    print("Processing data in chunks...")
    c = 0
    for rel in relationships:
        for ent in entities:
            for ent1 in entities:
                if rel in ['wears', 'says', 'has'] and ent in ['men', 'people']:
                    continue
                c += 1
                question = format_question(ent, rel, ent1)
                
                current_chat = [{"role": "system", "content": base_messages}]
                dynamic_shots = get_dynamic_few_shots(rel, IMAGE_BASE_DIR, num_shots=FEW_SHOT_COUNT)
                
                if dynamic_shots:
                    current_chat.extend(dynamic_shots)
                    
                current_chat.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": question}]
                })
                
                prompt_text = tokenizer.apply_chat_template(current_chat, tokenize=False, add_generation_prompt=True)
                
                for seed in SEEDS:
                    all_prompts.append(prompt_text)
                    all_metadata.append({
                        "ID": c, "Seed": seed, "Sub": ent, "Rel": rel, "Obj": ent1
                    })
                    
                    if len(all_prompts) >= CHUNK_SIZE:
                        process_chunk(all_prompts, all_metadata)

    if len(all_prompts) > 0:
        process_chunk(all_prompts, all_metadata)    

    # ================= 后处理与保存 =================
    print("Post-processing results and saving to CSV...")

    df_all = pd.DataFrame(all_results)
    df_valid = df_all[df_all['Answer'].isin(['Yes', 'No'])]

    # 统计 Yes 的数量
    df_yes_counts = df_valid[df_valid['Answer'] == 'Yes'].groupby(['Sub', 'Rel', 'Obj']).size().reset_index(name='yes_count')
    unique_triplets = df_valid[['Sub', 'Rel', 'Obj']].drop_duplicates()
    df_final = pd.merge(unique_triplets, df_yes_counts, on=['Sub', 'Rel', 'Obj'], how='left').fillna(0)

    # 【核心修复】动态计算多数决阈值
    majority_threshold = (len(SEEDS) // 2) + 1 if len(SEEDS) > 1 else 1

    # 根据动态阈值划分 positive 和 negative
    pos_df = df_final[df_final['yes_count'] >= majority_threshold][['Sub', 'Rel', 'Obj']]
    neg_df = df_final[df_final['yes_count'] < majority_threshold][['Sub', 'Rel', 'Obj']]

    # 导出文件
    os.makedirs("outputs_new", exist_ok=True)
    df_all.to_csv("outputs_new/qwen_triplet_outputs6.csv", index=False)
    pos_df.to_csv("outputs_new/qwen_triplet_outputs6_pos_final_human.csv", index=False)
    neg_df.to_csv("outputs_new/qwen_triplet_outputs6_neg_final_human.csv", index=False)

    print(f"Finish！")
    print(f"当前运行的种子数量 (Voting次数): {len(SEEDS)}")
    print(f"分类阈值 (Majority Threshold): 获得 >= {majority_threshold} 票即视为 Positive")
    print(f"共筛选出 {len(pos_df)} 个正样本，{len(neg_df)} 个负样本。")

if __name__ == "__main__":
    execute()
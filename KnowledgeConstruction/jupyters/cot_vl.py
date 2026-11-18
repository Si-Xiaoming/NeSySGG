import os
import random
import re
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置与模型加载
# ==========================================
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"  # 推荐使用 Qwen2-VL
TP_SIZE = 2  # GPU 数量
IMAGE_BASE_DIR = "relation_image_exports" # 你的根目录名称
FEW_SHOT_COUNT = 1 # 每个问题抽取的 few-shot 图片数量 (建议1-2张防止OOM)

print(f"正在加载多模态模型 {MODEL_PATH} ... ")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TP_SIZE,
    dtype="bfloat16",
    max_model_len=4096,           # 视觉模型需要较大上下文
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    limit_mm_per_prompt={"image": FEW_SHOT_COUNT} # 限制每条Prompt的图片数量
)

# ==========================================
# 2. 测试集与原有的问题格式化逻辑
# ==========================================
test_cases = [
    {"Sub": "car", "Rel": "on", "Obj": "street", "Expected": "Yes"},

    {"Sub": "person", "Rel": "wearing", "Obj": "hat", "Expected": "Yes"},
    {"Sub": "boat", "Rel": "flying in", "Obj": "room", "Expected": "No"},
    {"Sub": "cat", "Rel": "sitting on", "Obj": "table", "Expected": "Yes"},
    {"Sub": "pizza", "Rel": "eating", "Obj": "man", "Expected": "No"},
    {"Sub": "train", "Rel": "parked on", "Obj": "track", "Expected": "Yes"},
    {"Sub": "tree", "Rel": "riding", "Obj": "bike", "Expected": "No"},
    {"Sub": "dog", "Rel": "holding", "Obj": "branch", "Expected": "Yes"},
    {"Sub": "flower", "Rel": "growing on", "Obj": "rock", "Expected": "Yes"},

    {"Sub": "logo", "Rel": "of", "Obj": "banana", "Expected": "Yes"},

    {"Sub": "person", "Rel": "along", "Obj": "towel", "Expected": "No"},
    {"Sub": "vase", "Rel": "sitting on", "Obj": "airplane", "Expected": "No"},
    {"Sub": "bag", "Rel": "across", "Obj": "street", "Expected": "No"},
    {"Sub": "leg", "Rel": "of", "Obj": "branch", "Expected": "No"},
    {"Sub": "laptop", "Rel": "of", "Obj": "screen", "Expected": "No"},
    {"Sub": "girl", "Rel": "attached to", "Obj": "book", "Expected": "No"},
    {"Sub": "man", "Rel": "eating", "Obj": "surfboard", "Expected": "No"},
    {"Sub": "man", "Rel": "says", "Obj": "snowboard", "Expected": "No"},


    {"Sub": "animal", "Rel": "above", "Obj": "ear", "Expected": "No"},           # 讨论过的空间几何错误
    {"Sub": "horse", "Rel": "riding", "Obj": "boy", "Expected": "No"},           # 主客体颠倒：人骑马，马不骑人
    {"Sub": "food", "Rel": "eating", "Obj": "woman", "Expected": "No"},          # 主客体颠倒：食物不能吃人
    {"Sub": "shoe", "Rel": "walking on", "Obj": "sidewalk", "Expected": "No"},   # 拟人化陷阱：鞋子本身不会走
    {"Sub": "elephant", "Rel": "sitting on", "Obj": "chair", "Expected": "No"},  # 物理尺度陷阱：普通椅子承受不了大象
    {"Sub": "cup", "Rel": "holding", "Obj": "elephant", "Expected": "No"},       # 物理尺度陷阱：杯子装不下大象
    {"Sub": "motorcycle", "Rel": "wearing", "Obj": "helmet", "Expected": "No"},  # 拟人化陷阱：机器不穿戴衣物
    {"Sub": "wing", "Rel": "part of", "Obj": "dog", "Expected": "No"},           # 解剖学错误：狗没有翅膀
    {"Sub": "leg", "Rel": "attached to", "Obj": "wheel", "Expected": "No"},      # 生物与机械错乱
    {"Sub": "tree", "Rel": "growing on", "Obj": "dog", "Expected": "No"},        # 生物常识错误
    {"Sub": "boat", "Rel": "parked on", "Obj": "tree", "Expected": "No"},        # 极端边缘场景
    {"Sub": "building", "Rel": "wearing", "Obj": "hat", "Expected": "No"},       # 隐喻/卡通陷阱
    {"Sub": "car", "Rel": "in", "Obj": "box", "Expected": "No"},                 # 尺度陷阱：真车装不进普通纸箱
    {"Sub": "logo", "Rel": "painted on", "Obj": "horse", "Expected": "No"},      # 讨论过的边缘特例

    # ==========================================
    # 补充 18 个 "Yes" (总计达到 25 个 Yes)
    # 包含日常常识，以及防止模型过度保守的“迷惑正确项”
    # ==========================================
    {"Sub": "bird", "Rel": "standing on", "Obj": "cow", "Expected": "Yes"},      # 自然界常见（牛背鹭），测试模型是否敢说Yes
    {"Sub": "plant", "Rel": "growing on", "Obj": "building", "Expected": "Yes"}, # 爬山虎、青苔等
    {"Sub": "sign", "Rel": "says", "Obj": "number", "Expected": "Yes"},          # 语言习惯在视觉中的体现（限速牌）
    {"Sub": "cat", "Rel": "looking at", "Obj": "book", "Expected": "Yes"},       # 猫不认字，但物理上盯着书看是完全合理的
    {"Sub": "dog", "Rel": "covered in", "Obj": "snow", "Expected": "Yes"},       # 视觉常识：狗在雪地里玩耍
    {"Sub": "tree", "Rel": "growing on", "Obj": "hill", "Expected": "Yes"},      # 标准自然场景
    {"Sub": "bag", "Rel": "hanging from", "Obj": "branch", "Expected": "Yes"},   # 包挂在树枝上，视觉合理
    {"Sub": "bird", "Rel": "flying in", "Obj": "room", "Expected": "Yes"},       # 相比船在房间飞(No)，鸟飞进房间是常见意外(Yes)
    {"Sub": "person", "Rel": "riding", "Obj": "surfboard", "Expected": "Yes"},   # 冲浪的正确描述
    {"Sub": "door", "Rel": "of", "Obj": "car", "Expected": "Yes"},               # 从属关系对照
    {"Sub": "snow", "Rel": "covering", "Obj": "car", "Expected": "Yes"},         # 典型的冬季视觉场景
    {"Sub": "boat", "Rel": "on", "Obj": "beach", "Expected": "Yes"},             # 搁浅或停靠的船
    {"Sub": "man", "Rel": "carrying", "Obj": "box", "Expected": "Yes"},          # 典型人类动作
    {"Sub": "clock", "Rel": "on", "Obj": "building", "Expected": "Yes"},         # 钟楼/建筑上的钟
    {"Sub": "umbrella", "Rel": "covering", "Obj": "woman", "Expected": "Yes"},   # 撑伞的物理状态
    {"Sub": "pizza", "Rel": "in", "Obj": "box", "Expected": "Yes"},              # 日常物品包含关系
    {"Sub": "plane", "Rel": "above", "Obj": "mountain", "Expected": "Yes"},      # 正常的宏观空间关系
    {"Sub": "chair", "Rel": "near", "Obj": "table", "Expected": "Yes"}           # 标准室内陈设
]

def format_question(sub, rel, obj):
    sub_f = f'a group of {sub}' if sub in ['men', 'people'] else f'a {sub}'
    obj_f = f'a group of {obj}' if obj in ['men', 'people'] else f'a {obj}'
    
    rel_formatted = rel
    if rel != 'and':
        if rel == 'wears': rel_formatted = 'wearing'
        elif rel == 'says': rel_formatted = 'saying'
        elif rel == 'has': rel_formatted = 'have'
        
        rel_f = 'have' if rel_formatted == 'have' else f'be {rel_formatted}'
        return f"Can {sub_f} {rel_f} {obj_f}?"
    else:
        return f"Can {sub_f} {rel} {obj_f} appear together?"

# ==========================================
# 3. 动态提取与纯逻辑 CoT 生成 (核心引擎)
# ==========================================

# 跨模态解耦：在 CoT 中绝不提“蓝框/橙框”，只谈纯物理常识
def generate_auto_cot(sub, rel, obj, is_positive_example=True):
    if is_positive_example:
        return (f"Reasoning: 1) Based on real-world knowledge, a '{sub}' and a '{obj}' are common physical entities. "
                f"2) According to physical laws and spatial possibilities, it is completely normal and realistic for a '{sub}' to be {rel} a '{obj}'. "
                f"3) This is a visually plausible scenario that easily appears in real-world photographs.\nFinal Answer: Yes")
    else:
        return (f"Reasoning: 1) Consider the physical properties, scale, and biological nature of a '{sub}' and a '{obj}'. "
                f"2) In the real world, a '{sub}' cannot logically or physically be {rel} a '{obj}'. "
                f"3) This situation violates basic commonsense and would only appear in CGI, fantasy, or metaphorical contexts.\nFinal Answer: No")

def parse_info_from_filename(filename):
    # 解析例如: 2365496_rel-15-flying_in_sample-001_sub-12-bird_obj-106-room_boxed.jpg
    sub_match = re.search(r'_sub-[^_]+-([^_]+)_', filename)
    obj_match = re.search(r'_obj-[^_]+-([^_]+)_', filename)
    
    sub = sub_match.group(1).replace('_', ' ') if sub_match else "subject"
    obj = obj_match.group(1).replace('_', ' ') if obj_match else "object"
    return sub, obj

def get_folder_for_relation(rel_name, base_dir):
    # 将 "flying in" 转换为 "flying_in" 并匹配 "15_flying_in"
    safe_rel = rel_name.replace(" ", "_")
    if not os.path.exists(base_dir):
        return None
        
    for folder in os.listdir(base_dir):
        if folder.endswith(f"_{safe_rel}"):
            return os.path.join(base_dir, folder)
    return None


def get_dynamic_few_shots(rel_name, base_dir, num_shots=1):
    folder_path = get_folder_for_relation(rel_name, base_dir)
    if not folder_path:
        return [] # 如果没找到这个关系的文件夹，返回空列表
        
    # 只挑选带有 _boxed.jpg 或 _boxed.png 的图片
    boxed_imgs = [f for f in os.listdir(folder_path) if f.endswith('_boxed.jpg') or f.endswith('_boxed.png')]
    if not boxed_imgs:
        return []
        
    # 随机抽样
    sampled_imgs = random.sample(boxed_imgs, min(num_shots, len(boxed_imgs)))
    
    shots = []
    for img_name in sampled_imgs:
        img_path = os.path.join(folder_path, img_name)
        sub, obj = parse_info_from_filename(img_name)
        question = format_question(sub, rel_name, obj)
        
        # 因为是从 VG 数据集直接提出来的真实存在的三元组图片，必定是合乎物理法则的，所以固定是 Positive Example
        cot_text = generate_auto_cot(sub, rel_name, obj, is_positive_example=True)
        
        # 组装 VLM 格式
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

# ==========================================
# 4. 主推理引擎
# ==========================================

system_prompt_visual = """You are an expert in visual commonsense reasoning. 
You will be provided with examples consisting of an image and a question in the form: 'Can a <Subject> be <Relationship> on a <Object>?'.
(Note: In the example images, bounding boxes may be present to help you identify the entities, but you must NOT rely on them for your final logical reasoning).

Your task is to evaluate this question according to commonsense knowledge and visual plausibility.
Determine whether the situation is logically possible in the real world according to commonsense knowledge, and whether this situation could realistically appear in a real photograph.
Base your judgment purely on general physical laws, scales, and typical real-world conditions.
Answer step-by-step and end your response strictly with 'Final Answer: Yes' or 'Final Answer: No'."""


def run_dynamic_vlm_ablation():
    messages_batch = []
    
    for case in test_cases:
        target_rel = case['Rel']
        question = format_question(case['Sub'], target_rel, case['Obj'])
        
        # 1. 放入 System Prompt
        messages = [{"role": "system", "content": system_prompt_visual}]
        
        # 2. 动态检索并拼接 针对当前关系的 图像 Few-Shot
        dynamic_shots = get_dynamic_few_shots(target_rel, IMAGE_BASE_DIR, num_shots=FEW_SHOT_COUNT)
        if dynamic_shots:
            messages.extend(dynamic_shots)
            
        # 3. 放入真正的测试题（纯文本！这代表了真实世界的无框提问）
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        })
        
        messages_batch.append(messages)
        
    print(f"开始推理，共 {len(messages_batch)} 个问题...")
    
    # 限制模型最多输出500个token，temperature设为0保证稳定推理
    sampling_params = SamplingParams(temperature=0.0, max_tokens=500)
    outputs = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=True)
    
    # 解析结果
    results = []
    for case, out in zip(test_cases, outputs):
        answer = out.outputs[0].text.strip()
        is_correct = case['Expected'].lower() in answer.lower()
        results.append({
            "Relation": case['Rel'],
            "Question": format_question(case['Sub'], case['Rel'], case['Obj']),
            "Expected": case['Expected'],
            "Model_Output": answer,
            "Correct": is_correct
        })
        
    df_results = pd.DataFrame(results)
    accuracy = df_results['Correct'].mean() * 100
    print(f"\nFinish! {len(df_results)} samples' accuracy: {accuracy:.1f}%\n")
    return df_results

# 运行评测
print("=== Exp: Cross-Modality Dynamic CoT ===")
df_final = run_dynamic_vlm_ablation()
display(df_final)
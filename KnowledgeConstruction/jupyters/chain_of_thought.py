import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import warnings
warnings.filterwarnings('ignore')


MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B    meta-llama/Llama-3.1-8B-Instruct
TP_SIZE = 2  # 你的GPU数量

print(f" 正在加载 Tokenizer 和模型 {MODEL_PATH} ... ")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 初始化 vLLM 引擎
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TP_SIZE,
    dtype="bfloat16",
    max_model_len=1024,            # 上下文截断
    gpu_memory_utilization=0.85,  # 显存预占比例
    enforce_eager=True            # 关闭 CUDA Graph 防止 OOM
)


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

# 2. 匹配你原代码中的问题生成逻辑
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

# 3. 定义测试函数
def run_ablation(system_prompt_content, few_shots=[]):
    prompts = []
    
    # 构建所有问题的 Prompt
    for case in test_cases:
        question = format_question(case['Sub'], case['Rel'], case['Obj'])
        
        messages = [{"role": "system", "content": system_prompt_content}]
        messages.extend(few_shots)
        messages.append({"role": "user", "content": question})
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
        
    # vLLM 批量推理 (关闭 tqdm 防止 Jupyter 输出过长)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    # 结果解析与统计
    results = []
    for case, out in zip(test_cases, outputs):
        answer = out.outputs[0].text.strip()
        # 简单判定：如果模型输出包含预期答案（忽略大小写）则判对
        is_correct = case['Expected'].lower() in answer.lower()
        results.append({
            "Question": format_question(case['Sub'], case['Rel'], case['Obj']),
            "Expected": case['Expected'],
            "Model_Output": answer,
            "Correct": is_correct
        })
        
    df_results = pd.DataFrame(results)
    accuracy = df_results['Correct'].mean() * 100
    print(f"Finish!  {len(df_results['Correct'])} samples' accuracy: {accuracy:.1f}%\n")
    return df_results


pd.set_option('display.max_colwidth', None)
# 2. 定义 CoT Prompt
prompt_cot = """You are given a question in the form: 'Can a <Subject> be <Relationship> on a <Object>?'
For each question, analyze the physical and visual plausibility step by step.
Interpret this strictly as a statement about physical and commonsense plausibility in the real world and could appear in a real photo.
Your task is to evaluate this question according to commonsense knowledge and visual plausibility.
For the given question, determine whether the situation is logically possible in the real world according to commonsense knowledge,
and whether this situation could realistically appear in a real photograph or video (i.e., no CGI, no fantasy, no metaphorical interpretations).
Ignore fictional, metaphorical, symbolic, microscopic, or extremely contrived edge cases unless explicitly stated.
Base your judgment on typical real-world conditions.
End your response with 'Final Answer: Yes' or 'Final Answer: No'."""

shots_cot = [
    {"role": "user", "content": "Can an arm be flying in the sky?"},
    {"role": "assistant", "content": "Reasoning: 1) An arm is a biological part of a human body. 2) In the real world, arms cannot detach and fly independently. 3) This would only happen in CGI or fantasy. \nFinal Answer: No"},
    {"role": "user", "content": "Can a car be driving on a road?"},
    {"role": "assistant", "content": "Reasoning: 1) Cars are vehicles designed for roads. 2) This is a standard real-world activity. 3) It is visually plausible. \nFinal Answer: Yes"}
]

# 3. 运行测试（复用你的 run_ablation 函数，但传入新的采样参数）
# 注意：原 run_ablation 内部使用了 sampling_params，我们需要临时替换它
sampling_params = SamplingParams(
    temperature=temperature, 
    max_tokens=500, 
)

print("=== Exp 2: Chain-of-Thought (CoT) ===")
df_res2 = run_ablation(prompt_cot, [])
display(df_res2)


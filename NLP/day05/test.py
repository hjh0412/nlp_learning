from transformers import AutoConfig

config = AutoConfig.from_pretrained("./Qwen3.5-0.8B")
print(config.model_type)  # 应该输出 qwen3_5
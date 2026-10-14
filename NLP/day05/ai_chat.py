from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():

    model_name = "./qwen3.5-0.8b"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_name}, 使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True
    )
    messages = [
        {"role": "system", "content": "你是一个有用的智能助手，回答简洁准确。"}
    ]
    #print("✅ 👤🤖模型加载完成，输入问题开始对话，输入 exit 退出\n")
    while True:
  
        user_prompt = input("✅ 👤🤖模型加载完成，输入问题开始对话，输入 exit 退出\n")
        if user_prompt == 'exit':
            break
        # messages = [
        #     {"role": "system", "content": "你是一个有用的智能助手，回答简洁准确。"},
        #     {"role": "user", "content": "你好"}
        # ]
        messages.append({"role":"user","content":user_prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"🤖 助手：{response}\n")
        messages.append({"role": "assistant", "content": response})
        

if __name__ == "__main__":
    main()
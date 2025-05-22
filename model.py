from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"  # hoặc bất kỳ model nào bạn muốn

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to("cpu")

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# # Nhập prompt
# prompt = "Tính 2 + 2 bằng bao nhiêu?"

# # Mã hóa prompt và chuyển sang tensor
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # Sinh phản hồi
# output = model.generate(**inputs, max_new_tokens=20)


# # Giải mã và in ra phản hồi
# response = tokenizer.decode(output[0], skip_special_tokens=True)
# print(response)
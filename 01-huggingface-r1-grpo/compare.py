# Q: are labels used for anything here?
# can instead maximize rewards to certain type of response? alignment?

from datasets import load_dataset
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer

dataset = load_dataset("mlabonne/smoltldr")

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = (
    "Summarize the following text: Artificial intelligence is transforming the world."
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model output:", output_text)

from transformers import AutoTokenizer
import transformers
import torch

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "/cpfs01/shared/ADLab/hug_ckpts/llama2-13b/"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    # 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    # 'What is the median value of favourable line in the following graph? Year \t Unfavorable \t Favorable \n 2012 \t 65 \t 30 \n 2013 \t 50 \t 40 \n 2014 \t 0 \t 0 \n',
    'give an example of csv table',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
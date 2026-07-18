from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2_model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
prompt = "User: What is an AI?\nAssistant:"
inputs = tokenizer(prompt, return_tensors = "pt")

output = model.generate(
        **inputs,
        max_new_tokens = 50,
        temperature = 0.1,
        do_sample=True,
        top_p = 0.9,
pad_token_id = tokenizer.eos_token_id
)
print(tokenizer.decode(output[0],skip_special_tokens=True))

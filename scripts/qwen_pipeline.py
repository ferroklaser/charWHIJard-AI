from transformers import AutoModelForCausaualLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

#load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
def classify_review_qwen(review_text):
    # prepare model for input
    prompt = "Insert prompt here"
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # for switching between thinking and non thinking modes
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=64
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content

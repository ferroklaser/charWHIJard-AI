from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

#load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
def classify_review_qwen(row):

    row_text = (
        f"Business Name: {row['name']}\n"
        f"Category: {row['category']}\n"
        f"Review Text: {row['text']}\n"
        f"Text Length: {row['text_length']}\n"
        f"Has Pictures: {row['has_pics']}\n"
        f"Has Response from business: {row['has_response']}"
    )

    # prepare model for input
    prompt = """
    You are a review classifier. Use the information in each row to determination the relevance of the review. 

    The review would be automatically irrelevant  if 
    1. It is seems like an advertisement. 
    2. It is spam material. 
    3. If it seems like the one who wrote the text has yet to visit the establishment they are writing about. 
    
    Conversely, the relevance of the review should be based on if content of review is actually related 
    to the establishment and what they sell, service or provide.

    output strictly in this CSV format:
    <relevancy score (0-1)>, <is_advertisement (true/false)>, <is_rant_without_review (true/false)>
    """
    messages = [
    {"role": "user", "content": prompt},
    {"role": "user", "content": f"Classify this review: \"{row_text}\". Only give the label."}
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

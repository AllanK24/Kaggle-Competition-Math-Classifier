from torchinfo import summary

def summarize_model(model, tokenizer, prompt: str):
    """Summarize the model using torchinfo. The prompt is going to be tokenized
    and passed to the model. The summary will include input size, output size,
    number of parameters, and trainable parameters.
    The summary will be printed in a table format.

    Args:
        model: The model to be summarized.
        prompt (str): The input prompt for the model.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")
    
    # assume `inputs` is your BatchEncoding
    plain_inputs = {
        "input_ids":    inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    print(summary(
        model=model,
        input_data=plain_inputs,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
    ))
from torchinfo import summary

def summarize_model(model, inputs: dict):
    """Summarize the model using torchinfo.

    Args:
        model: The model to be summarized.
        inputs (dict): Tokenized inputs for the model.
    """
    
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
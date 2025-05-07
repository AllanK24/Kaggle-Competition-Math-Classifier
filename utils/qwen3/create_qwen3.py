import torch
from torch import nn
from peft import LoraConfig
from utils.qwen3.qwen3_classifier_class import Qwen3Classifier
from transformers import AutoConfig, AutoTokenizer, AutoModel

def create_qwen3_classifier(
    model_id:str,
    num_classes:int=8,
    freeze_norm_layer: bool=False,
    freeze_embedding: bool=True,
    num_decoder_layers_to_unfreeze:int=5,
    add_dropout: bool=False,
    dropout_prob: float=0.1,
    device:str|torch.device="cpu"
) -> tuple[nn.Module, AutoTokenizer]:
    """This function creates a Qwen 2.5 Classifier model and tokenizer.

    Args:
        model_id (str): Model ID for the Qwen 2.5 model.
        freeze_norm_layer (bool, optional): Whether to freeze the norm layer. Defaults to False.
        freeze_embedding (bool, optional): Whether to freeze the embedding layer. Defaults to True.
        num_layers_to_unfreeze (int, optional): Number of layers to unfreeze (counting from end). Defaults to 5.
        num_classes (int, optional): Number of classes for the classifier. Defaults to 8.
        device (str | torch.device, optional): Device, defaults to "cpu".

    Returns:
        tuple[nn.Module, AutoTokenizer]: A tuple containing the Qwen 2.5 Classifier model and tokenizer.
    """
    
    ### Create the Qwen 2.5 Classifier
    # Load the configuration
    config = AutoConfig.from_pretrained(model_id)
    
    # Load the base model (without head)
    base_model = AutoModel.from_pretrained(model_id, config=config)
    
    # Initialize the classifier
    model = Qwen3Classifier(config=config, base_model=base_model, num_classes=num_classes, add_dropout=add_dropout,
    dropout_prob=dropout_prob).to(device)
    
    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Embedding layer
    if not freeze_embedding:
        for param in model.qwen_base.embed_tokens.parameters():
            param.requires_grad = True
    
    # Normalization layer
    if not freeze_norm_layer:
        for param in model.qwen_base.norm.parameters():
            param.requires_grad = True
            
    # Unfreeze the last n decoder layers
    if num_decoder_layers_to_unfreeze > 0:
        total_layers = len(model.qwen_base.layers)
        for i in range(total_layers - num_decoder_layers_to_unfreeze, total_layers):
            for param in model.qwen_base.layers[i].parameters():
                param.requires_grad = True
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

def create_qwen3_classifier_with_lora(
    model_id:str,
    lora_config:LoraConfig,
    num_classes:int=8,
    add_dropout: bool=False,
    dropout_prob: float=0.1,
    device:str|torch.device="cpu"
) -> tuple[nn.Module, AutoTokenizer]:
    """This function creates a Qwen 2.5 Classifier model and tokenizer.

    Args:
        model_id (str): Model ID for the Qwen 2.5 model.
        lora_config (LoraConfig): LoRA configuration.
        num_classes (int, optional): Number of classes for the classifier. Defaults to 8.
        add_dropout (bool, optional): Whether to add dropout. Defaults to False.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.    
        device (str | torch.device, optional): Device, defaults to "cpu".

    Returns:
        tuple[nn.Module, AutoTokenizer]: A tuple containing the Qwen 2.5 Classifier model and tokenizer.
    """
    
    ### Create the Qwen 2.5 Classifier
    # Load the configuration
    config = AutoConfig.from_pretrained(model_id)
    
    # Load the base model (without head)
    base_model = AutoModel.from_pretrained(model_id, config=config)
    
    # Initialize the classifier
    ### Note: The LoRA automatically freezes the all base model parameters, besides the LoRA parameters, we need to unfreeze the classifier layer.
    model = Qwen3Classifier(config=config, base_model=base_model, lora_config=lora_config, num_classes=num_classes, add_dropout=add_dropout, dropout_prob=dropout_prob).to(device)
    
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer
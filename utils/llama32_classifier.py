import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel, PretrainedConfig
class Llama32Classifier(PreTrainedModel):
  def __init__(self, config: PretrainedConfig, base_model: nn.Module, num_classes: int, add_dropout: bool = False, dropout_prob: float = 0.1, lora_config: LoraConfig = None):
    super().__init__(config)
    self.num_labels = num_classes
    self.config = config

    if lora_config is not None:
      self.llama_base = get_peft_model(base_model, lora_config)
    else:
      self.llama_base = base_model

    if add_dropout:
      self.classifier = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(in_features=config.hidden_size, out_features=num_classes)
      )
    else:
      self.classifier = nn.Sequential(nn.Linear(in_features=config.hidden_size, out_features=num_classes))

  def forward(self, input_ids, attention_mask) -> torch.Tensor:
    base_outputs = self.llama_base(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    last_hidden_state = base_outputs.last_hidden_state

    sequence_lengths = torch.max(attention_mask.sum(dim=1) - 1, 
                                 torch.zeros(1, dtype=torch.long, device=input_ids.device))
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0],
                                      device=last_hidden_state.device), 
                                      sequence_lengths]

    return self.classifier(pooled_output)
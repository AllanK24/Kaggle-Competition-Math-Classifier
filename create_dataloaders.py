import torch
import transformers
import pandas as pd
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class MathDataset(Dataset):
    def __init__(self, csv_path:str|Path):
        try:
            self.dataset = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {csv_path}. Double check the dataset path. Error: {e}")
    
    def __getitem__(self, index):
        return {
            "prompt": self.dataset['prompt'][index], 
            "label": self.dataset['label'][index]
        } ### Return a dictionary or just item without keys?
    
    def __len__(self):
        return len(self.dataset)
    
def collate(batch, tokenizer: transformers.AutoTokenizer, max_length:int=None):
    prompts = [item['prompt'] for item in batch]
    labels = [item['label'] for item in batch]
    # If max_length is specified, use truncation and pad to that length
    # Otherwise, pad dynamically to the longest sequence in the batch
    if max_length is not None:
        encoding = tokenizer(
            prompts,
            padding='max_length', # fixed length padding
            max_length=max_length,
            return_tensors='pt'
        )
    else:
        encoding = tokenizer(
            prompts,
            padding=True,          # dynamic padding
            return_tensors='pt'
        )
        
    return {
        "input_ids": encoding['input_ids'],
        "attention_mask": encoding['attention_mask'],
        "label": torch.tensor(labels)
    }

def create_dataloaders(train_dataset_path, val_dataset_path, batch_size, num_workers, shuffle, tokenizer: transformers.AutoTokenizer, collate_fn=collate, max_length:int=None):
    train_dataset = MathDataset(train_dataset_path)
    val_dataset = MathDataset(val_dataset_path)
    
    collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=max_length) # partial function to pass tokenizer and max_length, because collate function expects only one argument
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader
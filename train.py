import torch
import accelerate
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          accelerator: accelerate.Accelerator,
          scheduler: torch.optim.lr_scheduler._LRScheduler=None,
          save_path: str|Path="/model/"
          ) -> dict:
    accelerator.print("[INFO] Starting training...")

    # Store history
    results = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": []
    }

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    for epoch in tqdm(range(epochs)):
        epoch_train_loss, epoch_train_f1 = train_step(model, train_dataloader, loss_fn, optimizer, accelerator, scheduler)
        epoch_val_loss, epoch_val_f1 = val_step(model, val_dataloader, loss_fn, accelerator)
        
        results["train_loss"].append(epoch_train_loss)
        results["train_f1"].append(epoch_train_f1)
        results["val_loss"].append(epoch_val_loss)
        results["val_f1"].append(epoch_val_f1)
        
        accelerator.print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train F1: {epoch_train_f1:.2f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val F1: {epoch_val_f1:.4f}")

    accelerator.print("[INFO] Training Completed")
    
    # Save the model
    accelerator.print("[INFO] Saving the model...")
    try:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        accelerator.print("[INFO] Model was unwrapped and saved successfully!")    
    except Exception as e:
        accelerator.print(f"[INFO] Error when saving model. Error: {e}")
    return results

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.CrossEntropyLoss,
               optimizer: torch.optim.Optimizer,
               accelerator: accelerate.Accelerator,
               scheduler: torch.optim.lr_scheduler._LRScheduler=None):
    model.train()
    
    epoch_loss = 0
    all_preds, all_labels = [], []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False, disable=not accelerator.is_main_process) # leave parameter, for disabling progress bar after it's completed
    
    # Enable mixed precision
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        with accelerator.autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
        epoch_loss+=loss.item()
        
        accelerator.backward(loss)
        optimizer.step()
        
        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # Visualizing dynamically loss per batch in progress bar
        progress_bar.set_postfix(loss=loss.item())

    if scheduler:
        scheduler.step()
    
    avg_epoch_loss = epoch_loss/len(dataloader)
    avg_epoch_f1 = f1_score(y_true=all_labels, y_pred=all_preds, average="micro")
    
    return avg_epoch_loss, avg_epoch_f1
    
def val_step(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.CrossEntropyLoss,
             accelerator: accelerate.Accelerator,
             ):
    epoch_loss = 0
    all_preds, all_labels = [], []
    
    model.eval()
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.inference_mode():
        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = loss_fn(logits, labels)
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Visualizing dynamically loss per batch in progress bar
            progress_bar.set_postfix(loss=loss.item())
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_epoch_f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='micro')
    
    return avg_epoch_loss, avg_epoch_f1
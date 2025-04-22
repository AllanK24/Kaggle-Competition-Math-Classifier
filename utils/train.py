import wandb
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
          f1_avg_mode:str='micro',
          checkpoint_every_n_epochs: int = 1,           # How often to save
          save_path: str|Path= "./finetuned_model/"
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
        epoch_train_loss, epoch_train_f1 = train_step(model, train_dataloader, loss_fn, optimizer, accelerator, scheduler, f1_avg_mode)
        epoch_val_loss, epoch_val_f1 = val_step(model, val_dataloader, loss_fn, accelerator, f1_avg_mode)
        
        # Epoch level scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateu):
                scheduler.step(epoch_val_loss) # Or another metric like val_f1
            else:
                scheduler.step()
        
        results["train_loss"].append(epoch_train_loss)
        results["train_f1"].append(epoch_train_f1)
        results["val_loss"].append(epoch_val_loss)
        results["val_f1"].append(epoch_val_f1)
        
        # Log via wandb
        wandb.log({
            "train_loss": results['train_loss'],
            "train_f1": results['train_f1'],
            "val_loss": results['val_loss'],
            "val_f1": results['val_f1']
        })
        
        accelerator.print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train F1: {epoch_train_f1:.2f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val F1: {epoch_val_f1:.4f}")
        
        if checkpoint_every_n_epochs > 0 and epoch+1 % checkpoint_every_n_epochs == 0:
            checkpoint_path = save_path / f"epoch_{epoch+1}"
            accelerator.print(f"[INFO] Saving checkpoint for epoch {epoch+1} to {checkpoint_path}...")
            
            if accelerator.is_main_process:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            accelerator.wait_for_everyone() # Wait until all processes finish
            try:
                accelerator.save_state(output_dir=checkpoint_path)
                accelerator.print(f"[INFO] Checkpoint saved successfully.")
            except Exception as e:
                accelerator.print(f"[INFO] Failed to save checkpoint. Error: {e}")

    accelerator.print("[INFO] Training Completed")
    
    # Save the model
    accelerator.print("[INFO] Saving the model...")
    
    try:
        if accelerator.is_main_process:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        accelerator.wait_for_everyone()
        
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
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               f1_avg_mode: str):
    model.train()
    
    epoch_loss = 0
    all_logits, all_labels_list = [], []
    
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
        
        # For more precise loss tracking across devices: avg_loss = accelerator.gather(loss).mean()
        epoch_loss += loss.item() * input_ids.size(0) # Accumulate weighted loss
              
        accelerator.backward(loss)
        optimizer.step()

        # Step-level scheduler if needed (uncomment if using step-based scheduler)
        # if scheduler and isinstance(scheduler, StepBasedScheduler):
        #     scheduler.step()

        all_logits.append(accelerator.gather_for_metrics(logits.detach()))
        all_labels_list.append(accelerator.gather_for_metrics(labels))
        
        # Visualizing dynamically loss per batch in progress bar
        if accelerator.is_main_process:
                  progress_bar.set_postfix(loss=loss.item())

    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels_list)

    all_preds_cat = torch.argmax(all_logits_cat, dim=-1)

    avg_epoch_f1 = f1_score(y_true=all_labels_cat.cpu().numpy(), y_pred=all_preds_cat.cpu().numpy(), average=f1_avg_mode)
    
    total_samples = len(dataloader.dataset) # Assumes dataloader has .dataset attribute
    avg_epoch_loss = epoch_loss / total_samples
    
    return avg_epoch_loss, avg_epoch_f1
    
def val_step(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.CrossEntropyLoss,
             accelerator: accelerate.Accelerator,
             f1_avg_mode:str,
             ):
    epoch_loss = 0
    all_logits, all_labels_list = [], []
    
    model.eval()
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.inference_mode():
        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            with accelerator.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                
            epoch_loss += loss.item() * input_ids.size(0)
            
            all_logits.append(accelerator.gather_for_metrics(logits.detach()))
            all_labels_list.append(accelerator.gather_for_metrics(labels))
            
            # Visualizing dynamically loss per batch in progress bar
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=loss.item())
            
    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels_list)
    
    all_preds_cat = torch.argmax(all_logits_cat, dim=-1)
    
    avg_epoch_f1 = f1_score(y_true=all_labels_cat.cpu().numpy(), y_pred=all_preds_cat.cpu().numpy(), average=f1_avg_mode)
    
     # Calculate average loss for the epoch
    total_samples = len(dataloader.dataset)
    avg_epoch_loss = epoch_loss / total_samples
    
    return avg_epoch_loss, avg_epoch_f1
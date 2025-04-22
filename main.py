import os
import wandb
from huggingface_hub import login

def main():
    # Logins
    wandb.login(key=os.environ.get("WANDB_TOKEN"))
    login(token=os.environ.get("HF_TOKEN"))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Initialize wandb before training
    wandb.init(
        project='', # Project name
        name='1st', # Run name
        config="" # Train config: lr, epochs and etc
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Finish wandb run
    wandb.finish()

if __name__=="__main__":
    pass
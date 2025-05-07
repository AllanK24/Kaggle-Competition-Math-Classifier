from pathlib import Path
from peft import PeftModel

def save_adapter_only(adapter: PeftModel, path: Path):
    """
    Save just the LoRA adapter weights (called on main process only).
    """
    path.mkdir(parents=True, exist_ok=True)
    adapter.save_pretrained(path, safe_serialization=True)            # adapter_model.bin + config
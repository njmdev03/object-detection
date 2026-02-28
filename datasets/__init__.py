from .pennfudan import PennFudanDataset
from .pet import PetDataset

def get_dataset(name):
    if name.lower() == "penn":
        return PennFudanDataset
    elif name.lower() in ["pet", "oxford_pet"]:
        return PetDataset
    else:
        raise ValueError(f"Unknown dataset: {name}")
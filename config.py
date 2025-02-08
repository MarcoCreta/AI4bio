import json

class Config():
    DATASET = "/homes/mcreta/AI4bio/static/data/UP000005640_9606.fasta"
    OUTPUT_PATH = "/homes/mcreta/AI4bio/static"

    DATA_NAME = "data/prot_gpt2_multi.pt"

    CLASSES = ["R-HSA-metabolism"]

    RANDOMNESS = {
        "PYTORCH_SEED" : 42,
        "NUMPY_SEED": 42,
        "PYTHON_SEED": 42,
    }

    FF_ARGS = {
        "INPUT_SIZE" : 1280,
        "OUTPUT_SIZE" : len(CLASSES)+1,
        "HIDDEN_SIZE" : 512,
        "DROPOUT" : 0.5,
    }

    TRAIN_ARGS = {
        "N_EPOCHS" : 3,
        "EMB_FUSION_FN" : None,
        "BATCH_SIZE" : 64,
        "LEARNING_RATE" : 5e-4,
        "WEIGHT_DECAY" : 1e-4,
        "EMB_CHUNKS" : 5,
    }

    @classmethod
    def json(cls):
        # Only include attributes that are not callable and not class-level special methods
        return {
            key: value
            for key, value in vars(cls).items()
            if not callable(value) and not isinstance(value, classmethod) and not key.startswith("__")
        }

    @classmethod
    def print(cls):
        """Print the configuration as a formatted JSON string."""
        try:
            print(json.dumps(cls.json(), indent=4) + "\n-------------------------------------------------------------------\n")
        except TypeError as e:
            print(f"Error serializing config: {e}")


Config.print()
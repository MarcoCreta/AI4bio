import json

class Config():
    DATASET = "/homes/mcreta/AI4bio/static/data/UP000005640_9606.fasta"
    OUTPUT_PATH = "/homes/mcreta/AI4bio/static"

    DATA_NAME = "data/prot_gpt2_multi.pt"

    CLASSES = ["R-HSA-metabolism", "R-HSA-109581"]

    RANDOMNESS = {
        "PYTORCH_SEED" : 42,
        "NUMPY_SEED": 42,
        "PYTHON_SEED": 42,
    }

    TRAIN_ARGS = {
        "N_EPOCHS" : 300,
        "EMB_FUSION_FN" : None,
        "BATCH_SIZE" : 64,
        "LEARNING_RATE" : 1e-5,
        "WEIGHT_DECAY" : 1e-3, #keep fixed, generate an upper bound of val below 0.1
        "EMB_CHUNKS" : 5,
        "ATTN_POOLING" : True,
        "EMB_SIZE" : 1280,
        "TOKEN_PE" : True,
        "CHUNK_RE" : True,
    }

    FF_ARGS = {
        "INPUT_SIZE" : 1280,
        "OUTPUT_SIZE" : 3,
        "HIDDEN_SIZE" : 512,
        "DROPOUT" : 0.5,
    }

    def __init__(self):
        self.FF_ARGS['INPUT_SIZE'] = self.TRAIN_ARGS['EMB_SIZE']
        self.FF_ARGS['OUTPUT_SIZE'] = len(self.CLASSES)+1


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
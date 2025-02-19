import json

class Config():
    DATASET = "/homes/mcreta/AI4bio/static/data/UP000005640_9606.fasta"
    OUTPUT_PATH = "/homes/mcreta/AI4bio/static"

    DATA_NAME = "data/prot_t5_multi.pt"

    CLASSES = ["R-HSA-metabolism", "R-HSA-109581", "R-HSA-signal_trasduction", "R-HSA-73894.5"]

    USE_DEFAULT = True

    RANDOMNESS = {
        "PYTORCH_SEED" : 66,
        "NUMPY_SEED": 99,
        "PYTHON_SEED": 33,
    }

    TRAIN_ARGS = {
        "N_EPOCHS" : 100,
        "BATCH_SIZE" : 64,
        "LEARNING_RATE" : 1e-4,
        "WEIGHT_DECAY" : 3e-3,
        "EMB_CHUNKS" : 3,
        "EMB_SIZE" : 1280,

        "ATTN_POOLING": False,
        "LEARN_PE": False,
        "TOKEN_PE" : False,
        "CHUNK_RE" : False,

        "WEIGHTS" : False,

        "LOSS" : "focal",
        "ALPHA" : 1,
        "GAMMA": 1,
    }


    FF_ARGS = {
        "INPUT_SIZE" : 1280,
        "OUTPUT_SIZE" : 2,
        "HIDDEN_SIZE" : 128,
        "DROPOUT" : 0.5,
        "THRESHOLD": 0.5,
    }

    @classmethod
    def init(cls):
        if cls.DATA_NAME == "data/prot_bert_multi.pt" : cls.TRAIN_ARGS['EMB_SIZE'] = 1024
        elif cls.DATA_NAME == "data/prot_gpt2_multi.pt" : cls.TRAIN_ARGS['EMB_SIZE'] = 1280
        elif cls.DATA_NAME == "data/prot_t5_multi.pt": cls.TRAIN_ARGS['EMB_SIZE'] = 1024

        if cls.TRAIN_ARGS['ATTN_POOLING']:
            cls.FF_ARGS['INPUT_SIZE'] = cls.TRAIN_ARGS['EMB_SIZE']
        else :
            cls.FF_ARGS['INPUT_SIZE'] = cls.TRAIN_ARGS['EMB_CHUNKS'] * cls.TRAIN_ARGS['EMB_SIZE']
            cls.TRAIN_ARGS['TOKEN_PE'] = None
            cls.TRAIN_ARGS['CHUNK_RE'] = None
            cls.TRAIN_ARGS['LEARN_PE'] = None

        if cls.TRAIN_ARGS['LOSS'] == "bce":
            cls.TRAIN_ARGS['GAMMA'] = None
            cls.TRAIN_ARGS['ALPHA'] = None
        if cls.TRAIN_ARGS['LOSS'] == "focal":
            cls.TRAIN_ARGS['weights'] = False

        #cls.FF_ARGS['HIDDEN_SIZE'] = cls.FF_ARGS['INPUT_SIZE']//2


        if cls.USE_DEFAULT:
            classes = ["default"]
            classes.extend(cls.CLASSES)
            cls.CLASSES = classes

        cls.FF_ARGS['OUTPUT_SIZE'] = len(cls.CLASSES)

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


Config.init()
Config.print()
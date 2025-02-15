from common.utils import count_tokens, print_counts
from data.dataloader import FastaDataset
from config import Config


if __name__ == "__main__":

    models = ["Rostlab/prot_bert", "Rostlab/prot_t5_xl_bfd", "nferruz/ProtGPT2"]

    for model in models:
        dataset = FastaDataset(file_path=Config.DATASET)

        counts = count_tokens(dataset, model)
        print_counts(counts, model, 3)

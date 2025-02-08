from common.utils import count_tokens, print_counts
from transformers import pipeline, AutoTokenizer, AutoModel
from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from config import Config


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    dataset = FastaDataset(file_path=Config.DATASET)

    counts = count_tokens(dataset, tokenizer)
    print_counts(counts)

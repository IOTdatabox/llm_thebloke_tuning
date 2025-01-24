import os
import logging
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
import pandas as pd
from itertools import islice

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShuffledJSONDatasetIterator:
    def __init__(self):
        dataframes = []

        for filename in ["java", "javascript", "simple", "multiple", "sql", "live_simple", "live_multiple"]:
            bfcl_path = "bitagent.data/bfcl/BFCL_v3_{filename}.json"
            bfcl_answer_path = "bitagent.data/bfcl/possible_answer/BFCL_v3_{filename}.json"
            file_path = bfcl_path.format(filename=filename)
            answer_path = bfcl_answer_path.format(filename=filename)
            df_data = pd.read_json(file_path, lines=True)
            df_answer = pd.read_json(answer_path, lines=True)
            df_data['ground_truth'] = df_answer['ground_truth']
            dataframes.append(df_data[['id','question','function','ground_truth']])
        self.all_data = pd.concat(dataframes)
        self._shuffle_data()

    def _shuffle_data(self):
        self.shuffled_data = self.all_data.sample(frac=1).reset_index(drop=True)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.shuffled_data):
            row = self.shuffled_data.iloc[self.index]
            self.index += 1
            return row
        else:
            self._shuffle_data()  # Shuffle and reset index if end is reached
            return self.__next__()

def huggingface_loader(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    logger.debug(f"Loading {dataset_name}")
    dataset_dir = f"{root_data_dir}/{dataset_name.replace('/','_')}"
    if os.path.exists(f"{dataset_dir}/state.json"):
        logger.debug(f"Loading from disk ({dataset_dir}) ...")
        ds = load_from_disk(dataset_dir)
    else:
        logger.debug("Loading from web ...") 
        ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
        ds.save_to_disk(dataset_dir)
    logger.debug("Loaded.")
    return ds

def load_bfcl_dataset(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    snapshot_download(repo_id=dataset_name, allow_patterns="*.json", repo_type="dataset", local_dir="bitagent.data/bfcl/")
    return ShuffledJSONDatasetIterator()

def sample_and_save_datasets(output_dir="bitagent.data/samples", sample_size=1000):
    """Sample datasets and save to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and sample Glaive dataset
    try:
        glaive_ds = huggingface_loader("glaiveai/glaive-function-calling-v2")
        glaive_df = pd.DataFrame(glaive_ds)
        glaive_sample = glaive_df.sample(n=min(sample_size, len(glaive_df)))
        glaive_sample.to_csv(f"{output_dir}/glaive_sample.csv", index=False)
        logger.info(f"Saved Glaive sample to {output_dir}/glaive_sample.csv")
    except Exception as e:
        logger.error(f"Error processing Glaive dataset: {str(e)}")

    # Load and sample BitAgent dataset
    try:
        bitagent_ds = huggingface_loader("BitAgent/tool_calling")
        bitagent_df = pd.DataFrame(bitagent_ds)
        bitagent_sample = bitagent_df.sample(n=min(sample_size, len(bitagent_df)))
        bitagent_sample.to_csv(f"{output_dir}/bitagent_sample.csv", index=False)
        logger.info(f"Saved BitAgent sample to {output_dir}/bitagent_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")

    # Load and sample Berkeley Function Calling dataset
    try:
        bfcl_ds = load_bfcl_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")
        bfcl_sample = pd.DataFrame(list(islice(bfcl_ds, sample_size)))
        bfcl_sample.to_csv(f"{output_dir}/bfcl_sample.csv", index=False)
        logger.info(f"Saved BFCL sample to {output_dir}/bfcl_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BFCL dataset: {str(e)}")

if __name__ == "__main__":
    sample_and_save_datasets()
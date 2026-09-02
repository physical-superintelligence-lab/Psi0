# adapted from gr00t/data/dataset.py::calculate_dataset_statistics
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        # parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns: # type:ignore
        print(f"Computing statistics for {le_modality}...")
        # check if the data is the modality is actually a list of numbers
        # skip if it is a string
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            print(f"Skipping {le_modality} because it is a string")
            continue

        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
            
    return dataset_statistics

if __name__ == "__main__":
    import argparse
    import json


    LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
    LE_ROBOT_STATS_FILENAME = "meta/stats.json"
    
    parser = argparse.ArgumentParser(description="Calculate dataset statistics for LeRobot parquet files.")
    # parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument("--work-dir", type=str, default="_lerobot_build")
    parser.add_argument("--task", type=str, default=None, help="Process only this specific task (category/task_name)")
    parser.add_argument("--task-dir", type=str, default=None, help="Path to the lerobot data dir.")
    args = parser.parse_args()

    # dataset_path = Path(f"{args.work_dir}/{args.task}")
    dataset_path = Path(args.task_dir) if args.task_dir else Path(f"{args.work_dir}/{args.task}")
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    parquet_files = list((dataset_path).glob(LE_ROBOT_DATA_FILENAME))

    le_statistics = calculate_dataset_statistics(parquet_files)
    with open(stats_path, "w") as f:
        json.dump(le_statistics, f, indent=4)
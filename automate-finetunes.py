#!/usr/bin/env python
import subprocess
import itertools

def main():
    # Define the parameter lists
    batch_sizes = [4, 8, 16]
    epoch_lens = [20]
    ks = [4]
    data_files = [
        "test_data_33_unmod.csv",
        "test_data_67_unmod.csv",
        "test_data_100_unmod.csv",
        "test_data_33_mod_single.csv",
        "test_data_67_mod_single.csv",
        "test_data_100_mod_single.csv",
        "test_data_33_mod_mix.csv",
        "test_data_67_mod_mix.csv",
        "test_data_100_mod_mix.csv"
    ]
    model_names = [
        "google/vivit-b-16x2-kinetics400",
        "facebook/timesformer-base-finetuned-k400"
    ]
    
    # Path to your fine-tune script
    finetune_script = "finetune.py"
    
    for model_name in model_names:
        # For the two models, iterate over all data files
        for data_file in data_files:
            for bs, epochs, k in itertools.product(batch_sizes, epoch_lens, ks):
                print(f"\nRunning fine-tune with batch_size={bs}, data_file={data_file}, model_name={model_name}")
                cmd = [
                    "python", finetune_script,
                    "--batch_size", str(bs),
                    "--epochs", str(epochs),
                    "--k", str(k),
                    "--data_file", data_file,
                    "--model_name", model_name
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("Error:", result.stderr)

if __name__ == "__main__":
    main()
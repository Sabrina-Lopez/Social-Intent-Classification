import subprocess
import itertools
from collections import defaultdict
import wandb
from dotenv import load_dotenv
import os
import random

def main():
    api = wandb.Api()
    project_path = "sabrinameganlopez015-live-robotics/Fine-tuned Intent Classification 6-Model Testing"
    saved_model_dir = "./saved_model_dir"

    project_exists = False
    # Figure out once whether the project actually exists
    try:
        _ = api.project(project_path)
        project_exists = True
    except ValueError:
        project_exists = False

    # Define the parameter lists
    batch_sizes = [
        4 #, 
        # 8,
        # 16
        ]
    data_files = [
        "test_data_100_unmod_latest.csv"
    ]
    # Read directory of saved models
    """
    model_names = [
        name for name in os.listdir(saved_model_dir)    
        if os.path.isdir(os.path.join(saved_model_dir, name))
    ]
    """
    # Recursively gather all model subfolders inside each folder in saved_model_dir
    model_names = []
    for root, dirs, _ in os.walk(saved_model_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                model_names.append(os.path.relpath(full_path, saved_model_dir))
    
    keywords = ["vjepa2", "timesformer", "vivit"]
    filtered_list = [item for item in model_names if any(keyword in item for keyword in keywords)]
    model_names = filtered_list

    # model_names.reverse()
    random.Random(128).shuffle(model_names)
    # model_names.append("google/vivit-b-16x2-kinetics400")
    # model_names.append("[INSERT VIVIT FACTORIZED ENCODER MODEL HERE]")
    # model_names.append("facebook/timesformer-base-finetuned-k400")
    # model_names.append("facebook/timesformer-base-finetuned-k600")
    # model_names.append("facebook/timesformer-base-finetuned-ssv2")
    # model_names.append("facebook/vjepa2-vitl-fpc16-256-ssv2")
    # model_names.append("facebook/vjepa2-vitl-fpc16-256-diving48")
    test_scripts = [
        # "test.py",
        "seedless-test.py",
        "seedless-test-no-vjepa2.py"
    ]

    for script, model_name, dataset, bs in itertools.product(test_scripts, model_names, data_files, batch_sizes):  

        # No VJEPA2 packages when in live-robotics-pip2 environment, no TimeSformer or ViVit packages when in live-robotics-pip environment
        # if "timesformer" in model_name or "vivit" in model_name: continue
        if "vjepa2" in model_name: continue

        if "timesformer" in model_name and "no-vjepa2" not in script: continue
        if "vjepa2" in model_name and "no-vjepa2" in script: continue
        if "vivit" in model_name and "no-vjepa2" not in script: continue

        data_file = {"test": dataset}
        dataset_type = os.path.splitext(data_file["test"])[0]

        # """
        if project_exists:
            try:
                filters = {
                    # "config.batch_size": bs,
                    "config.model_name": str(os.path.join(saved_model_dir, model_name))
                }
                runs = api.runs(project_path, filters=filters)
                
                # for run in runs:
                #     config = run.config
                #     print(config, len(runs))
                
                
                if (len(runs) != 0): 
                    # print('continue')
                    continue 
            except ValueError as e:
                runs = []
        else:
            # No project means no existing runs to collide with
            runs = []
        # """

        print(f"\nRunning test with batch_size={bs}, data_file={dataset}, model_name={model_name}")
        cmd = [
            "python", script,
            "--batch_size", str(bs),
            "--data_file", dataset,
            "--model_name", str(os.path.join(saved_model_dir, model_name)),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("Error:", result.stderr)


if __name__ == "__main__":
    main()
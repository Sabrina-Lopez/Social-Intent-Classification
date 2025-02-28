import subprocess

def main():
    # Define the parameter lists
    batch_sizes = [4]
    data_files = [
        "test_data_33_unmod.csv",
        "test_data_67_unmod.csv",
        "test_data_100_unmod.csv",
        # "test_data_33_mod_single.csv",
        # "test_data_67_mod_single.csv",
        # "test_data_100_mod_single.csv",
        # "test_data_33_mod_mix.csv",
        # "test_data_67_mod_mix.csv",
        # "test_data_100_mod_mix.csv"
    ]
    model_names = [
        "google/vivit-b-16x2-kinetics400",
        "vivit-finetuned-train_data_33_unmod",
        "vivit-finetuned-train_data_67_unmod",
        "vivit-finetuned-train_data_100_unmod",
        # "vivit-finetuned-train_data_33_mod_single",
        # "vivit-finetuned-train_data_67_mod_single",
        # "vivit-finetuned-train_data_100_mod_single",
        # "vivit-finetuned-train_data_33_mod_mix",
        # "vivit-finetuned-train_data_67_mod_mix",
        # "vivit-finetuned-train_data_100_mod_mix",
        "facebook/timesformer-base-finetuned-k400",
        "timesformer-finetuned-train_data_33_unmod",
        "timesformer-finetuned-train_data_67_unmod",
        "timesformer-finetuned-train_data_100_unmod",
        # "timesformer-finetuned-train_data_33_mod_single",
        # "timesformer-finetuned-train_data_67_mod_single",
        # "timesformer-finetuned-train_data_100_mod_single",
        # "timesformer-finetuned-train_data_33_mod_mix",
        # "timesformer-finetuned-train_data_67_mod_mix",
        # "timesformer-finetuned-train_data_100_mod_mix"
    ]
    finetunes = [
        "default",
        "lora"
    ]
    

    # Path to your test script
    test_script = "test.py"
    
    for model_name in model_names:
        # For these two models, iterate over all data files
        if model_name in ["google/vivit-b-16x2-kinetics400", "facebook/timesformer-base-finetuned-k400"]:
            for data_file in data_files:
                for bs in batch_sizes:
                    for finetune in finetunes:
                        print(f"\nRunning test with finetune={finetune}, batch_size={bs}, data_file={data_file}, model_name={model_name}")
                        model_name = model_name + "-method-" + finetune
                        cmd = [
                            "python", test_script,
                            "--batch_size", str(bs),
                            "--data_file", data_file,
                            "--model_name", model_name
                        ]

                        result = subprocess.run(cmd, capture_output=True, text=True)
                        print(result.stdout)
                        
                        if result.stderr:
                            print("Error:", result.stderr)
        else:
            # For all other models, choose the data file based on a substring match
            if "33_unmod" in model_name:
                data_file = data_files[0]
            elif "67_unmod" in model_name:
                data_file = data_files[1]
            elif "100_unmod" in model_name:
                data_file = data_files[2]
            elif "33_mod_single" in model_name:
                data_file = data_files[3]
            elif "67_mod_single" in model_name:
                data_file = data_files[4]
            elif "100_mod_single" in model_name:
                data_file = data_files[5]
            elif "33_mod_mix" in model_name:
                data_file = data_files[6]
            elif "67_mod_mix" in model_name:
                data_file = data_files[7]
            elif "100_mod_mix" in model_name:
                data_file = data_files[8]
            else:
                data_file = data_files[0]


            for bs in batch_sizes:
                for finetune in finetunes:
                    print(f"\nRunning test with finetune={finetune}, batch_size={bs}, data_file={data_file}, model_name={model_name}")
                    model_name = model_name + "-method-" + finetune
                    cmd = [
                        "python", test_script,
                        "--batch_size", str(bs),
                        "--data_file", data_file,
                        "--model_name", model_name
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    print(result.stdout)
                
                    if result.stderr:
                        print("Error:", result.stderr)


if __name__ == "__main__":
    main()

import subprocess
import itertools

def main():
    # Define the parameter lists
    # batch_sizes = [16]
    batch_sizes = [
        # 4, 
        # 8, 
        16]
    epoch_lens = [
        # 40, 
        # 50, 
        60]
    # ks = [4]
    ks = [1]
    data_files = [
        # "train_data_33_unmod.csv",
        # "train_data_67_unmod.csv",
        "train_data_100_unmod.csv",
        # "train_data_33_mod_single.csv",
        # "train_data_67_mod_single.csv",
        # "train_data_100_mod_single.csv",
        # "train_data_33_mod_mix.csv",
        # "train_data_67_mod_mix.csv",
        # "train_data_100_mod_mix.csv"
        
    ]
    model_names = [
        "google/vivit-b-16x2-kinetics400",
        "facebook/timesformer-base-finetuned-k400"
    ]
    finetunes = [
        "default", # Fine-tuning that involves manually freezing model layers 
        "lora"
    ]


    # Path to your fine-tune script
    finetune_scripts = [
        # "finetune.py" #,
        "balanced-finetune.py"
    ]


    lora_alpha = [8, 16, 32, 64]
    lora_dropout = [0.0, 
                    # 0.1, 
                    0.2, 
                    # 0.3
                    ]

    hidden_dropout = [0.0, 
                    0.1, 
                    0.2, 
                    0.3]
    attn_dropout = [0.0, 
                    0.1, 
                    0.2, 
                    0.3]


    for model_name in model_names:
        # For the two models, iterate over all data files
        for data_file in data_files:
            for bs, epochs, k, l_alpha, l_dropout, h_dropout, a_dropout in itertools.product(batch_sizes, epoch_lens, ks, lora_alpha, lora_dropout, hidden_dropout, attn_dropout):
                for finetune in finetunes:
                    for script in finetune_scripts:
                        
                        if (script == "finetune.py"):
                            print(f"\nRunning fine-tune with finetune={finetune}, batch_size={bs}, data_file={data_file}, model_name={model_name}")
                            cmd = [
                                "python", script,
                                "--batch_size", str(bs),
                                "--epochs", str(epochs),
                                "--k", str(k),
                                "--data_file", data_file,
                                "--model_name", model_name,
                                "--finetune", finetune,
                                "--lora_alpha", l_alpha,
                                "--lora_dropout", l_dropout,
                                "--hidden_dropout", h_dropout,
                                "--attn_dropout", a_dropout
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            print(result.stdout)
                            
                            if result.stderr:
                                print("Error:", result.stderr)

                        elif (script == "balanced-finetune.py"):
                            balanced_data_files = "balanced_train_f1_data_unmod.csv,balanced_train_f2_data_unmod.csv,balanced_train_f3_data_unmod.csv,balanced_train_f4_data_unmod.csv"
                            print(f"\nRunning fine-tune with finetune={finetune}, batch_size={bs}, data_files={balanced_data_files}, model_name={model_name}")

                            cmd = [
                                "python", script,
                                "--batch_size", str(bs),
                                "--epochs", str(epochs),
                                "--k", str(k),
                                "--model_name", model_name,
                                "--finetune", finetune
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            print(result.stdout)
                            
                            if result.stderr:
                                print("Error:", result.stderr)


if __name__ == "__main__":
    main()
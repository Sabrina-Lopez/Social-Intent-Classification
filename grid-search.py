import argparse
import itertools
from collections import defaultdict
from finetune_grid_search import main as main
from balanced_finetune_grid_search import main as balanced_main

def grid_search():

    model_names = [
        "google/vivit-b-16x2-kinetics400",              
        "facebook/timesformer-base-finetuned-k400"
    ]
    finetune_methods = ["default", "lora"]
    batch_sizes = [4, 8, 16]
    epochs_list = [40, 50, 60]
    k_folds = [1]
    # k_folds = [1, 4]

    lora_alpha = [8, 16, 32, 64]
    lora_dropout = [0.0, 0.1, 0.2, 0.4]
    # lora_alpha = [16]
    # lora_dropout = [0.0, 0.1]

    hidden_dropout = [0.0, 0.05, 0.1, 0.2, 0.4]
    attn_dropout = [0.0, 0.05, 0.1, 0.2, 0.4]
    # hidden_dropout = [0.0, 0.1, 0.2]
    # attn_dropout = [0.0, 0.1, 0.2]


    data_file = "train_data_100_unmod.csv"  # Adjust dataset path as needed
    # data_file = "test_data_67_unmod.csv"
    # data_file = "test_data_33_unmod.csv"


    balanceBool = True


    # Dictionary to store results:
    # Key: (model_name, finetune_method)
    # Value: tuple(best_metric, best_params_dict)
    # NEED TO FIX
    best_results = defaultdict(lambda: (0, {}))


    data_files = "balanced_train_f1_data_unmod.csv,balanced_train_f2_data_unmod.csv,balanced_train_f3_data_unmod.csv,balanced_train_f4_data_unmod.csv"
    

    # Loop over all combinations
    for model_name, finetune, batch_size, epochs, k_fold in itertools.product(
        model_names, finetune_methods, batch_sizes, epochs_list, k_folds):


        # For "lora" fine-tuning, iterate over lora_alpha and lora_dropout as well.
        if finetune == "lora":
            for l_alpha, l_dropout in itertools.product(lora_alpha, lora_dropout):
                for h_dropout, a_dropout in itertools.product(hidden_dropout, attn_dropout):
                    print("=" * 50)
                    print(f"Running experiment with:")
                    print(f"  Model: {model_name}")
                    print(f"  Finetuning: {finetune}")
                    print(f"  Batch size: {batch_size}")
                    print(f"  Epochs: {epochs}")
                    print(f"  K-fold: {k_fold}")
                    print(f"  LoRA alpha: {l_alpha}")
                    print(f"  LoRA dropout: {l_dropout}")
                    print(f"  Hidden dropout: {h_dropout}")
                    print(f"  Attention dropout: {a_dropout}")
                    print("=" * 50)

                    
                    l_alpha_name = str(l_alpha).replace(".", "")
                    l_dropout_name = str(l_dropout).replace(".", "")
                    h_dropout_name = str(h_dropout).replace(".", "")
                    a_dropout_name = str(a_dropout).replace(".", "")


                    # Create a unique project title for wandb using key hyperparameters
                    project_title = (
                        "Social Intent Classification Grid Search"
                        # f"{model_name.replace('/', '_')}_{finetune}_"
                        # f"bs_{batch_size}_ep_{epochs}_k_{k_fold}_"
                        # f"lalpha_{l_alpha_name}_ldropout_{l_dropout_name}_"
                        # f"hdropout_{h_dropout_name}_attndropout_{a_dropout_name}"
                    )


                    # Create args namespace with all parameters
                    if (balanceBool):
                        args = argparse.Namespace(
                            batch_size=batch_size,
                            epochs=epochs,
                            k=k_fold,
                            model_name=model_name,
                            finetune=finetune,
                            lora_alpha=l_alpha,
                            lora_dropout=l_dropout,
                            hidden_dropout=h_dropout,
                            attn_dropout=a_dropout,
                            project_title=project_title,
                            data_files=data_files,
                            balanceBool=balanceBool
                        )
                    else:
                        args = argparse.Namespace(
                            batch_size=batch_size,
                            epochs=epochs,
                            k=k_fold,
                            model_name=model_name,
                            finetune=finetune,
                            lora_alpha=l_alpha,
                            lora_dropout=l_dropout,
                            hidden_dropout=h_dropout,
                            attn_dropout=a_dropout,
                            project_title=project_title,
                            data_file=data_file,
                            balanceBool=balanceBool
                        )

                    
                    # Run training and get the average best validation accuracy across folds
                    if balanceBool:
                        score, loss = balanced_main(args)
                    else:
                        score, loss = main(args)
                    print(f"Average best validation accuracy: {score:.3f}")
                    print(f"Average loss for best validation accuracy: {loss:.3f}")
                    
                    key = (model_name, finetune)

                    if (not best_results):
                        best_results[key] = ((score, loss), {
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "k_fold": k_fold,
                            "lora_alpha": l_alpha,
                            "lora_dropout": l_dropout,
                            "hidden_dropout": h_dropout,
                            "attn_dropout": a_dropout,
                            "project_title": project_title
                        })
                    else:
                        (best_score, best_loss), best_params = best_results[key]
                    
                        if (score > best_score):
                            best_results[key] = ((score, loss), {
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "k_fold": k_fold,
                                "lora_alpha": l_alpha,
                                "lora_dropout": l_dropout,
                                "hidden_dropout": h_dropout,
                                "attn_dropout": a_dropout,
                                "project_title": project_title
                            })
        else:  # for "default" finetuning (which doesn't use LoRA-specific parameters)
            for l_alpha, l_dropout in itertools.product(lora_alpha, lora_dropout):
                for h_dropout, a_dropout in itertools.product(hidden_dropout, attn_dropout):
                    print("=" * 50)
                    print(f"Running experiment with:")
                    print(f"  Model: {model_name}")
                    print(f"  Finetuning: {finetune}")
                    print(f"  Batch size: {batch_size}")
                    print(f"  Epochs: {epochs}")
                    print(f"  K-fold: {k_fold}")
                    print(f"  Hidden dropout: {h_dropout}")
                    print(f"  Attention dropout: {a_dropout}")
                    print("=" * 50)


                    h_dropout_name = str(h_dropout).replace(".", "")
                    a_dropout_name = str(a_dropout).replace(".", "")


                    # Create a unique project title for wandb using key hyperparameters
                    project_title = (
                        "Social Intent Classification Grid Search"
                        # f"{model_name.replace('/', '_')}_{finetune}_"
                        # f"bs_{batch_size}_ep_{epochs}_k_{k_fold}_"
                        # f"hdropout_{h_dropout_name}_attndropout_{a_dropout_name}"
                    )

                    
                    if (balanceBool):
                        args = argparse.Namespace(
                            batch_size=batch_size,
                            epochs=epochs,
                            k=k_fold,
                            model_name=model_name,
                            finetune=finetune,
                            lora_alpha=l_alpha,
                            lora_dropout=l_dropout,
                            hidden_dropout=h_dropout,
                            attn_dropout=a_dropout,
                            project_title=project_title,
                            data_files=data_files,
                            balanceBool=balanceBool
                        )
                    else:
                        args = argparse.Namespace(
                            batch_size=batch_size,
                            epochs=epochs,
                            k=k_fold,
                            model_name=model_name,
                            finetune=finetune,
                            lora_alpha=l_alpha,
                            lora_dropout=l_dropout,
                            hidden_dropout=h_dropout,
                            attn_dropout=a_dropout,
                            project_title=project_title,
                            data_file=data_file,
                            balanceBool=balanceBool
                        )
                    
                    if balanceBool:
                        score, loss = balanced_main(args)
                    else:
                        score, loss = main(args)
                    print(f"Average best validation accuracy: {score:.3f}")
                    print(f"Average loss for best validation accuracy: {loss:.3f}")
                        
                    key = (model_name, finetune)
                    (best_score, best_loss), best_params = best_results[key]
                    if score > best_score:
                        best_results[key] = ((score, loss), {
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "k_fold": k_fold,
                            "hidden_dropout": h_dropout,
                            "attn_dropout": a_dropout,
                            "project_title": project_title
                        })
    

    # Print out the best parameters for each (model, finetune) combination
    print("\n=== Best Hyperparameters per Model & Finetuning Method ===")
    for (model, finetune), (score, params) in best_results.items():
        print(f"Model: {model}, Finetuning: {finetune}")
        print(f"  Best Score: {score:.3f}")
        print(f"  Best Params: {params}")
        print("-"*50)


if __name__ == "__main__":
    grid_search()
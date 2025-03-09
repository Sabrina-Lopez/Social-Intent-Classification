import numpy as np
import random
import os
import torch
import wandb
import argparse
from dotenv import load_dotenv
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from utils import preprocess_function, sample_frame_indices
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split
from dataloader_class import VideoDataset
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model


load_dotenv()
wandb.login(key = os.environ.get("WANDB_API_KEY"))


def main(args):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Define hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    n_splits = args.k


    csv_files = args.data_files.split(',')


    # Define custom labels
    custom_labels = ["help", "hinder", "physical"]
    num_custom_labels = len(custom_labels)

    # Create label2id and id2label dicts
    label2id = {"help": 0, "hinder": 1, "physical": 2}
    id2label = {0: "help", 1: "hinder", 2: "physical"}

    # Load the base config from the pretrained checkpoint
    base_model_name = args.model_name


    config = None
    graph_name = None
    model = None
    image_processor = None


    # Load the base config from the pretrained checkpoint
    if ("vivit" in args.model_name):
        config = VivitConfig.from_pretrained(base_model_name)
    elif ("timesformer" in args.model_name):
        config = TimesformerConfig.from_pretrained(base_model_name)

    # Update number of labels and label mappings
    config.num_labels = num_custom_labels
    config.id2label = id2label
    config.label2id = label2id
    config.num_frames = 16
    config.hidden_dropout_prob = args.hidden_dropout
    config.attention_probs_dropout_prob = args.attn_dropout


    # Based on the fine-tuning method, setup the models accordingly
    finetune_method = args.finetune

    
    # Setup LoRA configuration
    lora_config = None
    r = 16


    # Get the pretrained image processor and model
    if ("vivit" in args.model_name):
        if (finetune_method == "lora"):
            lora_config = LoraConfig(
                # target_modules=["encoder.layer.10", "encoder.layer.11"], # Does not work
                # target_modules=["query", "value"],
                target_modules=["query", "key", "value"],
                bias="none",
                r=r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )

            image_processor = VivitImageProcessor.from_pretrained(base_model_name)
            model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
            model = get_peft_model(model, lora_config)
        elif (finetune_method == "default"):
            image_processor = VivitImageProcessor.from_pretrained(base_model_name)
            model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
            
            # Freeze last hidden layers of model
            for name, param in model.named_parameters():
                # If it's in the classification head, keep it trainable
                if ("classifier" in name):
                    param.requires_grad = True
                # If it's in the last 2 layers of the encoder, keep them trainable
                elif ("encoder.layer.10" in name or "encoder.layer.11" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    elif ("timesformer" in args.model_name):
        if (finetune_method == "lora"):
            lora_config = LoraConfig(
                target_modules=["qkv"],
                # target_modules=["encoder.layer.10", "encoder.layer.11"], # Does not work
                bias="none",
                r=r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )

            image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
            model = get_peft_model(model, lora_config)
        elif (finetune_method == "default"):
            image_processor = VivitImageProcessor.from_pretrained(base_model_name)
            model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)

            # Freeze last hidden layers of model
            for name, param in model.named_parameters():
                # If it's in the classification head, keep it trainable
                if ("classifier" in name):
                    param.requires_grad = True
                # If it's in the last 2 layers of the encoder, keep them trainable
                elif ("encoder.layer.10" in name or "encoder.layer.11" in name):
                    param.requires_grad = True
            else:
                param.requires_grad = False
    

    dataset_type = "fold_dataset"
    name = None
    if ("vivit" in args.model_name):
        name = f"vivit_fold_batch_{args.batch_size}"
    elif ("timesformer" in args.model_name):
        name = f"times_fold_batch_{args.batch_size}"


    """
    for name, param in model.named_parameters():
        print(name)

    return 
    """


    # Initialize wandb with project and configuration details
    wandb.init(
        project=args.project_title,
        config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "k": n_splits,
            "data_type": dataset_type,
            "model_name": base_model_name,
            "finetune": finetune_method,
            "lora_r": r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "balanced": args.balanceBool
        },
        name=name
    )

    
    fold_best_scores = []
    fold_best_losses = []


    # Train the model using k-fold cross-validation and custom dataset
    for fold in range(n_splits):
        print(f"=== Fold {fold+1} / {n_splits} ===")


        # Load the validation dataset for the current fold
        val_data = load_dataset('csv', data_files={"validation": csv_files[fold]}, sep=',')
        val_data["validation"] = val_data["validation"].map(preprocess_function)
        val_dataset = val_data["validation"].with_format("torch", columns=["video_path", "start", "end", "label"])

        # Load the training dataset from the remaining CSV files
        train_files = [csv_files[i] for i in range(len(csv_files)) if i != fold]
        train_data = load_dataset('csv', data_files={"train": train_files}, sep=',')
        train_data["train"] = train_data["train"].map(preprocess_function)
        train_dataset = train_data["train"].with_format("torch", columns=["video_path", "start", "end", "label"])

        # Create our custom VideoDataset for training and validation
        train_video_dataset = VideoDataset(train_dataset, image_processor, sample_frame_indices)
        val_video_dataset = VideoDataset(val_dataset, image_processor, sample_frame_indices)


        accumulation_steps = None
        # If fine-tuning with LoRA and a batch size of 16 is provided,
        # use an effective batch size of 8 and accumulate gradients over 2 mini-batches
        if finetune_method == "lora" and args.batch_size == 16:
            effective_batch_size = 8
            accumulation_steps = 2
        else:
            effective_batch_size = args.batch_size
            accumulation_steps = 1


        train_loader = DataLoader(train_video_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False)

        # Re-initialize or re-load the model each fold to keep re-frozen structure
        if ("vivit" in args.model_name):
            model = VivitForVideoClassification.from_pretrained(
                base_model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
        elif ("timesformer" in args.model_name):
            model = TimesformerForVideoClassification.from_pretrained(
                base_model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
       

        if (finetune_method == "lora"):
            model = get_peft_model(model, lora_config)
        elif (finetune_method == "default"):   
            # Freeze all but last layers:
            for name, param in model.named_parameters():
                if ("classifier" in name):
                    param.requires_grad = True
                elif ("encoder.layer.10" in name or "encoder.layer.11" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        


        # Setup optimizer for only unfrozen params
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        # Store best accuracy and loss for this fold
        fold_best = 0.0
        best_train_loss_for_best_val = None


        # Train loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for i, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps # Divide loss by accumulation steps so gradients accumulate properly
                loss.backward()
                total_loss += loss.item() * accumulation_steps # Multiply back to get the original loss value for logging

                # Perform optimizer step every accumulation_steps mini-batches
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            
            # In case the last mini-batch doesn't trigger the accumulation step:
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            print(f"[Fold {fold+1}, Epoch {epoch+1}] Train loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f"[Fold {fold+1}, Epoch {epoch+1}] Val Accuracy: {accuracy:.3f}")


            # Update best accuracy and record its corresponding training loss
            if accuracy > fold_best:
                fold_best = accuracy
                best_train_loss_for_best_val = avg_train_loss
            

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "fold": fold,
                "val_accuracy": accuracy,
                "train_loss": avg_train_loss
            })


        print(f"Best validation accuracy for fold {fold+1}: {fold_best:.3f} with training loss: {best_train_loss_for_best_val:.4f}")
        fold_best_scores.append(fold_best)
        fold_best_losses.append(best_train_loss_for_best_val)

        
        """
        # Plot and save loss/accuracy graph for current fold
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, 'r-o', label='Train Loss')
        plt.plot(epochs, val_accuracies, 'b-o', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title(f'Fold {fold+1} - Training Loss and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{graph_name}_Fold_{fold+1}_{dataset_type}_metrics.png')
        plt.close()  # Close the figure to free memory

        print("")  # Blank line between folds
        """


    if ("vivit" in args.model_name):
        graph_name = "Vivit"
    elif ("timesformer" in args.model_name):
        graph_name = "Timesformer"    

    
    avg_score = sum(fold_best_scores) / len(fold_best_scores)
    avg_loss = sum(fold_best_losses) / len(fold_best_losses)
    print(f"Average best validation accuracy across folds: {avg_score:.3f}")
    print(f"Average training loss at best validation accuracy across folds: {avg_loss:.4f}")


    # Save the model
    model.save_pretrained(f"{graph_name.lower()}-finetuned-{dataset_type}-method-{finetune_method}")


    # Finish the wandb run
    wandb.finish()


    # Return last accuracy score and loss to grid_search.py
    return avg_score, avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a test dataset")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of folds for K-fold cross validation"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="balanced_train_f1_data_unmod.csv,balanced_train_f2_data_unmod.csv,balanced_train_f3_data_unmod.csv,balanced_train_f4_data_unmod.csv",
        help="CSV files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Pretrained model or saved fine-tuned model name to use"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        # default="default",
        default="lora",
        help="Fine-tuning method (i.e., default, LoRA)"
    )
    parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler"
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0,
        help="The dropout ratio for the attention probabilities"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha value for LoRA fine-tuning"
    )
    parser.add_argument(
        "--project_title",
        type=str,
        default="Fine-tuned Intent Classification",
        help="Title for the current Wandb project"
    )
    parser.add_argument(
        "--balanceBool",
        type=bool,
        default=True,
        help="Boolean for is using balanced split for dataset or not"
    )
    args = parser.parse_args()
    main(args)

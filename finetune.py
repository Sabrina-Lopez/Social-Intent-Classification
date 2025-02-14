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
from sklearn.model_selection import KFold
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


    # Retrieve and enable the loading of the train data
    data_files = {"train": args.data_file}
    data_csv = load_dataset('csv', data_files=data_files, sep=',')

    data_csv["train"] = data_csv["train"].map(preprocess_function)
    dataset = data_csv["train"].with_format(
        "torch", columns=["video_path", "start", "end", "label"]
    )


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


    # Load the base config from the pretrained checkpoint
    if ("vivit" in args.model_name):
        config = VivitConfig.from_pretrained(base_model_name)
    elif("timesformer" in args.model_name):
        config = TimesformerConfig.from_pretrained(base_model_name)

    # Update number of labels and label mappings
    config.num_labels = num_custom_labels
    config.id2label = id2label
    config.label2id = label2id
    config.num_frames = 16


    # Setup LoRA configuration
    lora_config = None


    # Get the pretrained image processor and model
    if ("vivit" in args.model_name):
        lora_config = LoraConfig(
            target_modules=["query", "key", "value"],
            bias="none"
        )
        image_processor = VivitImageProcessor.from_pretrained(base_model_name)
        model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
    elif("timesformer" in args.model_name):
        lora_config = LoraConfig(
            target_modules=["qkv"],
            bias="none"
        )
        image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)

    
    model = get_peft_model(model, lora_config)
    

    # Declare k-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


    for name, module in model.named_modules():
        print(name, module)


    """
    for name, _ in model.named_parameters():
        print(name)
    """

    """
    # Freeze last hidden layers of model
    for name, param in model.named_parameters():
        # If it's in the classification head, keep it trainable
        if "classifier" in name:
            param.requires_grad = True
        # If it's in the last 2 layers of the encoder, keep them trainable
        elif "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    """
    
    dataset_type = os.path.splitext(data_files["train"])[0]

    name = None
    prefix = 'train_data_'
    substring = dataset_type[len(prefix):]
    if ("vivit" in args.model_name):
        name = "vivit_" + substring + f"_batch_{args.batch_size}"
    elif ("timesformer" in args.model_name):
        name = "times_" + substring + f"_batch_{args.batch_size}"


    # Initialize wandb with project and configuration details
    wandb.init(
        project="Intent Classification Fine-tuning Results",
        config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "k": n_splits,
            "data_type": dataset_type,
            "model_name": base_model_name
        },
        name=name
    )


    # Train the model using k-fold cross-validation and custom dataset
    for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
        print(f"=== Fold {fold+1} / {n_splits} ===")
        
        train_ids = list(map(int, train_ids))
        val_ids = list(map(int, val_ids))

        # Create subsets
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        # Create our custom VideoDataset
        train_dataset = VideoDataset(train_subset, image_processor, sample_frame_indices)
        val_dataset = VideoDataset(val_subset, image_processor, sample_frame_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        
        model = get_peft_model(model, lora_config)


        
        # Freeze all but last layers:
        for name, param in model.named_parameters():
            if "classifier" in name or "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        


        # Setup optimizer for only unfrozen params
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_losses = []
        val_accuracies = []

        # Train loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
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
            val_accuracies.append(accuracy)
            print(f"[Fold {fold+1}, Epoch {epoch+1}] Val Accuracy: {accuracy:.3f}")
            

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "fold": fold,
                "val_accuracy": accuracy,
                "train_loss": avg_train_loss
            })


        if ("vivit" in args.model_name):
            graph_name = "Vivit"
        elif ("timesformer" in args.model_name):
            graph_name = "Timesformer"      


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


    # Save the model
    model.save_pretrained(f"{graph_name.lower()}-finetuned-{dataset_type}")


    # Finish the wandb run
    wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a test dataset")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of folds for K-fold cross validation"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="test_data_33_unmod.csv",
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Pretrained model or saved fine-tuned model name to use"
    )
    args = parser.parse_args()
    main(args)
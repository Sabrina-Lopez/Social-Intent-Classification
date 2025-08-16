import numpy as np
import random
import os
import torch
import wandb
import argparse
from dotenv import load_dotenv
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification, VJEPA2Config
from utils import preprocess_function, sample_frame_indices
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split
from dataloader_class import VideoDataset
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import re
import time

load_dotenv()
wandb.login(key = os.environ.get("WANDB_API_KEY"))


def main(args):
    # Define hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    n_splits = args.k


    # Retrieve and enable the loading of the train data
    data_file = {"train": args.data_file}
    data_csv = load_dataset('csv', data_files=data_file, sep=',')

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
    model = None
    image_processor = None


   # Load the base config from the pretrained checkpoint
    if ("vjepa2" in args.model_name):
        config = VJEPA2Config.from_pretrained(base_model_name)


    ignore_mismatched_sizes = True


    # Update number of labels and label mappings
    config.num_labels = num_custom_labels
    config.id2label = id2label
    config.label2id = label2id
    config.num_frames = 16
    config.hidden_dropout_prob = args.hidden_dropout
    config.attention_probs_dropout_prob = args.attn_dropout
    config.drop_path_rate = args.stoch_depth


    # Based on the fine-tuning method, setup the models accordingly
    finetune_method = args.finetune
    

    # Declare k-fold
    fold_iter = None
    # Set up the fold iterator
    # Single fold: perform a simple train/validation split (80/20)
    indices = list(range(len(dataset)))
    train_ids, val_ids = train_test_split(indices, test_size=0.2, random_state=42)
    fold_iter = [(train_ids, val_ids)]


    # Get the pretrained image processor and model
    if ("vjepa2" in args.model_name):
        image_processor = AutoVideoProcessor.from_pretrained(base_model_name)
        model = VJEPA2ForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes)

        # For fine-tuning model's full backbone
        for name, param in model.named_parameters():
            if ("classifier" in name):
                param.requires_grad = True
            else: param.requires_grad = True

    
    print("=== Processor loaded for:", base_model_name)
    print(image_processor)


    # Get the type of dataset that is being used
    dataset_type = os.path.splitext(args.data_file)[0]

    name = None
    if ("vjepa2" in args.model_name):
        name = "vjepa2"


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
            "hidden_dropout": str(args.hidden_dropout),
            "attn_dropout": str(args.attn_dropout),
            "lr": str(args.lr),
            "w_decay": str(args.decay),
            "drop_path_rate": args.stoch_depth,
            "label_smoothing": args.label_smoothing
        },
        name=name
    )


    # Train the model using k-fold cross-validation and custom dataset
    for fold, (train_ids, val_ids) in enumerate(fold_iter):
    # for fold in range(n_splits):
        print(f"=== Fold {fold+1} / {n_splits} ===")

        train_ids = list(map(int, train_ids))
        val_ids = list(map(int, val_ids))

        # Create subsets
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        # Create our custom VideoDataset
        train_dataset = VideoDataset(train_subset, image_processor, sample_frame_indices)
        val_dataset = VideoDataset(val_subset, image_processor, sample_frame_indices)


        accumulation_steps = None
        # If fine-tuning with LoRA and a batch size of 16 is provided,
        # use an effective batch size of 8 and accumulate gradients over 2 mini-batches
        if (args.batch_size == 16):
            effective_batch_size = 8
            accumulation_steps = 2
        else:
            effective_batch_size = args.batch_size
            accumulation_steps = 1


        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Re-initialize or re-load the model each fold to keep re-frozen structure
        if ("vjepa2" in args.model_name):
            model = VJEPA2ForVideoClassification.from_pretrained(
                base_model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
       
        # For fine-tuning model backbone
        for name, param in model.named_parameters():
            if ("classifier" in name):
                param.requires_grad = True
            else: param.requires_grad = True     


        # Setup optimizer for only unfrozen params
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(num_epochs):
            # Training Loop
            model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0

            for i, batch in enumerate(train_loader):
                pixel_values = None
                if "vjepa2" in args.model_name:
                    pixel_values = batch["pixel_values_videos"].to(device)
                else: pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                # Check for invalid inputs before forward
                if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                    print("❌ Problem with batch: Pixel values contain NaN or Inf.")

                optimizer.zero_grad()

                try:
                    if ("vjepa2" in args.model_name):
                        outputs = model(pixel_values_videos=pixel_values)
                    else:
                        outputs = model(pixel_values=pixel_values)

                    logits = outputs.logits

                    # Debug info
                    print("Logits stats (train):", logits.mean().item(), logits.std().item())

                    # Check for invalid logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("❌ Problem with batch: NaN or Inf in logits (train)")

                    loss = criterion(logits, labels)

                    # Check for NaN in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("❌ Problem with batch: NaN or Inf in loss (train)")

                    loss = loss / accumulation_steps
                    loss.backward()

                    # Clip gradients to prevent exploding
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    total_train_loss += loss.item() * accumulation_steps  # Un-normalized loss

                    # Training accuracy
                    preds = torch.argmax(logits, dim=-1)
                    correct_train += (preds == labels).sum().item()
                    total_train += labels.size(0)

                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                except Exception as e:
                    print(f"❌ Problem with batch due to unexpected error: {e}")

            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = correct_train / total_train
            print(f"[Fold {fold+1}, Epoch {epoch+1}] ✅ Train loss: {avg_train_loss:.4f}, acc: {train_acc:.3f}")

            # Validation Loop
            model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = None
                    if "vjepa2" in args.model_name:
                        pixel_values = batch["pixel_values_videos"].to(device)
                    else: pixel_values = batch["pixel_values"].to(device)
                    labels = batch["label"].to(device)

                    if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                        print("❌ Problem with val batch: Pixel values contain NaN or Inf.")

                    try:
                        if ("vjepa2" in args.model_name):
                            outputs = model(pixel_values_videos=pixel_values)
                        else:
                            outputs = model(pixel_values=pixel_values)

                        logits = outputs.logits

                        print("Logits stats (val):", logits.mean().item(), logits.std().item())

                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            print("❌ Problem with val batch: NaN or Inf in logits")

                        loss = criterion(logits, labels)

                        if torch.isnan(loss) or torch.isinf(loss):
                            print("❌ Problem with val batch: NaN or Inf in loss (val)")

                        total_val_loss += loss.item()

                        preds = torch.argmax(logits, dim=-1)
                        correct_val += (preds == labels).sum().item()
                        total_val += labels.size(0)

                    except Exception as e:
                        print(f"❌ Problem with val batch due to unexpected error: {e}")

            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct_val / total_val
            print(f"[Fold {fold+1}, Epoch {epoch+1}] ✅ Val loss: {avg_val_loss:.4f}, acc: {val_acc:.3f}")

            # Log to Wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc
            })  

        
        graph_name = args.model_name    

        # Save the model; ensure saving to a directory folder
        os.makedirs(args.saved_model_dir, exist_ok=True)

        # Specify hyperparams in model name for easier reference post-testing
        # Build model + processor save paths
        model_folder = (
            f"{graph_name.lower()}-default-finetuned-{dataset_type}"
            f"-method-{finetune_method}"
            f"-bs-{args.batch_size}"
            f"-k-{args.k}"
            f"-hd-{str(args.hidden_dropout).replace('.', '')}"
            f"-ad-{str(args.attn_dropout).replace('.', '')}"
            f"-lr-{str(args.lr).replace('.', '')}"
            f"-wd-{str(args.decay).replace('.', '')}"
            f"-ls-{str(args.label_smoothing).replace('.', '')}"
            f"-sd-{str(args.stoch_depth).replace('.', '')}"
        )


        # Ensure unique folder name
        model_folder = f"{model_folder}_{int(time.time())}"

        model_save_path = os.path.join(args.saved_model_dir, model_folder)
        processor_save_path = os.path.join(args.saved_model_dir, model_folder)

        # Save model into saved model output directory
        model.save_pretrained(model_save_path)
        image_processor.save_pretrained(processor_save_path)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of folds for K-fold cross validation"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="train_data_100_unmod_latest.csv",
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/vjepa2-vitl-fpc32-256-diving48",
        help="Pretrained model or saved fine-tuned model name to use"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default="default",
        help="Fine-tuning method"
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
        "--lr",
        type=float,
        default=0.0001, # Our fine-tuning default
        help="The learning rate"
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.01, # Adam default, our fine-tuning default
        help="The weight decay"
    )
    parser.add_argument(
        "--project_title",
        type=str,
        default="Fine-tuned Intent Classification 6-Model Training",
        help="Title for the current Wandb project"
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default="./saved_model_dir",
        help="Directory to save the fine-tuned models and processors"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross entropy loss"
    )
    parser.add_argument(
        "--stoch_depth",
        type=float,
        default=0.0,
        help="Stochastic depth, which will randomly drop entire transformer layers during training to encourage robustness"
    )
    args = parser.parse_args()
    main(args)
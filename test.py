import numpy as np
import random
import os
import torch
import wandb
import argparse
from dotenv import load_dotenv
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from utils import preprocess_function, custom_collate_fn, evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


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


    # Retrieve and enable the loading of the test data
    data_files = {"test": args.data_file}
    data_csv = load_dataset('csv', data_files=data_files, sep=',')

    data_csv["test"] = data_csv["test"].map(preprocess_function)
    test_dataset = data_csv["test"].with_format(
        "torch", columns=["video_path", "start", "end", "label"]
    )

    batch_size = args.batch_size
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
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
    finetune_bool = False
    graph_name = None


    finetune_method = args.finetune


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

    # Get the pretrained image processor and model
    if ("vivit" in args.model_name):
        image_processor = VivitImageProcessor.from_pretrained(base_model_name)
        model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
    elif ("timesformer" in args.model_name):
        image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)


    dataset_type = os.path.splitext(data_files["test"])[0]

    name = None
    prefix = 'test_data_'
    substring = dataset_type[len(prefix):]
    if ("vivit" in args.model_name):
        name = "vivit_" + substring
    elif ("timesformer" in args.model_name):
        name = "times_" + substring


    if ("finetuned-train" in args.model_name):
        finetune_bool = True


    # Initialize wandb with project and configuration details
    wandb.init(
        project="Intent Classification Testing Results",
        config={
            "batch_size": batch_size,
            "data_type": dataset_type,
            "model_name": base_model_name,
            "finetune_bool": finetune_bool,
            "finetune": finetune_method,
        },
        name=name
    )


    # Evaluate the testing dataset
    all_preds, all_labels, losses, accuracies = evaluate(model, test_loader, image_processor)

    accuracy = accuracy_score(all_labels, all_preds)
    # Get classification report as a dictionary
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=["help", "hinder", "physical"],
        output_dict=True
    )

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["help", "hinder", "physical"]))


    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        # Per-class metrics
        "precision_help": report_dict["help"]["precision"],
        "precision_hinder": report_dict["hinder"]["precision"],
        "precision_physical": report_dict["physical"]["precision"],
        "recall_help": report_dict["help"]["recall"],
        "recall_hinder": report_dict["hinder"]["recall"],
        "recall_physical": report_dict["physical"]["recall"],
        "f1_help": report_dict["help"]["f1-score"],
        "f1_hinder": report_dict["hinder"]["f1-score"],
        "f1_physical": report_dict["physical"]["f1-score"],
        "support_help": report_dict["help"]["support"],
        "support_hinder": report_dict["hinder"]["support"],
        "support_physical": report_dict["physical"]["support"],
        # Macro averages
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        # Weighted averages
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"]
    })


    if (finetune_bool == False):
        if ("vivit" in args.model_name):
            graph_name = "Unfinetune_Vivit"
        elif ("timesformer" in args.model_name):
            graph_name = "Unfinetune_Timesformer"
    else:
        if ("vivit" in args.model_name):
            graph_name = "Finetune_Vivit"
        elif ("timesformer" in args.model_name):
            graph_name = "Finetune_Timesformer"

    """
    # Plot and save loss/accuracy graph for testing
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'r-o', label='Test Loss')
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-o', label='Test Accuracy')
    plt.xlabel('Batch Number')
    plt.ylabel('Metric Value')
    plt.title(f'Evaluation - Testing Loss and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{graph_name}_{dataset_type}_metrics.png')
    plt.close()  # Close the figure to free memory
    """

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
    parser.add_argument(
        "--finetune",
        type=str,
        default="default",
        help="Fine-tuning method used to fine-tune the model being tested (i.e., default, LoRA)"
    )
    args = parser.parse_args()
    main(args)

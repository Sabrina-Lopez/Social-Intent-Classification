from dotenv import load_dotenv
import os
import wandb
import argparse
import numpy as np
from transformers import VivitImageProcessor
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from utils import read_video_pyav, preprocess_function, custom_collate_fn, sample_frame_indices
from tensorflow.keras.layers import TFSMLayer
from wandb.integration.keras import WandbCallback
import av


load_dotenv()
wandb.login(key=os.environ.get("WANDB_API_KEY"))


def inspect_and_load_saved_model(sm_path: str):
    """Load SavedModel on CPU, return trackable, serving function, and signature info."""
    with tf.device("/CPU:0"):
        sm = tf.saved_model.load(sm_path)
        fn = sm.signatures["serving_default"]
        in_sig_map = fn.structured_input_signature[1]
        input_keys = list(in_sig_map.keys())
        input_key = input_keys[0] if input_keys else "inputs"
        input_spec = in_sig_map[input_key]
        expected_T = int(input_spec.shape[1])  # [B, T, H, W, C]
        sig_dtype = input_spec.dtype
        output_keys = list(fn.structured_outputs.keys())
    print("=== SavedModel signature ===")
    print("Input keys:", input_keys)
    print("Output keys:", output_keys)
    print("Expected T:", expected_T, "Signature dtype:", sig_dtype.name)
    return sm, fn, input_key, output_keys, expected_T, sig_dtype


class SavedModelWrapper(tf.keras.layers.Layer):
    """Wrap a SavedModel 'serving_default' as a Keras Layer, keeping variables trackable."""
    def __init__(self, sm, serving_fn, input_key, unfreeze: bool):
        super().__init__(trainable=unfreeze, name="saved_model_wrapper")
        # Track the loaded module as an attribute so Keras sees its variables
        self.sm = sm
        self.serving_fn = serving_fn
        self.input_key = input_key

    def call(self, x):
        # Forward through the serving function; returns a dict
        outputs = self.serving_fn(**{self.input_key: x})
        return outputs


def _pad_or_truncate_t(pix: np.ndarray, clip_len: int) -> np.ndarray:
    """Pad last frame or truncate to match clip_len along time dimension."""
    t = pix.shape[0]
    if t < clip_len:
        pad = np.repeat(pix[-1:], clip_len - t, axis=0)
        return np.concatenate([pix, pad], axis=0)
    return pix[:clip_len]


def _extract_frames_from_item(item, clip_len: int):
    """Read a fixed-length clip from a video segment using PyAV."""
    start_s = float(item.get("start") or 0.0)
    end_s = float(item.get("end") or 0.0)
    path = item["video_path"]
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        total_frames = int(stream.frames or 0)
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps) or total_frames
        seg_len = max(0, end_frame - start_frame)
        if seg_len == 0:
            raise ValueError("Empty segment.")
        rel_idx = sample_frame_indices(clip_len=clip_len, seg_len=seg_len)
        abs_idx = (rel_idx + start_frame).astype(np.int64)
        frames = read_video_pyav(container, abs_idx)  # returns list/array of PIL/ndarray frames
    return frames


def make_tf_dataset(
    hf_dataset,
    batch_size: int,
    image_processor: VivitImageProcessor,
    label2id: dict,
    clip_len: int,
    pixel_range: str,
):
    """HF dataset → tf.data.Dataset with fixed (T,H,W,C)= (clip_len,224,224,3) float32 pixels and int labels."""

    def gen():
        for item in hf_dataset:
            try:
                frames = _extract_frames_from_item(item, clip_len)
                # HF VivitImageProcessor outputs (T, C, H, W) in [-1, 1]
                pix = image_processor(list(frames), return_tensors="np")["pixel_values"][0]
                pix = np.transpose(pix, (0, 2, 3, 1)).astype(np.float32)  # (T,H,W,C)
                pix = _pad_or_truncate_t(pix, clip_len)
                lbl = label2id.get(item["label"], int(item["label"]))
                yield pix, lbl
            except Exception as e:
                print(f"[WARN] Skipped {item.get('video_path','<unknown>')} due to error: {e}")

    output_signature = (
        tf.TensorSpec(shape=(clip_len, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def scale_pixels(x, y):
        # Processor gives [-1,1] by default. Map per user flag to match SavedModel expectations.
        if pixel_range == "zero_one":
            x = (x + 1.0) / 2.0
        elif pixel_range == "none":
            x = tf.clip_by_value(x * 127.5 + 127.5, 0.0, 255.0)
        return x, y

    ds = ds.map(scale_pixels, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(
    sm,
    serving_fn,
    input_key: str,
    output_keys: list,
    sig_dtype: tf.dtypes.DType,
    expected_T: int,
    clip_len: int,
    num_classes: int,
    unfreeze: bool,
    lr: float,
):
    """Assemble a Keras model with temporal adaptation, dtype cast, SavedModel core, and optional classifier head."""
    inp = Input(shape=(clip_len, 224, 224, 3), dtype=tf.float32, name=input_key)

    # Temporal adapter to match SavedModel's expected T
    def temporal_adapter(x):
        if expected_T == clip_len:
            return x
        elif expected_T % clip_len == 0:
            return tf.repeat(x, repeats=expected_T // clip_len, axis=1)
        else:
            pad = expected_T - clip_len
            tail = tf.repeat(x[:, -1:, ...], repeats=pad, axis=1)
            return tf.concat([x, tail], axis=1)

    x = Lambda(temporal_adapter, name="temporal_adapter")(inp)
    x = Lambda(lambda t: tf.cast(t, sig_dtype), name="cast_to_sig_dtype")(x)

    # SavedModel core
    core = SavedModelWrapper(sm, serving_fn, input_key, unfreeze=unfreeze)
    out_dict = core(x)

    # Choose the most standard output
    feats = None
    if "logits" in out_dict:
        feats = out_dict["logits"]
        base_is_logits = True
    elif "probabilities" in out_dict:
        feats = out_dict["probabilities"]
        base_is_logits = False
    else:
        # Fallback to first output key deterministically
        first_key = output_keys[0]
        feats = out_dict[first_key]
        # Heuristic: if name contains 'logit' assume logits; else assume probabilities
        base_is_logits = ("logit" in first_key.lower())

    # Squeeze a singleton temporal dim if present
    feats_shape = feats.shape
    if len(feats_shape) == 3 and feats_shape[1] == 1:
        feats = tf.squeeze(feats, axis=1)

    # If SavedModel output dimension != our classes, add a classifier head
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
        print("check")
    need_head = (feats.shape[-1] != num_classes)
    if need_head:
        out = Dense(num_classes, activation="softmax", name="classifier")(feats)
        from_logits = False
    else:
        out = feats
        from_logits = base_is_logits

    model = Model(inputs=inp, outputs=out, name="vivit_finetune")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=[SparseCategoricalAccuracy()],
    )
    return model


def train_model(model, train_ds, val_ds, args, expected_T, sig_dtype):
    wandb.init(
        project=args.project_title,
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "clip_len": args.clip_len,
            "unfreeze": args.unfreeze,
            "expected_T": expected_T,
            "sig_dtype": str(sig_dtype.name),
            "pixel_range": args.pixel_range,
        },
        name="vivit-tf-finetune",
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[WandbCallback(log_weights=False, log_gradients=False)],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "vivit_finetuned.keras"))
    tf.saved_model.save(model, os.path.join(args.output_dir, "saved_vivit_fe_model"))

    wandb.finish()


def main(args):
    # Load and split HF dataset
    data = load_dataset("csv", data_files={"train": args.data_file}, sep=",")["train"]
    # Keep hook to existing preprocess if needed
    data = data.map(preprocess_function, num_proc=4)
    idx_train, idx_val = train_test_split(range(len(data)), test_size=0.2, random_state=42)
    ds_train_hf = data.select(idx_train)
    ds_val_hf = data.select(idx_val)

    # Labels and processor
    label2id = {"help": 0, "hinder": 1, "physical": 2}
    num_classes = len(label2id)
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    # Load SavedModel once and inspect
    sm, serving_fn, input_key, output_keys, expected_T, sig_dtype = inspect_and_load_saved_model(args.tf_model_path)

    # Build tf.data datasets
    train_ds = make_tf_dataset(
        ds_train_hf,
        batch_size=args.batch_size,
        image_processor=image_processor,
        label2id=label2id,
        clip_len=args.clip_len,
        pixel_range=args.pixel_range,
    )
    val_ds = make_tf_dataset(
        ds_val_hf,
        batch_size=args.batch_size,
        image_processor=image_processor,
        label2id=label2id,
        clip_len=args.clip_len,
        pixel_range=args.pixel_range,
    )

    # Build and train model
    model = build_model(
        sm=sm,
        serving_fn=serving_fn,
        input_key=input_key,
        output_keys=output_keys,
        sig_dtype=sig_dtype,
        expected_T=expected_T,
        clip_len=args.clip_len,
        num_classes=num_classes,
        unfreeze=args.unfreeze,
        lr=args.lr,
    )

    train_model(model, train_ds, val_ds, args, expected_T, sig_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tune TF ViViT")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_file", type=str, default="./train_data_100_unmod_latest.csv")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--tf_model_path", type=str, default="./vivit-factorized-files/tf_saved_model")
    parser.add_argument("--project_title", type=str, default="ViViT-TF-Finetuning")
    parser.add_argument("--output_dir", type=str, default="./tf_finetuned_model")
    parser.add_argument("--pixel_range", type=str, default="hf", choices=["hf", "zero_one", "none"],
                        help="If your SavedModel expects [0,1], use 'zero_one'. Leave 'hf' ([-1,1]) otherwise. Use 'none' for [0,255].")
    parser.add_argument("--unfreeze", action="store_true", default=True, help="Set True to fine-tune the base SavedModel.")
    args = parser.parse_args()
    main(args)
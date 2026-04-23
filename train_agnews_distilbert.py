import argparse
import os
import numpy as np
import inspect
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="agnews-distilbert")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")

    # 🔥 GPU-optimized defaults
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--demo", action="store_true", help="Run a fast demo with small subset and 1 epoch")
    p.add_argument("--max_train_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")

    if args.demo:
        print("Demo mode: subsampling dataset for a fast run")
        dataset["train"] = dataset["train"].select(range(3000))
        dataset["test"] = dataset["test"].select(range(500))
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 32
        args.per_device_eval_batch_size = 64

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 🔥 Reduced max_length for speed
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model (num_labels=4)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=4
    )
    model.to(device)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return accuracy.compute(predictions=preds, references=labels)

    # ---------------- TrainingArguments (adaptive + optimized) ----------------
    wanted_kwargs = dict(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),

        # 🔥 Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
    )

    sig = inspect.signature(TrainingArguments)
    supported = set(sig.parameters.keys())

    mappings = {}
    if "evaluation_strategy" not in supported and "eval_strategy" in supported:
        mappings["eval_strategy"] = wanted_kwargs.pop("evaluation_strategy")
    if "logging_strategy" not in supported:
        wanted_kwargs.pop("logging_strategy", None)
    if "save_strategy" not in supported:
        wanted_kwargs.pop("save_strategy", None)

    for k, v in mappings.items():
        wanted_kwargs[k] = v

    filtered_kwargs = {k: v for k, v in wanted_kwargs.items() if k in supported}

    training_args = TrainingArguments(**filtered_kwargs)

    print("TrainingArguments keys:", list(filtered_kwargs.keys()))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---------------- Quick inference ----------------
    print("\n=== Quick inference examples ===")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    examples = [
        "Apple announces the release of its new M-series laptop.",
        "The national team won the championship after a thrilling final.",
        "The central bank raised interest rates today.",
        "Global leaders met to discuss climate policy.",
    ]

    classifier = pipeline_inference(args.output_dir, device=device)
    for t in examples:
        out = classifier(t)[0]
        label_id = int(out["label"].split("_")[-1])
        print(f"TEXT: {t}\nPRED: {labels[label_id]}, SCORE: {out['score']:.3f}\n")


def pipeline_inference(model_dir, device="cpu"):
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir)
    m.to(device)

    pipeline_device = 0 if device == "cuda" else -1
    return pipeline(
        "text-classification",
        model=m,
        tokenizer=tok,
        device=pipeline_device,
    )


if __name__ == "__main__":
    main()

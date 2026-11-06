import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from .clipping import ClippedAdam, GradientClipper, OPTIMIZER_METHODS, norm
from .metrics import calibrated_results, classification_metrics, fit_temperature, prediction_drift
from .models import CTRLogistic, CTRMLP


GRAD_FIELDS = (
    "step", "phase", "epoch", "loss", "raw_norm", "clipped_norm", "coef",
    "triggered", "moment_reset", "injected_spike", "lr", "update_norm", "time_ms",
)


def load_split(path, metadata):
    table = pq.read_table(path)
    categories = np.column_stack([table[key].to_numpy() for key in metadata["categorical"]])
    numbers = np.column_stack([table[key].to_numpy() for key in metadata["numeric"]])
    labels = table["label"].to_numpy().astype(np.float32)
    return table["row_id"].to_numpy(), (
        torch.from_numpy(categories).long(), torch.from_numpy(numbers), torch.from_numpy(labels)
    )


def predict(model, split, batch_size):
    model.eval()
    ids, (categories, numbers, labels) = split
    with torch.inference_mode():
        logits = torch.cat([
            model(cat_batch, num_batch)
            for cat_batch, num_batch in zip(categories.split(batch_size), numbers.split(batch_size))
        ])
    predictions = logits.sigmoid().numpy()
    return pa.table({
        "row_id": ids, "label": labels.numpy(), "logit": logits.numpy(), "prediction": predictions,
    })


def scores(predictions):
    return classification_metrics(predictions["label"].to_numpy(),
                                  predictions["prediction"].to_numpy())


class Training:
    def __init__(self, config, metadata):
        self.batch_size = config["batch_size"]
        if self.batch_size < 2:
            raise ValueError("BatchNorm requires batch size >= 2")
        architecture = config.get("architecture", "mlp")
        inputs = (metadata["cardinalities"], len(metadata["numeric"]))
        if architecture == "mlp":
            self.model = CTRMLP(*inputs, config["embedding_dim"], config["dropout"])
        elif architecture == "logistic":
            self.model = CTRLogistic(*inputs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.parameters = list(self.model.parameters())
        method = config["method"]
        options = config.get("method_options", {})
        betas = tuple(config.get("betas", (0.9, 0.999)))
        kwargs = dict(lr=config["lr"], betas=betas, weight_decay=config["weight_decay"])
        if method in OPTIMIZER_METHODS:
            self.optimizer = ClippedAdam(self.parameters, method, **kwargs, **options)
            self.clipper = GradientClipper("none")
        else:
            self.optimizer = torch.optim.AdamW(self.parameters, **kwargs)
            self.clipper = GradientClipper(method, **options)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.steps = 0
        self.spike_steps = set(config.get("spike_steps", ()))
        self.spike_factor = config.get("spike_factor", 1.0)

    def epoch(self, data, writer, phase, epoch, lr, warmup=0):
        self.model.train()
        rows = len(data[2])
        if rows < 2:
            raise ValueError("BatchNorm requires at least two training rows")
        batches = list(torch.randperm(rows).split(self.batch_size))
        if len(batches[-1]) == 1:
            batches[-2:] = [torch.cat(batches[-2:])]
        epoch_loss = 0.0
        for indices in batches:
            step_start = time.perf_counter()
            self.steps += 1
            step_lr = lr * min(1.0, self.steps / warmup) if warmup else lr
            self.optimizer.param_groups[0]["lr"] = step_lr
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_function(
                self.model(data[0][indices], data[1][indices]), data[2][indices],
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at step {self.steps}")
            loss.backward()
            injected = self.steps in self.spike_steps
            if injected:
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(self.spike_factor)
            raw, clipped, triggered = self.clipper.clip(self.parameters)
            previous = [p.detach().clone() for p in self.parameters]
            self.optimizer.step()
            reset = False
            if isinstance(self.optimizer, ClippedAdam):
                stats = self.optimizer.last_stats
                clipped, triggered, reset = (
                    stats["moment_input_norm"], stats["triggered"], stats["moment_reset"]
                )
            update = norm(p.detach() - old for p, old in zip(self.parameters, previous))
            if not math.isfinite(update):
                raise FloatingPointError(f"Non-finite update at step {self.steps}")
            value = loss.item()
            epoch_loss += value * len(indices)
            writer.writerow(dict(
                step=self.steps, phase=phase, epoch=epoch, loss=value, raw_norm=raw,
                clipped_norm=clipped, coef=clipped / raw if raw else 1.0,
                triggered=int(triggered), moment_reset=int(reset), injected_spike=int(injected), lr=step_lr,
                update_norm=update, time_ms=(time.perf_counter() - step_start) * 1000,
            ))
        return epoch_loss / rows


def evaluate_daily(trainer, load, validation_predictions, validation, config, out, writer, log):
    frozen = copy.deepcopy(trainer.model)
    temperature = fit_temperature(validation_predictions)
    prior_reference = validation_predictions["prediction"].to_numpy()
    records = []
    tables = []
    for day in range(6, 10):
        name = f"day_{day}"
        daily = load(name)
        for variant, model in (("frozen", frozen), ("updated", trainer.model)):
            predictions = predict(model, daily, trainer.batch_size)
            metrics = scores(predictions)
            calibration, predictions = calibrated_results(predictions, temperature)
            record = {"day": day, "variant": variant, **metrics, "calibration": calibration}
            if variant == "updated":
                reference = predict(model, validation, trainer.batch_size)["prediction"].to_numpy()
                record["reference_drift"] = prediction_drift(prior_reference, reference)
                prior_reference = reference
            records.append(record)
            tables.append(predictions.append_column("day", pa.repeat(day, len(predictions)))
                          .append_column("variant", pa.repeat(variant, len(predictions))))
            print(f"day={day} {variant}: auc={metrics['auc']:.6f} "
                  f"logloss={metrics['logloss']:.6f} ece={metrics['ece']:.6f}", file=log)
        if day < 9:
            for epoch in range(1, config.get("finetune_epochs", 1) + 1):
                trainer.epoch(
                    daily[1], writer, name, epoch, config.get("finetune_lr", 0.0003),
                )
    pq.write_table(pa.concat_tables(tables), out / "predictions.parquet")
    with (out / "per_day_metrics.csv").open("w", newline="") as stream:
        fields = ("day", "variant", "rows", "ctr", "auc", "logloss", "brier", "ece")
        table = csv.DictWriter(stream, fields, extrasaction="ignore")
        table.writeheader()
        table.writerows(records)
    return records


def run(config, out):
    start = time.perf_counter()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("threads", 4))
    torch.use_deterministic_algorithms(True)

    directory = Path(config["data"])
    metadata = json.loads((directory / "stats.json").read_text())

    def load(name):
        return load_split(directory / f"{name}.parquet", metadata)

    daily = config.get("mode", "static") == "daily"
    if daily:
        if metadata["dataset"] not in ("avazu", "criteo_attribution"):
            raise ValueError("Daily mode requires a temporal dataset")
        days = [load(f"day_{i}")[1] for i in range(5)]
        train = tuple(torch.cat(parts) for parts in zip(*days))
        del days
        validation = load("day_5")
    else:
        _, train = load("train")
        validation = load("validation")
    trainer = Training(config, metadata)
    method = config["method"]
    epochs = []
    metrics = {
        "seed": seed, "method": method, "source_sha256": metadata["source_sha256"],
        "architecture": config.get("architecture", "mlp"),
        "train_rows": len(train[2]), "parameters": sum(p.numel() for p in trainer.parameters),
        "epochs": epochs,
    }
    with (out / "grad_stats.csv").open("w", newline="") as grad_file, (
        out / "train.log"
    ).open("w", buffering=1) as log:
        writer = csv.DictWriter(grad_file, fieldnames=GRAD_FIELDS)
        writer.writeheader()
        print(
            f"dataset={metadata['dataset']} source_sha256={metadata['source_sha256']} "
            f"seed={seed} method={method} torch={torch.__version__}",
            file=log,
        )
        try:
            for epoch in range(1, config["epochs"] + 1):
                loss = trainer.epoch(
                    train, writer, "pretrain" if daily else "train", epoch,
                    config["lr"], config.get("warmup_steps", 0),
                )
                predictions = predict(trainer.model, validation, trainer.batch_size)
                epochs.append({"epoch": epoch, "train_loss": loss, **scores(predictions)})
                print(json.dumps({key: value for key, value in epochs[-1].items()
                                  if key != "reliability"}), file=log)
            if daily:
                del train
                metrics["daily"] = evaluate_daily(
                    trainer, load, predictions, validation, config, out, writer, log,
                )
            elif config.get("evaluate_test", False):
                temperature = fit_temperature(predictions)
                test = predict(trainer.model, load("test"), trainer.batch_size)
                metrics["test"] = scores(test)
                metrics["calibration"], test = calibrated_results(test, temperature)
                pq.write_table(test, out / "predictions.parquet")
        except FloatingPointError as error:
            print(f"Step {trainer.steps}: {error}", file=log)
            raise
        metrics["steps"] = trainer.steps
        metrics["seconds"] = time.perf_counter() - start
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        print(f"Completed {trainer.steps} steps in {metrics['seconds']:.2f} s", file=log)
    print(f"{out}: steps={trainer.steps}, validation AUC={epochs[-1]['auc']:.6f}, "
          f"validation LogLoss={epochs[-1]['logloss']:.6f}", flush=True)

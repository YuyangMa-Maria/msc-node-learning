"""Train a lightweight multi-task VSN with damage, defect and quality heads.

A shared tiny encoder supports binary visible damage, five non-ordinal defect
labels and synthetic degradation classification. The auxiliary quality task is
evaluated as a robustness aid; it is not assumed to solve out-of-distribution
scene recognition or to represent structural severity.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vsn_codebrim_ptq_robustness import (  # noqa: E402
    CONDITIONS,
    apply_corruption,
)
from evaluate_vsn_sdnet_zero_shot import binary_metrics, sigmoid  # noqa: E402
from train_vsn_codebrim_students import fit_temperature  # noqa: E402
from train_vsn_codebrim_teachers import (  # noqa: E402
    CodebrimTeacher,
    DEFECT_LABELS,
)
from train_vsn_student_baseline import VsnStudentDwCnn  # noqa: E402


QUALITY_TYPES = (
    "clean",
    "low_light",
    "blur",
    "gaussian_noise",
    "occlusion",
    "jpeg",
)
QUALITY_TYPE_TO_INDEX = {name: index for index, name in enumerate(QUALITY_TYPES)}
QUALITY_TARGETS = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.25}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VARIANTS = ("multilabel", "multitask_quality")


class DeployableMultiTaskVsn(nn.Module):
    def __init__(self, quality_heads: bool) -> None:
        super().__init__()
        self.student = VsnStudentDwCnn(width=1.0, dropout=0.1)
        self.defect_head = nn.Linear(96, len(DEFECT_LABELS))
        self.quality_type_head = (
            nn.Linear(96, len(QUALITY_TYPES)) if quality_heads else None
        )
        self.quality_score_head = nn.Linear(96, 1) if quality_heads else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | None]:
        feature_map = self.student.features(images)
        embedding = self.student.head[1](self.student.head[0](feature_map))
        shared = self.student.head[2](embedding)
        return {
            "binary": self.student.head[3](shared).flatten(),
            "defects": self.defect_head(shared),
            "quality_type": (
                self.quality_type_head(shared)
                if self.quality_type_head is not None
                else None
            ),
            "quality_score": (
                self.quality_score_head(shared).flatten()
                if self.quality_score_head is not None
                else None
            ),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def random_quality_view(image: Image.Image) -> tuple[Image.Image, int, int, float]:
    quality_name = random.choice(QUALITY_TYPES)
    if quality_name == "clean":
        severity = 0
        output = image
    else:
        severity = random.choice((1, 2, 3))
        output = apply_corruption(
            image,
            quality_name,
            severity,
            random.getrandbits(31),
        )
    return (
        output,
        QUALITY_TYPE_TO_INDEX[quality_name],
        severity,
        QUALITY_TARGETS[severity],
    )


class TrainingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        student_size: int,
        teacher_size: int,
        quality_heads: bool,
    ) -> None:
        self.rows = rows
        self.quality_heads = quality_heads
        self.shared_augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
        self.student_post = image_transform(student_size)
        self.teacher_post = image_transform(teacher_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as opened:
            shared_image = self.shared_augmentation(opened.convert("RGB"))
            clean_student = self.student_post(shared_image)
            teacher_image = self.teacher_post(shared_image)
            if self.quality_heads:
                quality_image, quality_type, severity, quality_target = (
                    random_quality_view(shared_image)
                )
                quality_student = self.student_post(quality_image)
            else:
                quality_student = torch.empty(0)
                quality_type = 0
                severity = 0
                quality_target = 1.0
        damage = torch.tensor(float(row["damage"]), dtype=torch.float32)
        defects = torch.tensor(
            [float(row[label]) for label in DEFECT_LABELS],
            dtype=torch.float32,
        )
        return (
            clean_student,
            quality_student,
            teacher_image,
            damage,
            defects,
            torch.tensor(quality_type, dtype=torch.long),
            torch.tensor(severity, dtype=torch.long),
            torch.tensor(quality_target, dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class CleanEvaluationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int) -> None:
        self.rows = rows
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as opened:
            image = self.transform(opened.convert("RGB"))
        return (
            image,
            torch.tensor(float(row["damage"]), dtype=torch.float32),
            torch.tensor(
                [float(row[label]) for label in DEFECT_LABELS],
                dtype=torch.float32,
            ),
            torch.tensor(index, dtype=torch.long),
        )


class QualityEvaluationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        mode: str,
        corruption: str | None = None,
        severity: int | None = None,
        seed: int = 91_000,
    ) -> None:
        self.rows = rows
        self.transform = image_transform(image_size)
        self.mode = mode
        self.corruption = corruption
        self.severity = severity
        self.seed = seed
        if mode not in ("mixed", "condition"):
            raise ValueError(f"Unsupported quality evaluation mode: {mode}")
        if mode == "condition" and (corruption is None or severity is None):
            raise ValueError("Condition mode requires corruption and severity")

    def __len__(self) -> int:
        return len(self.rows)

    def condition_for_index(self, index: int) -> tuple[str, int]:
        if self.mode == "mixed":
            return CONDITIONS[index % len(CONDITIONS)]
        assert self.corruption is not None and self.severity is not None
        return self.corruption, self.severity

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        corruption, severity = self.condition_for_index(index)
        with Image.open(PROJECT_ROOT / row["path"]) as opened:
            image = apply_corruption(
                opened.convert("RGB"),
                corruption,
                severity,
                self.seed + index,
            )
            tensor = self.transform(image)
        return (
            tensor,
            torch.tensor(float(row["damage"]), dtype=torch.float32),
            torch.tensor(
                [float(row[label]) for label in DEFECT_LABELS],
                dtype=torch.float32,
            ),
            torch.tensor(
                QUALITY_TYPE_TO_INDEX[corruption],
                dtype=torch.long,
            ),
            torch.tensor(severity, dtype=torch.long),
            torch.tensor(QUALITY_TARGETS[severity], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def safe_class_metric(
    metric: Any,
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    try:
        return float(metric(labels, scores))
    except ValueError:
        return float("nan")


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    per_class: dict[str, dict[str, float]] = {}
    for index, label_name in enumerate(DEFECT_LABELS):
        per_class[label_name] = {
            "prevalence": float(labels[:, index].mean()),
            "f1": float(
                f1_score(
                    labels[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "average_precision": safe_class_metric(
                average_precision_score,
                labels[:, index],
                probabilities[:, index],
            ),
            "roc_auc": safe_class_metric(
                roc_auc_score,
                labels[:, index],
                probabilities[:, index],
            ),
        }
    return {
        "samples": int(labels.shape[0]),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "macro_average_precision": float(
            average_precision_score(labels, probabilities, average="macro")
        ),
        "micro_average_precision": float(
            average_precision_score(labels, probabilities, average="micro")
        ),
        "exact_match": float(np.all(labels == predictions, axis=1).mean()),
        "per_class": per_class,
    }


def entropy_confidence(probabilities: np.ndarray) -> np.ndarray:
    bounded = np.clip(probabilities, 1.0e-6, 1.0 - 1.0e-6)
    entropy = -(
        bounded * np.log(bounded)
        + (1.0 - bounded) * np.log(1.0 - bounded)
    ) / math.log(2.0)
    return 1.0 - entropy


def collect_clean_outputs(
    model: DeployableMultiTaskVsn,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    count = len(loader.dataset)
    binary_logits = np.empty(count, dtype=np.float32)
    defect_logits = np.empty((count, len(DEFECT_LABELS)), dtype=np.float32)
    damage = np.empty(count, dtype=np.int64)
    defects = np.empty((count, len(DEFECT_LABELS)), dtype=np.int64)
    with torch.inference_mode():
        for images, batch_damage, batch_defects, indices in loader:
            outputs = model(images.to(device, non_blocking=True))
            index_values = indices.numpy()
            binary_logits[index_values] = (
                outputs["binary"].detach().cpu().numpy()
            )
            defect_logits[index_values] = (
                outputs["defects"].detach().cpu().numpy()
            )
            damage[index_values] = batch_damage.numpy().astype(np.int64)
            defects[index_values] = batch_defects.numpy().astype(np.int64)
    return {
        "binary_logits": binary_logits,
        "defect_logits": defect_logits,
        "damage": damage,
        "defects": defects,
    }


def collect_quality_outputs(
    model: DeployableMultiTaskVsn,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    count = len(loader.dataset)
    binary_logits = np.empty(count, dtype=np.float32)
    defect_logits = np.empty((count, len(DEFECT_LABELS)), dtype=np.float32)
    quality_type_logits = np.empty((count, len(QUALITY_TYPES)), dtype=np.float32)
    quality_score_logits = np.empty(count, dtype=np.float32)
    damage = np.empty(count, dtype=np.int64)
    defects = np.empty((count, len(DEFECT_LABELS)), dtype=np.int64)
    quality_type = np.empty(count, dtype=np.int64)
    severity = np.empty(count, dtype=np.int64)
    quality_target = np.empty(count, dtype=np.float32)
    with torch.inference_mode():
        for (
            images,
            batch_damage,
            batch_defects,
            batch_quality_type,
            batch_severity,
            batch_quality_target,
            indices,
        ) in loader:
            outputs = model(images.to(device, non_blocking=True))
            if (
                outputs["quality_type"] is None
                or outputs["quality_score"] is None
            ):
                raise RuntimeError("Quality outputs requested from multilabel model")
            index_values = indices.numpy()
            binary_logits[index_values] = (
                outputs["binary"].detach().cpu().numpy()
            )
            defect_logits[index_values] = (
                outputs["defects"].detach().cpu().numpy()
            )
            quality_type_logits[index_values] = (
                outputs["quality_type"].detach().cpu().numpy()
            )
            quality_score_logits[index_values] = (
                outputs["quality_score"].detach().cpu().numpy()
            )
            damage[index_values] = batch_damage.numpy().astype(np.int64)
            defects[index_values] = batch_defects.numpy().astype(np.int64)
            quality_type[index_values] = batch_quality_type.numpy()
            severity[index_values] = batch_severity.numpy()
            quality_target[index_values] = batch_quality_target.numpy()
    return {
        "binary_logits": binary_logits,
        "defect_logits": defect_logits,
        "quality_type_logits": quality_type_logits,
        "quality_score_logits": quality_score_logits,
        "damage": damage,
        "defects": defects,
        "quality_type": quality_type,
        "severity": severity,
        "quality_target": quality_target,
    }


def quality_metrics(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    type_predictions = outputs["quality_type_logits"].argmax(axis=1)
    score_predictions = sigmoid(outputs["quality_score_logits"])
    target = outputs["quality_target"]
    severity = outputs["severity"]
    by_severity = {}
    for level in range(4):
        mask = severity == level
        by_severity[str(level)] = {
            "samples": int(mask.sum()),
            "target_mean": float(target[mask].mean()),
            "predicted_mean": float(score_predictions[mask].mean()),
            "mae": float(np.abs(score_predictions[mask] - target[mask]).mean()),
        }
    return {
        "type_accuracy": float(
            (type_predictions == outputs["quality_type"]).mean()
        ),
        "type_balanced_accuracy": float(
            balanced_accuracy_score(outputs["quality_type"], type_predictions)
        ),
        "type_macro_f1": float(
            f1_score(
                outputs["quality_type"],
                type_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "score_mae": float(np.abs(score_predictions - target).mean()),
        "score_rmse": float(
            np.sqrt(np.mean(np.square(score_predictions - target)))
        ),
        "score_target_correlation": float(
            np.corrcoef(score_predictions, target)[0, 1]
        ),
        "by_severity": by_severity,
    }


def confidence_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    quality_scores: np.ndarray | None,
) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    correct = (predictions == labels).astype(np.int64)
    base = entropy_confidence(probabilities)
    adjusted = base if quality_scores is None else base * quality_scores

    def error_detection(confidence: np.ndarray) -> dict[str, float]:
        high_conf_error = ((correct == 0) & (confidence >= 0.8)).mean()
        return {
            "mean": float(confidence.mean()),
            "mean_correct": float(confidence[correct == 1].mean()),
            "mean_incorrect": (
                float(confidence[correct == 0].mean())
                if np.any(correct == 0)
                else float("nan")
            ),
            "high_confidence_error_rate": float(high_conf_error),
            "correctness_auroc": safe_class_metric(
                roc_auc_score,
                correct,
                confidence,
            ),
        }

    return {
        "base_entropy_confidence": error_detection(base),
        "quality_adjusted_confidence": error_detection(adjusted),
    }


def load_teacher(path: Path, device: torch.device) -> CodebrimTeacher:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    teacher = CodebrimTeacher(multitask=True, pretrained=False)
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def train_epoch(
    model: DeployableMultiTaskVsn,
    teacher: CodebrimTeacher,
    loader: DataLoader,
    device: torch.device,
    binary_criterion: nn.Module,
    defect_criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {
        "total": 0.0,
        "binary_clean": 0.0,
        "defect": 0.0,
        "quality_type": 0.0,
        "quality_score": 0.0,
        "binary_robust": 0.0,
    }
    sample_count = 0
    for (
        clean_images,
        quality_images,
        teacher_images,
        damage,
        defects,
        quality_type,
        _,
        quality_target,
        _,
    ) in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        teacher_images = teacher_images.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        defects = defects.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)

        clean_outputs = model(clean_images)
        with torch.inference_mode():
            teacher_binary, teacher_defects = teacher(teacher_images)
        if teacher_defects is None:
            raise RuntimeError("Hierarchical teacher did not return defect logits")
        temperature = args.temperature
        binary_hard = binary_criterion(clean_outputs["binary"], damage)
        binary_kd = (
            F.binary_cross_entropy_with_logits(
                clean_outputs["binary"] / temperature,
                torch.sigmoid(teacher_binary / temperature),
            )
            * temperature
            * temperature
        )
        binary_clean = (
            args.hard_loss_weight * binary_hard
            + (1.0 - args.hard_loss_weight) * binary_kd
        )
        defect_hard = defect_criterion(clean_outputs["defects"], defects)
        defect_kd = (
            F.binary_cross_entropy_with_logits(
                clean_outputs["defects"] / temperature,
                torch.sigmoid(teacher_defects / temperature),
            )
            * temperature
            * temperature
        )
        defect_loss = (
            args.defect_hard_weight * defect_hard
            + (1.0 - args.defect_hard_weight) * defect_kd
        )
        loss = binary_clean + args.defect_weight * defect_loss

        quality_type_loss = torch.zeros((), device=device)
        quality_score_loss = torch.zeros((), device=device)
        binary_robust = torch.zeros((), device=device)
        if args.variant == "multitask_quality":
            quality_images = quality_images.to(device, non_blocking=True)
            quality_type = quality_type.to(device, non_blocking=True)
            quality_target = quality_target.to(device, non_blocking=True)
            quality_outputs = model(quality_images)
            if (
                quality_outputs["quality_type"] is None
                or quality_outputs["quality_score"] is None
            ):
                raise RuntimeError("Quality model is missing quality outputs")
            quality_type_loss = F.cross_entropy(
                quality_outputs["quality_type"],
                quality_type,
            )
            quality_score_loss = F.smooth_l1_loss(
                torch.sigmoid(quality_outputs["quality_score"]),
                quality_target,
            )
            robust_hard = binary_criterion(quality_outputs["binary"], damage)
            robust_kd = (
                F.binary_cross_entropy_with_logits(
                    quality_outputs["binary"] / temperature,
                    torch.sigmoid(teacher_binary / temperature),
                )
                * temperature
                * temperature
            )
            binary_robust = (
                args.hard_loss_weight * robust_hard
                + (1.0 - args.hard_loss_weight) * robust_kd
            )
            loss = (
                loss
                + args.quality_type_weight * quality_type_loss
                + args.quality_score_weight * quality_score_loss
                + args.robust_binary_weight * binary_robust
            )

        loss.backward()
        optimiser.step()
        batch_count = clean_images.size(0)
        sample_count += batch_count
        totals["total"] += float(loss.item()) * batch_count
        totals["binary_clean"] += float(binary_clean.item()) * batch_count
        totals["defect"] += float(defect_loss.item()) * batch_count
        totals["quality_type"] += float(quality_type_loss.item()) * batch_count
        totals["quality_score"] += float(quality_score_loss.item()) * batch_count
        totals["binary_robust"] += float(binary_robust.item()) * batch_count
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def validation_summary(
    model: DeployableMultiTaskVsn,
    clean_loader: DataLoader,
    quality_loader: DataLoader | None,
    device: torch.device,
    variant: str,
) -> dict[str, Any]:
    clean = collect_clean_outputs(model, clean_loader, device)
    binary = binary_metrics(clean["damage"], sigmoid(clean["binary_logits"]))
    defects = multilabel_metrics(
        clean["defects"],
        sigmoid(clean["defect_logits"]),
    )
    quality = None
    if quality_loader is not None:
        quality = quality_metrics(
            collect_quality_outputs(model, quality_loader, device)
        )
    binary_utility = float(binary["macro_f1"])
    defect_utility = float(defects["macro_f1"])
    if variant == "multilabel":
        selection_score = 0.625 * binary_utility + 0.375 * defect_utility
    else:
        assert quality is not None
        quality_utility = 0.5 * float(quality["type_macro_f1"]) + 0.5 * (
            1.0 - float(quality["score_mae"])
        )
        selection_score = (
            0.5 * binary_utility
            + 0.3 * defect_utility
            + 0.2 * quality_utility
        )
    return {
        "binary": binary,
        "defects": defects,
        "quality": quality,
        "selection_score": selection_score,
    }


def evaluate_robustness(
    model: DeployableMultiTaskVsn,
    rows: list[dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
    temperature: float,
) -> dict[str, Any]:
    condition_rows: list[dict[str, Any]] = []
    mixed_quality_parts: list[dict[str, np.ndarray]] = []
    for condition_index, (corruption, severity) in enumerate(CONDITIONS):
        loader = make_loader(
            QualityEvaluationDataset(
                rows,
                image_size=args.student_image_size,
                mode="condition",
                corruption=corruption,
                severity=severity,
                seed=args.corruption_seed,
            ),
            batch_size=args.batch_size,
            workers=args.eval_workers,
            shuffle=False,
            seed=args.seed + 500 + condition_index,
            pin_memory=device.type == "cuda",
        )
        if args.variant == "multitask_quality":
            outputs = collect_quality_outputs(model, loader, device)
            quality_scores = sigmoid(outputs["quality_score_logits"])
            type_predictions = outputs["quality_type_logits"].argmax(axis=1)
            quality_observation = {
                "type_accuracy": float(
                    (
                        type_predictions
                        == QUALITY_TYPE_TO_INDEX[corruption]
                    ).mean()
                ),
                "score_mean": float(quality_scores.mean()),
                "score_mae": float(
                    np.abs(
                        quality_scores - QUALITY_TARGETS[severity]
                    ).mean()
                ),
            }
            mixed_quality_parts.append(outputs)
        else:
            model.eval()
            count = len(loader.dataset)
            outputs = {
                "binary_logits": np.empty(count, dtype=np.float32),
                "defect_logits": np.empty(
                    (count, len(DEFECT_LABELS)),
                    dtype=np.float32,
                ),
                "damage": np.empty(count, dtype=np.int64),
                "defects": np.empty(
                    (count, len(DEFECT_LABELS)),
                    dtype=np.int64,
                ),
            }
            with torch.inference_mode():
                for (
                    images,
                    batch_damage,
                    batch_defects,
                    _,
                    _,
                    _,
                    indices,
                ) in loader:
                    batch_outputs = model(images.to(device, non_blocking=True))
                    idx = indices.numpy()
                    outputs["binary_logits"][idx] = (
                        batch_outputs["binary"].detach().cpu().numpy()
                    )
                    outputs["defect_logits"][idx] = (
                        batch_outputs["defects"].detach().cpu().numpy()
                    )
                    outputs["damage"][idx] = batch_damage.numpy().astype(
                        np.int64
                    )
                    outputs["defects"][idx] = batch_defects.numpy().astype(
                        np.int64
                    )
            quality_scores = None
            quality_observation = None
        probabilities = sigmoid(outputs["binary_logits"] / temperature)
        condition_rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "binary": binary_metrics(outputs["damage"], probabilities),
                "defects": multilabel_metrics(
                    outputs["defects"],
                    sigmoid(outputs["defect_logits"]),
                ),
                "quality": quality_observation,
                "confidence": confidence_metrics(
                    outputs["damage"],
                    probabilities,
                    quality_scores,
                ),
            }
        )

    overall_quality = None
    if mixed_quality_parts:
        keys = (
            "binary_logits",
            "defect_logits",
            "quality_type_logits",
            "quality_score_logits",
            "damage",
            "defects",
            "quality_type",
            "severity",
            "quality_target",
        )
        combined = {
            key: np.concatenate([part[key] for part in mixed_quality_parts])
            for key in keys
        }
        overall_quality = quality_metrics(combined)
    return {
        "conditions": condition_rows,
        "overall_quality": overall_quality,
    }


def aggregate_robustness(conditions: list[dict[str, Any]]) -> dict[str, float]:
    severe = [row for row in conditions if int(row["severity"]) == 3]
    return {
        "binary_clean_macro_f1": float(
            conditions[0]["binary"]["macro_f1"]
        ),
        "binary_severe_mean_macro_f1": float(
            np.mean([row["binary"]["macro_f1"] for row in severe])
        ),
        "binary_all_condition_mean_macro_f1": float(
            np.mean([row["binary"]["macro_f1"] for row in conditions])
        ),
        "defect_clean_macro_f1": float(
            conditions[0]["defects"]["macro_f1"]
        ),
        "defect_severe_mean_macro_f1": float(
            np.mean([row["defects"]["macro_f1"] for row in severe])
        ),
        "base_high_confidence_error_rate_mean": float(
            np.mean(
                [
                    row["confidence"]["base_entropy_confidence"][
                        "high_confidence_error_rate"
                    ]
                    for row in conditions
                ]
            )
        ),
        "adjusted_high_confidence_error_rate_mean": float(
            np.mean(
                [
                    row["confidence"]["quality_adjusted_confidence"][
                        "high_confidence_error_rate"
                    ]
                    for row in conditions
                ]
            )
        ),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    clean = result["test"]["clean"]
    aggregate = result["test"]["robustness"]["aggregate"]
    model = result["model"]
    lines = [
        f"# Deployable Multi-Task VSN: {result['variant']}",
        "",
        "## Protocol",
        "",
        f"- Seed: `{result['protocol']['seed']}`.",
        "- Official parent-disjoint CODEBRIM train/validation/test splits.",
        "- Test data were not used for checkpoint selection or temperature fitting.",
        "- Five defect labels are visual defect attributes, not severity levels.",
        "- Quality labels are generated by controlled synthetic corruptions.",
        "",
        "## Clean test",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Binary Macro-F1 | {clean['binary']['macro_f1']:.4f} |",
        f"| Binary AUROC | {clean['binary']['roc_auc']:.4f} |",
        f"| Binary ECE | {clean['binary']['ece_15']:.4f} |",
        f"| Defect Macro-F1 | {clean['defects']['macro_f1']:.4f} |",
        f"| Defect Macro-AP | {clean['defects']['macro_average_precision']:.4f} |",
        "",
        "## Controlled robustness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Binary severity-3 mean Macro-F1 | {aggregate['binary_severe_mean_macro_f1']:.4f} |",
        f"| Binary all-condition mean Macro-F1 | {aggregate['binary_all_condition_mean_macro_f1']:.4f} |",
        f"| Defect severity-3 mean Macro-F1 | {aggregate['defect_severe_mean_macro_f1']:.4f} |",
        f"| Base high-confidence error rate | {aggregate['base_high_confidence_error_rate_mean']:.4f} |",
        f"| Quality-adjusted high-confidence error rate | {aggregate['adjusted_high_confidence_error_rate_mean']:.4f} |",
        "",
        "## TinyML footprint",
        "",
        f"- Parameters: `{model['parameters']:,}`.",
        f"- Estimated FP32 weights: `{model['fp32_weight_kib']:.2f} KiB`.",
        f"- Estimated INT8 weights: `{model['int8_weight_kib']:.2f} KiB`.",
        "",
    ]
    quality = result["test"]["robustness"]["overall_quality"]
    if quality is not None:
        lines.extend(
            [
                "## Quality task",
                "",
                f"- Corruption-type Macro-F1: `{quality['type_macro_f1']:.4f}`.",
                f"- Quality-score MAE: `{quality['score_mae']:.4f}`.",
                f"- Quality target correlation: `{quality['score_target_correlation']:.4f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Claims boundary",
            "",
            result["claims_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    all_rows = read_rows(args.index)
    split_rows = {
        split: [row for row in all_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    limits = {
        "train": args.limit_train,
        "val": args.limit_val,
        "test": args.limit_test,
    }
    for split, limit in limits.items():
        if limit is not None:
            split_rows[split] = split_rows[split][:limit]

    quality_heads = args.variant == "multitask_quality"
    model = DeployableMultiTaskVsn(quality_heads=quality_heads).to(device)
    teacher = load_teacher(args.teacher_checkpoint, device)
    train_damage = np.asarray(
        [int(row["damage"]) for row in split_rows["train"]],
        dtype=np.int64,
    )
    binary_pos_weight = float(
        (train_damage == 0).sum() / max((train_damage == 1).sum(), 1)
    )
    train_defects = np.asarray(
        [
            [int(row[label]) for label in DEFECT_LABELS]
            for row in split_rows["train"]
        ],
        dtype=np.int64,
    )
    defect_pos_weights = (
        (train_defects == 0).sum(axis=0)
        / np.maximum((train_defects == 1).sum(axis=0), 1)
    ).astype(np.float32)
    binary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(binary_pos_weight, device=device)
    )
    defect_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(defect_pos_weights, device=device)
    )

    train_loader = make_loader(
        TrainingDataset(
            split_rows["train"],
            student_size=args.student_image_size,
            teacher_size=args.teacher_image_size,
            quality_heads=quality_heads,
        ),
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    val_clean_loader = make_loader(
        CleanEvaluationDataset(
            split_rows["val"],
            image_size=args.student_image_size,
        ),
        batch_size=args.batch_size,
        workers=args.eval_workers,
        shuffle=False,
        seed=args.seed + 1,
        pin_memory=device.type == "cuda",
    )
    val_quality_loader = (
        make_loader(
            QualityEvaluationDataset(
                split_rows["val"],
                image_size=args.student_image_size,
                mode="mixed",
                seed=args.corruption_seed,
            ),
            batch_size=args.batch_size,
            workers=args.eval_workers,
            shuffle=False,
            seed=args.seed + 2,
            pin_memory=device.type == "cuda",
        )
        if quality_heads
        else None
    )
    test_clean_loader = make_loader(
        CleanEvaluationDataset(
            split_rows["test"],
            image_size=args.student_image_size,
        ),
        batch_size=args.batch_size,
        workers=args.eval_workers,
        shuffle=False,
        seed=args.seed + 3,
        pin_memory=device.type == "cuda",
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(args.epochs, 1),
    )
    run_id = (
        f"codebrim_deployable_{args.variant}_{args.preset}_seed{args.seed}"
    )
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    start = time.perf_counter()

    print(
        f"Run={run_id} device={device} parameters="
        f"{sum(parameter.numel() for parameter in model.parameters())}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model,
            teacher,
            train_loader,
            device,
            binary_criterion,
            defect_criterion,
            optimiser,
            args,
        )
        validation = validation_summary(
            model,
            val_clean_loader,
            val_quality_loader,
            device,
            args.variant,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train": train_losses,
            "validation": validation,
            "learning_rate": optimiser.param_groups[0]["lr"],
        }
        history.append(row)
        quality_text = (
            f" quality_f1={validation['quality']['type_macro_f1']:.4f}"
            f" quality_mae={validation['quality']['score_mae']:.4f}"
            if validation["quality"] is not None
            else ""
        )
        print(
            f"{run_id} epoch={epoch:02d} loss={train_losses['total']:.4f} "
            f"binary_f1={validation['binary']['macro_f1']:.4f} "
            f"defect_f1={validation['defects']['macro_f1']:.4f}"
            f"{quality_text} selection={validation['selection_score']:.4f}",
            flush=True,
        )
        score = float(validation["selection_score"])
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": args.variant,
                    "preset": args.preset,
                    "quality_heads": quality_heads,
                    "epoch": epoch,
                    "validation": validation,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    val_outputs = collect_clean_outputs(model, val_clean_loader, device)
    temperature = fit_temperature(
        val_outputs["binary_logits"],
        val_outputs["damage"],
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if args.validation_only:
        pilot_result = {
            "run_id": run_id,
            "variant": args.variant,
            "preset": args.preset,
            "protocol": {
                "validation_only": True,
                "official_test_evaluated": False,
                "seed": args.seed,
                "selection_score": checkpoint["validation"]["selection_score"],
            },
            "loss_weights": {
                "defect_total": args.defect_weight,
                "quality_type": (
                    args.quality_type_weight if quality_heads else 0.0
                ),
                "quality_score": (
                    args.quality_score_weight if quality_heads else 0.0
                ),
                "robust_binary": (
                    args.robust_binary_weight if quality_heads else 0.0
                ),
            },
            "training": {
                "epochs_requested": args.epochs,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "elapsed_seconds": time.perf_counter() - start,
                "history": history,
            },
            "best_validation": checkpoint["validation"],
            "calibration": {"binary_temperature": temperature},
            "model": {
                "parameters": parameters,
                "base_binary_parameters": sum(
                    parameter.numel() for parameter in model.student.parameters()
                ),
                "additional_parameters": parameters
                - sum(
                    parameter.numel()
                    for parameter in model.student.parameters()
                ),
            },
        }
        (output_dir / "validation_only_result.json").write_text(
            json.dumps(pilot_result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Completed validation-only {run_id}: "
            f"selection={best_score:.4f}; official test not evaluated",
            flush=True,
        )
        return pilot_result
    test_outputs = collect_clean_outputs(model, test_clean_loader, device)
    clean_binary = binary_metrics(
        test_outputs["damage"],
        sigmoid(test_outputs["binary_logits"] / temperature),
    )
    clean_defects = multilabel_metrics(
        test_outputs["defects"],
        sigmoid(test_outputs["defect_logits"]),
    )
    robustness = evaluate_robustness(
        model,
        split_rows["test"],
        device,
        args,
        temperature,
    )
    robustness["aggregate"] = aggregate_robustness(
        robustness["conditions"]
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "variant": args.variant,
        "preset": args.preset,
        "protocol": {
            "index": str(args.index),
            "official_parent_disjoint_splits": True,
            "test_used_for_selection": False,
            "calibration_split": "official validation",
            "threshold": 0.5,
            "seed": args.seed,
            "quality_types": list(QUALITY_TYPES),
            "quality_targets": QUALITY_TARGETS,
            "synthetic_quality_scope": (
                "controlled corruption robustness, not real disaster validation"
            ),
        },
        "loss_weights": {
            "binary_hard": args.hard_loss_weight,
            "binary_kd": 1.0 - args.hard_loss_weight,
            "defect_total": args.defect_weight,
            "defect_hard_within_task": args.defect_hard_weight,
            "defect_kd_within_task": 1.0 - args.defect_hard_weight,
            "quality_type": args.quality_type_weight if quality_heads else 0.0,
            "quality_score": args.quality_score_weight if quality_heads else 0.0,
            "robust_binary": args.robust_binary_weight if quality_heads else 0.0,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "sample_counts": {
                split: len(rows) for split, rows in split_rows.items()
            },
            "elapsed_seconds": time.perf_counter() - start,
            "history": history,
        },
        "calibration": {"binary_temperature": temperature},
        "model": {
            "architecture": "VsnStudentDwCnn shared encoder with linear task heads",
            "parameters": parameters,
            "base_binary_parameters": sum(
                parameter.numel() for parameter in model.student.parameters()
            ),
            "additional_parameters": parameters
            - sum(parameter.numel() for parameter in model.student.parameters()),
            "fp32_weight_kib": parameters * 4 / 1024,
            "int8_weight_kib": parameters / 1024,
            "quality_heads": quality_heads,
        },
        "test": {
            "clean": {
                "binary": clean_binary,
                "defects": clean_defects,
            },
            "robustness": robustness,
        },
        "claims_boundary": (
            "The binary output is a visual-damage proxy rather than a collapse "
            "probability. CODEBRIM defect labels are non-ordinal attributes. "
            "Quality supervision uses controlled synthetic corruptions and does "
            "not establish field robustness or out-of-distribution detection."
        ),
    }
    torch.save(
        {
            "model": model.state_dict(),
            "variant": args.variant,
            "preset": args.preset,
            "quality_heads": quality_heads,
            "image_size": args.student_image_size,
            "defect_labels": list(DEFECT_LABELS),
            "quality_types": list(QUALITY_TYPES),
            "binary_temperature": temperature,
            "parameters": parameters,
        },
        output_dir / "deployed_multitask_state_dict.pt",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, output_dir / "REPORT.md")
    print(
        f"Completed {run_id}: clean_binary_f1={clean_binary['macro_f1']:.4f} "
        f"clean_defect_f1={clean_defects['macro_f1']:.4f} "
        f"severe_binary_f1="
        f"{robustness['aggregate']['binary_severe_mean_macro_f1']:.4f}",
        flush=True,
    )
    return result


def apply_preset(args: argparse.Namespace) -> None:
    presets = {
        "light_aux": {
            "defect_weight": 0.15,
            "quality_type_weight": 0.10,
            "quality_score_weight": 0.10,
            "robust_binary_weight": 0.20,
        },
        "balanced": {
            "defect_weight": 0.25,
            "quality_type_weight": 0.15,
            "quality_score_weight": 0.15,
            "robust_binary_weight": 0.25,
        },
    }
    values = presets[args.preset]
    for name, value in values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a deployable multi-task CODEBRIM TinyML VSN."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--preset",
        choices=("light_aux", "balanced"),
        default="light_aux",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "vsn_codebrim"
        / "codebrim_multitask_index.csv",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "vsn_codebrim_teachers"
        / "codebrim_hierarchical_multitask_seed2026"
        / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "vsn_codebrim_deployable_multitask",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-image-size", type=int, default=128)
    parser.add_argument("--teacher-image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--hard-loss-weight", type=float, default=0.5)
    parser.add_argument("--defect-hard-weight", type=float, default=0.5)
    parser.add_argument("--defect-weight", type=float)
    parser.add_argument("--quality-type-weight", type=float)
    parser.add_argument("--quality-score-weight", type=float)
    parser.add_argument("--robust-binary-weight", type=float)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--corruption-seed", type=int, default=91_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    apply_preset(args)
    args.index = args.index.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    train_run(args)


if __name__ == "__main__":
    main()

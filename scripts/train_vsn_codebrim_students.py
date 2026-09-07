"""Train compact CODEBRIM students for binary and multi-label tasks."""

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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vsn_sdnet_zero_shot import (  # noqa: E402
    binary_metrics,
    group_bootstrap_ci,
    sigmoid,
)
from train_vsn_codebrim_teachers import (  # noqa: E402
    CodebrimTeacher,
    DEFECT_LABELS,
)
from train_vsn_student_baseline import VsnStudentDwCnn  # noqa: E402


RUN_KINDS = (
    "scratch",
    "binary_kd",
    "binary_kd_robust",
    "binary_kd_consistency",
    "hierarchical_multitask_kd",
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomModerateCorruption:
    """Apply at most one label-preserving visual corruption per training view."""

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Corruption probability must be in [0, 1]")
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        corruption = random.choice(
            ("low_light", "blur", "gaussian_noise", "occlusion", "jpeg")
        )
        if corruption == "low_light":
            return ImageEnhance.Brightness(image).enhance(random.uniform(0.30, 0.80))
        if corruption == "blur":
            return image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5))
            )
        if corruption == "gaussian_noise":
            array = np.asarray(image).astype(np.float32) / 255.0
            rng = np.random.default_rng(random.getrandbits(64))
            sigma = random.uniform(0.02, 0.10)
            array = np.clip(array + rng.normal(0.0, sigma, size=array.shape), 0.0, 1.0)
            return Image.fromarray((array * 255).astype(np.uint8))
        if corruption == "occlusion":
            array = np.asarray(image.copy()).copy()
            height, width = array.shape[:2]
            side = max(1, int(min(height, width) * random.uniform(0.10, 0.30)))
            x0 = random.randint(0, max(width - side, 0))
            y0 = random.randint(0, max(height - side, 0))
            array[y0 : y0 + side, x0 : x0 + side] = 0
            return Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(20, 70))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class DeterministicMixedCorruption:
    """Assign one fixed severity-2 corruption to each validation sample."""

    def __call__(self, image: Image.Image, index: int) -> Image.Image:
        corruption = index % 5
        if corruption == 0:
            return ImageEnhance.Brightness(image).enhance(0.45)
        if corruption == 1:
            return image.filter(ImageFilter.GaussianBlur(radius=2.0))
        if corruption == 2:
            array = np.asarray(image).astype(np.float32) / 255.0
            rng = np.random.default_rng(10_000 + index)
            array = np.clip(
                array + rng.normal(0.0, 0.07, size=array.shape),
                0.0,
                1.0,
            )
            return Image.fromarray((array * 255).astype(np.uint8))
        if corruption == 3:
            array = np.asarray(image.copy()).copy()
            height, width = array.shape[:2]
            side = max(1, int(min(height, width) * 0.22))
            rng = random.Random(20_000 + index)
            x0 = rng.randint(0, max(width - side, 0))
            y0 = rng.randint(0, max(height - side, 0))
            array[y0 : y0 + side, x0 : x0 + side] = 0
            return Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=35)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class CodebrimStudentDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        student_size: int,
        teacher_size: int,
        train: bool,
        need_teacher: bool,
        robust_student_view: bool,
        robust_corruption_probability: float,
        deterministic_mixed_corruption: bool,
    ) -> None:
        self.rows = rows
        self.need_teacher = need_teacher
        self.robust_student_view = robust_student_view
        self.deterministic_mixed_corruption = deterministic_mixed_corruption
        self.robust_augmentation = RandomModerateCorruption(
            robust_corruption_probability
        )
        self.mixed_validation_corruption = DeterministicMixedCorruption()
        self.shared_augmentation = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            )
            if train
            else None
        )
        self.student_post = transforms.Compose(
            [
                transforms.Resize((student_size, student_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.teacher_post = transforms.Compose(
            [
                transforms.Resize((teacher_size, teacher_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as image:
            image = image.convert("RGB")
            if self.shared_augmentation is not None:
                image = self.shared_augmentation(image)
            if self.deterministic_mixed_corruption:
                student_source = self.mixed_validation_corruption(image, index)
            elif self.robust_student_view:
                student_source = self.robust_augmentation(image)
            else:
                student_source = image
            student_image = self.student_post(student_source)
            clean_student_image = self.student_post(image)
            teacher_image = (
                self.teacher_post(image) if self.need_teacher else torch.empty(0)
            )
        damage = torch.tensor(float(row["damage"]), dtype=torch.float32)
        defects = torch.tensor(
            [float(row[label]) for label in DEFECT_LABELS],
            dtype=torch.float32,
        )
        return (
            student_image,
            teacher_image,
            damage,
            defects,
            index,
            clean_student_image,
        )


class DistillableVsnStudent(nn.Module):
    def __init__(self, auxiliary_defect_head: bool) -> None:
        super().__init__()
        self.student = VsnStudentDwCnn(width=1.0, dropout=0.1)
        self.auxiliary_defect_head = (
            nn.Linear(96, len(DEFECT_LABELS)) if auxiliary_defect_head else None
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        feature_map = self.student.features(images)
        embedding = self.student.head[1](self.student.head[0](feature_map))
        binary_logits = self.student.head[3](self.student.head[2](embedding)).flatten()
        defect_logits = (
            self.auxiliary_defect_head(embedding)
            if self.auxiliary_defect_head is not None
            else None
        )
        return binary_logits, defect_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def make_loader(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    train: bool,
    need_teacher: bool,
    device: torch.device,
    robust_student_view: bool = False,
    deterministic_mixed_corruption: bool = False,
) -> DataLoader:
    return DataLoader(
        CodebrimStudentDataset(
            rows,
            student_size=args.student_image_size,
            teacher_size=args.teacher_image_size,
            train=train,
            need_teacher=need_teacher,
            robust_student_view=robust_student_view,
            robust_corruption_probability=args.robust_corruption_probability,
            deterministic_mixed_corruption=deterministic_mixed_corruption,
        ),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def load_teacher(
    run_kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> CodebrimTeacher | None:
    if run_kind == "scratch":
        return None
    multitask = run_kind == "hierarchical_multitask_kd"
    checkpoint_path = (
        args.multitask_teacher_checkpoint
        if multitask
        else args.binary_teacher_checkpoint
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = CodebrimTeacher(multitask=multitask, pretrained=False)
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def train_epoch(
    model: DistillableVsnStudent,
    teacher: CodebrimTeacher | None,
    loader: DataLoader,
    device: torch.device,
    hard_criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    args: argparse.Namespace,
    run_kind: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_hard = 0.0
    total_binary_kd = 0.0
    total_defect_kd = 0.0
    consistency_training = run_kind == "binary_kd_consistency"
    for (
        student_images,
        teacher_images,
        damage,
        _,
        _,
        clean_student_images,
    ) in loader:
        student_images = student_images.to(device, non_blocking=True)
        if consistency_training:
            clean_student_images = clean_student_images.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        student_binary_logits, student_defect_logits = model(student_images)
        hard_loss = hard_criterion(student_binary_logits, damage)
        binary_kd_loss = torch.zeros((), device=device)
        defect_kd_loss = torch.zeros((), device=device)
        if teacher is not None:
            teacher_images = teacher_images.to(device, non_blocking=True)
            with torch.inference_mode():
                teacher_binary_logits, teacher_defect_logits = teacher(teacher_images)
            temperature = args.temperature
            binary_soft_targets = torch.sigmoid(teacher_binary_logits / temperature)
            binary_kd_loss = (
                F.binary_cross_entropy_with_logits(
                    student_binary_logits / temperature,
                    binary_soft_targets,
                )
                * temperature
                * temperature
            )
            loss = args.hard_loss_weight * hard_loss + (
                1.0 - args.hard_loss_weight
            ) * binary_kd_loss
            if consistency_training:
                clean_binary_logits, _ = model(clean_student_images)
                clean_hard_loss = hard_criterion(clean_binary_logits, damage)
                clean_binary_kd_loss = (
                    F.binary_cross_entropy_with_logits(
                        clean_binary_logits / temperature,
                        binary_soft_targets,
                    )
                    * temperature
                    * temperature
                )
                clean_loss = (
                    args.hard_loss_weight * clean_hard_loss
                    + (1.0 - args.hard_loss_weight) * clean_binary_kd_loss
                )
                robust_weight = args.consistency_robust_weight
                loss = (1.0 - robust_weight) * clean_loss + robust_weight * loss
                hard_loss = (
                    (1.0 - robust_weight) * clean_hard_loss
                    + robust_weight * hard_loss
                )
                binary_kd_loss = (
                    (1.0 - robust_weight) * clean_binary_kd_loss
                    + robust_weight * binary_kd_loss
                )
            if student_defect_logits is not None:
                if teacher_defect_logits is None:
                    raise RuntimeError("Multi-task KD requires Teacher defect logits")
                defect_soft_targets = torch.sigmoid(teacher_defect_logits / temperature)
                defect_kd_loss = (
                    F.binary_cross_entropy_with_logits(
                        student_defect_logits / temperature,
                        defect_soft_targets,
                    )
                    * temperature
                    * temperature
                )
                loss = loss + args.defect_kd_weight * defect_kd_loss
        else:
            loss = hard_loss
        loss.backward()
        optimiser.step()
        batch_size = student_images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_hard += float(hard_loss.item()) * batch_size
        total_binary_kd += float(binary_kd_loss.item()) * batch_size
        total_defect_kd += float(defect_kd_loss.item()) * batch_size
    sample_count = max(len(loader.dataset), 1)
    return {
        "total": total_loss / sample_count,
        "hard_binary": total_hard / sample_count,
        "binary_kd": total_binary_kd / sample_count,
        "defect_kd": total_defect_kd / sample_count,
    }


def collect_outputs(
    model: DistillableVsnStudent,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    logits = np.empty(len(loader.dataset), dtype=np.float32)
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.inference_mode():
        for student_images, _, damage, _, indices, _ in loader:
            student_images = student_images.to(device, non_blocking=True)
            damage = damage.to(device, non_blocking=True)
            batch_logits, _ = model(student_images)
            loss = criterion(batch_logits, damage)
            index_values = indices.numpy()
            logits[index_values] = batch_logits.detach().cpu().numpy()
            labels[index_values] = damage.detach().cpu().numpy().astype(np.int64)
            total_loss += float(loss.item()) * len(indices)
    return logits, labels, total_loss / max(len(loader.dataset), 1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    logits: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> None:
    fields = [
        "path",
        "filename",
        "split",
        "parent_group",
        "damage",
        "logit",
        "probability_raw",
        "probability_val_calibrated",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:5]},
                    "logit": f"{float(logits[index]):.9g}",
                    "probability_raw": f"{float(raw_probabilities[index]):.9g}",
                    "probability_val_calibrated": (
                        f"{float(calibrated_probabilities[index]):.9g}"
                    ),
                }
            )


def train_run(args: argparse.Namespace, run_kind: str) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    all_rows = read_rows(args.index)
    split_rows = {
        split: [row for row in all_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    need_teacher = run_kind != "scratch"
    robust_training = run_kind in (
        "binary_kd_robust",
        "binary_kd_consistency",
    )
    loaders = {
        "train": make_loader(
            split_rows["train"],
            args,
            train=True,
            need_teacher=need_teacher,
            device=device,
            robust_student_view=robust_training,
        ),
        "val": make_loader(
            split_rows["val"],
            args,
            train=False,
            need_teacher=False,
            device=device,
        ),
        "test": make_loader(
            split_rows["test"],
            args,
            train=False,
            need_teacher=False,
            device=device,
        ),
    }
    if run_kind == "binary_kd_consistency":
        loaders["robust_val"] = make_loader(
            split_rows["val"],
            args,
            train=False,
            need_teacher=False,
            device=device,
            deterministic_mixed_corruption=True,
        )
    auxiliary_head = run_kind == "hierarchical_multitask_kd"
    model = DistillableVsnStudent(auxiliary_defect_head=auxiliary_head).to(device)
    teacher = load_teacher(run_kind, args, device)
    train_labels = np.asarray([int(row["damage"]) for row in split_rows["train"]])
    positive_weight = float((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))
    hard_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
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

    run_id = f"codebrim_student_{run_kind}_seed{args.seed}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_val_macro_f1 = -math.inf
    best_robust_val_macro_f1: float | None = None
    best_selection_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    start_time = time.perf_counter()

    print(f"Run={run_id} device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model,
            teacher,
            loaders["train"],
            device,
            hard_criterion,
            optimiser,
            args,
            run_kind,
        )
        val_logits, val_labels, val_loss = collect_outputs(model, loaders["val"], device)
        val_metrics = binary_metrics(val_labels, sigmoid(val_logits))
        robust_val_metrics = None
        robust_val_loss = None
        selection_score = float(val_metrics["macro_f1"])
        if run_kind == "binary_kd_consistency":
            robust_val_logits, robust_val_labels, robust_val_loss = collect_outputs(
                model,
                loaders["robust_val"],
                device,
            )
            robust_val_metrics = binary_metrics(
                robust_val_labels,
                sigmoid(robust_val_logits),
            )
            clean_score = float(val_metrics["macro_f1"])
            robust_score = float(robust_val_metrics["macro_f1"])
            selection_score = (
                2.0 * clean_score * robust_score / max(clean_score + robust_score, 1e-12)
            )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses,
                "val_loss": val_loss,
                "val": val_metrics,
                "robust_val_loss": robust_val_loss,
                "robust_val": robust_val_metrics,
                "selection_score": selection_score,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        print(
            f"{run_id} epoch={epoch:02d} train_loss={train_losses['total']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
            + (
                f" robust_val_macro_f1={robust_val_metrics['macro_f1']:.4f} "
                f"selection_hmean={selection_score:.4f}"
                if robust_val_metrics is not None
                else ""
            ),
            flush=True,
        )
        if selection_score > best_selection_score + args.min_delta:
            best_selection_score = selection_score
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_robust_val_macro_f1 = (
                float(robust_val_metrics["macro_f1"])
                if robust_val_metrics is not None
                else None
            )
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "wrapper_model": model.state_dict(),
                    "student_model": model.student.state_dict(),
                    "run_kind": run_kind,
                    "epoch": epoch,
                    "student_image_size": args.student_image_size,
                    "auxiliary_defect_head": auxiliary_head,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "robust_val_metrics": robust_val_metrics,
                    "selection_score": selection_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["wrapper_model"])
    val_logits, val_labels, val_loss = collect_outputs(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels, test_loss = collect_outputs(model, loaders["test"], device)
    raw_probabilities = sigmoid(test_logits)
    calibrated_probabilities = sigmoid(test_logits / temperature)
    raw_metrics = binary_metrics(test_labels, raw_probabilities)
    calibrated_metrics = binary_metrics(test_labels, calibrated_probabilities)
    deployed_parameters = sum(parameter.numel() for parameter in model.student.parameters())
    training_parameters = sum(parameter.numel() for parameter in model.parameters())
    teacher_checkpoint = None
    if run_kind in (
        "binary_kd",
        "binary_kd_robust",
        "binary_kd_consistency",
    ):
        teacher_checkpoint = str(args.binary_teacher_checkpoint)
    elif run_kind == "hierarchical_multitask_kd":
        teacher_checkpoint = str(args.multitask_teacher_checkpoint)

    result: dict[str, object] = {
        "run_id": run_id,
        "run_kind": run_kind,
        "architecture": "VsnStudentDwCnn",
        "protocol": {
            "index": str(args.index),
            "official_parent_disjoint_splits": True,
            "test_used_during_training": False,
            "selection_metric": (
                "harmonic mean of clean and deterministic mixed-corruption "
                "validation macro-F1"
                if run_kind == "binary_kd_consistency"
                else "validation binary macro-F1"
            ),
            "decision_threshold": 0.5,
            "calibration_data": "official validation split",
            "threshold_tuning": None,
            "seed": args.seed,
        },
        "distillation": {
            "teacher_checkpoint": teacher_checkpoint,
            "temperature": args.temperature if need_teacher else None,
            "hard_loss_weight": args.hard_loss_weight if need_teacher else 1.0,
            "binary_kd_weight": (
                1.0 - args.hard_loss_weight if need_teacher else 0.0
            ),
            "defect_kd_weight": (
                args.defect_kd_weight
                if run_kind == "hierarchical_multitask_kd"
                else 0.0
            ),
            "training_only_auxiliary_head": auxiliary_head,
            "auxiliary_head_removed_for_deployment": auxiliary_head,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_robust_val_macro_f1": best_robust_val_macro_f1,
            "best_selection_score": best_selection_score,
            "batch_size": args.batch_size,
            "student_image_size": args.student_image_size,
            "teacher_image_size": args.teacher_image_size if need_teacher else None,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "positive_weight": positive_weight,
            "sample_counts": {split: len(values) for split, values in split_rows.items()},
            "elapsed_seconds": time.perf_counter() - start_time,
            "robustness_training": {
                "enabled": robust_training,
                "teacher_view": (
                    "weak shared augmentation"
                    if robust_training
                    else "same shared augmentation as Student"
                ),
                "student_view": (
                    "weak augmentation plus one random moderate corruption"
                    if robust_training
                    else "weak shared augmentation"
                ),
                "corruption_probability": (
                    args.robust_corruption_probability
                    if robust_training
                    else 0.0
                ),
                "corruptions": (
                    ["low_light", "blur", "gaussian_noise", "occlusion", "jpeg"]
                    if robust_training
                    else []
                ),
                "ranges": (
                    {
                        "low_light_factor": [0.30, 0.80],
                        "blur_radius": [0.5, 2.5],
                        "gaussian_noise_sigma": [0.02, 0.10],
                        "occlusion_side_fraction": [0.10, 0.30],
                        "jpeg_quality": [20, 70],
                    }
                    if robust_training
                    else {}
                ),
                "dual_view_consistency": run_kind == "binary_kd_consistency",
                "consistency_robust_weight": (
                    args.consistency_robust_weight
                    if run_kind == "binary_kd_consistency"
                    else None
                ),
                "selection_validation": (
                    "deterministic one-of-five severity-2 mixed corruption"
                    if run_kind == "binary_kd_consistency"
                    else "clean validation only"
                ),
            },
        },
        "model": {
            "deployed_parameters": deployed_parameters,
            "training_parameters": training_parameters,
            "training_only_parameters": training_parameters - deployed_parameters,
            "deployed_fp32_weight_mb_estimate": (
                deployed_parameters * 4 / (1024 * 1024)
            ),
            "deployed_int8_weight_mb_estimate": (
                deployed_parameters / (1024 * 1024)
            ),
        },
        "calibration": {
            "temperature": temperature,
            "validation_loss": val_loss,
        },
        "test": {
            "loss": test_loss,
            "raw": raw_metrics,
            "val_calibrated": calibrated_metrics,
            "group_bootstrap_95ci": {
                "raw": group_bootstrap_ci(
                    split_rows["test"],
                    test_labels,
                    raw_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
                "val_calibrated": group_bootstrap_ci(
                    split_rows["test"],
                    test_labels,
                    calibrated_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
            },
        },
        "claims_boundary": (
            "The Student predicts a CODEBRIM visual any-defect evidence proxy, not structural "
            "severity or collapse probability."
        ),
    }
    (output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    torch.save(
        {
            "model": checkpoint["student_model"],
            "run_kind": run_kind,
            "image_size": args.student_image_size,
            "deployed_parameters": deployed_parameters,
            "temperature": temperature,
        },
        output_dir / "deployed_student_state_dict.pt",
    )
    write_predictions(
        output_dir / "test_predictions.csv",
        split_rows["test"],
        test_logits,
        raw_probabilities,
        calibrated_probabilities,
    )
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def write_report(output_dir: Path, results: list[dict[str, object]]) -> None:
    lines = [
        "# CODEBRIM 15k-Parameter VSN Student Distillation",
        "",
        "## Controlled Variables",
        "",
        "- Identical deployed VsnStudentDwCnn architecture and 128 x 128 input.",
        "- Identical official train/validation/test data and seed.",
        "- Auxiliary defect head exists only during hierarchical KD training and is removed from deployment.",
        "- Test decision threshold remains fixed at 0.5.",
        "",
        "## Results",
        "",
        "| Student | Deployed params | Training-only params | Balanced acc. | Macro-F1 | Positive F1 | AUROC | AUPRC | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["test"]["val_calibrated"]
        lines.append(
            f"| {result['run_kind']} | {result['model']['deployed_parameters']:,} | "
            f"{result['model']['training_only_parameters']:,} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['f1_positive']:.4f} | {metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} | {metrics['ece_15']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Knowledge distillation is useful only if a distilled Student improves over the from-scratch "
            "Student under the same deployed architecture. Hierarchical KD is useful beyond binary-logit "
            "KD only if its paired parent-group difference is stable or it improves robustness in the next stage.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CODEBRIM TinyML Student KD ablations.")
    parser.add_argument(
        "--run-kinds",
        nargs="+",
        choices=RUN_KINDS,
        default=["scratch", "binary_kd", "hierarchical_multitask_kd"],
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
        "--binary-teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "vsn_codebrim_teachers"
        / "codebrim_binary_only_seed2026"
        / "best.pt",
    )
    parser.add_argument(
        "--multitask-teacher-checkpoint",
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
        default=PROJECT_ROOT / "outputs" / "vsn_codebrim_students",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-image-size", type=int, default=128)
    parser.add_argument("--teacher-image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--hard-loss-weight", type=float, default=0.5)
    parser.add_argument("--defect-kd-weight", type=float, default=0.25)
    parser.add_argument("--robust-corruption-probability", type=float, default=0.75)
    parser.add_argument("--consistency-robust-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_run(args, run_kind) for run_kind in args.run_kinds]
    write_report(args.output_dir, results)
    print(f"Wrote CODEBRIM Student comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
FASE 5: ENTRENAMIENTO PRIORIDAD - MODELO SINGLE-HEAD
====================================================
Genera una etiqueta de prioridad con una regla basada en Parada/ABC/duracion
y entrena un clasificador para reproducirla.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import imageio
    VIS_AVAILABLE = True
except Exception:
    VIS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=False):
        return iterable

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ADVERTENCIA] PyTorch o Transformers no estan instalados.")


CONFIG = {
    "model_name": "dccuchile/bert-base-spanish-wwm-cased",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 6,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "use_class_weights": True,
    "use_balanced_sampler": True,
    "early_stopping_patience": 2,
    "enable_gif": True,
    "gif_fps": 2,
    "frame_every_steps": 50,
    "smooth_window": 50,
    "log_every_steps": 25,
    "seed": 42,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "data_path": "../data/IW69_limpio.csv",
    "output_model": "prioridad_model.bin",
    "output_report": "training_report_prioridad.md",
    "label_encoder": "label_encoder_prioridad.json",
    "dur_threshold_high": 5.0,
    "dur_threshold_mid": 2.0,
}


def set_seed(seed):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class PriorityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SingleHeadBERT(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


class TrainingVisualizer:
    def __init__(self, output_dir, prefix="prioridad", frame_every_steps=50, smooth_window=50):
        self.output_dir = output_dir
        self.prefix = prefix
        self.metrics_history = []
        self.batch_history = []
        self.frames = []
        self.step_counter = 0
        self.frame_every_steps = frame_every_steps
        self.smooth_window = smooth_window
        self.frames_dir = os.path.join(output_dir, f"frames_{prefix}")
        os.makedirs(self.frames_dir, exist_ok=True)

        plt.style.use("dark_background")
        sns.set_palette("husl")

    def update_batch(self, loss_value):
        self.step_counter += 1
        self.batch_history.append({"step": self.step_counter, "loss": loss_value})
        if self.step_counter % self.frame_every_steps == 0:
            self._plot_frame()

    def update_epoch(self, epoch, train_loss, val_metrics):
        self.metrics_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "f1_macro": val_metrics["f1_macro"],
            "f1_weighted": val_metrics["f1_weighted"],
        })
        self._plot_frame(end_of_epoch=True)

    def _plot_frame(self, end_of_epoch=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if self.batch_history:
            df_batch = pd.DataFrame(self.batch_history)
            window = min(self.smooth_window, len(df_batch))
            smooth_loss = df_batch["loss"].rolling(window=window, min_periods=1).mean()
            ax1.plot(df_batch["step"], df_batch["loss"], alpha=0.3, color="#00f2ff", linewidth=1, label="Raw Loss")
            ax1.plot(df_batch["step"], smooth_loss, color="white", linewidth=2, label=f"Avg Loss ({window})")
            ax1.set_title(f"Loss por lote (step {self.step_counter})")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.2)
        else:
            ax1.text(0.5, 0.5, "Esperando lotes...", ha="center", va="center", color="gray")

        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            ax2.plot(df["epoch"], df["f1_macro"], marker="s", color="#ff00ff", linewidth=2, label="F1 Macro")
            ax2.plot(df["epoch"], df["f1_weighted"], marker="^", color="#00ff00", linewidth=2, label="F1 Weighted")
            ax2.axhline(y=0.95, color="cyan", linestyle="--", alpha=0.6, label="Meta 0.95")
            title = "F1 (Validacion)"
            if end_of_epoch:
                title += f" | Epoca {df['epoch'].iloc[-1]}"
            ax2.set_title(title)
            ax2.set_xlabel("Epoca")
            ax2.set_ylabel("F1")
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.2)
        else:
            ax2.text(0.5, 0.5, "Esperando epocas...", ha="center", va="center", color="gray")

        fname = os.path.join(self.frames_dir, f"frame_{self.step_counter:06d}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=110)
        plt.close()
        self.frames.append(fname)

    def create_gif(self, fps=2):
        if not self.frames:
            return
        gif_path = os.path.join(self.output_dir, f"training_{self.prefix}.gif")
        with imageio.get_writer(gif_path, mode="I", duration=1.0 / max(fps, 1)) as writer:
            for fn in self.frames:
                writer.append_data(imageio.imread(fn))
        return gif_path


def normalize_abc(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s


def build_priority_label(row):
    abc = normalize_abc(row.get("Indicador ABC"))
    parada = str(row.get("Parada", "")).strip().upper()
    dur = row.get("duracion_horas")

    if abc in ["1", "A"]:
        prio = 1
    elif abc in ["2", "B"]:
        prio = 2
    elif abc in ["3", "C"]:
        prio = 3
    else:
        prio = 4

    if parada == "X":
        prio = max(1, prio - 1)
        if pd.notna(dur) and float(dur) >= CONFIG["dur_threshold_high"]:
            prio = max(1, prio - 1)

    return f"{prio}-" + ["Critica", "Alta", "Media", "Baja"][prio - 1]


def build_text(row):
    descripcion = str(row.get("descripcion", ""))
    problema = str(row.get("TextoCódProblem", ""))
    parada = str(row.get("Parada", ""))
    abc = str(row.get("Indicador ABC", ""))
    dur = row.get("duracion_horas")
    dur_txt = "" if pd.isna(dur) else f"{dur:.1f}"
    return f"{descripcion}. Problema: {problema}. Parada: {parada}. ABC: {abc}. DuracionHoras: {dur_txt}."


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, CONFIG["data_path"])
    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)

    df["prioridad_label"] = df.apply(build_priority_label, axis=1)
    df["texto_completo"] = df.apply(build_text, axis=1)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["prioridad_label"].astype(str))

    encoders_path = os.path.join(script_dir, CONFIG["label_encoder"])
    with open(encoders_path, "w", encoding="utf-8") as f:
        json.dump({"prioridad": list(le.classes_)}, f, ensure_ascii=False, indent=2)

    return df, le


def create_dataloaders(df, tokenizer):
    texts = df["texto_completo"].values
    labels = df["label_encoded"].values
    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - CONFIG["train_split"]),
        random_state=CONFIG["seed"],
        stratify=labels,
    )
    val_ratio = CONFIG["val_split"] / (CONFIG["val_split"] + CONFIG["test_split"])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=CONFIG["seed"],
        stratify=labels[temp_idx],
    )

    train_ds = PriorityDataset(texts[train_idx], labels[train_idx], tokenizer, CONFIG["max_length"])
    val_ds = PriorityDataset(texts[val_idx], labels[val_idx], tokenizer, CONFIG["max_length"])
    test_ds = PriorityDataset(texts[test_idx], labels[test_idx], tokenizer, CONFIG["max_length"])

    if CONFIG.get("use_balanced_sampler", False):
        class_counts = np.bincount(labels[train_idx])
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels[train_idx]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None, epoch=None, visualizer=None):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"], weight=class_weights)

    progress = tqdm(loader, desc=f"Entrenando (epoca {epoch})" if epoch else "Entrenando", leave=False)
    for i, batch in enumerate(progress, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        if visualizer:
            visualizer.update_batch(loss.item())
        if CONFIG.get("log_every_steps", 0) and (i % CONFIG["log_every_steps"] == 0):
            print(f"  step {i}/{len(loader)} - loss {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true.extend(batch["label"].cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(true, preds),
        "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(true, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
    }
    return metrics, preds, true


def generate_report(metrics_history, final_metrics, le, training_time):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, CONFIG["output_report"])

    report = f"""# REPORTE ENTRENAMIENTO PRIORIDAD
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tiempo total: {training_time:.2f} minutos

## Configuracion
- Modelo: {CONFIG['model_name']}
- Max length: {CONFIG['max_length']}
- Epochs: {CONFIG['epochs']}
- Batch size: {CONFIG['batch_size']}
- Learning rate: {CONFIG['learning_rate']}

## Metricas finales (Test)
- Accuracy: {final_metrics['accuracy']:.4f}
- F1 Macro: {final_metrics['f1_macro']:.4f}
- F1 Weighted: {final_metrics['f1_weighted']:.4f}
- Precision Macro: {final_metrics['precision_macro']:.4f}
- Recall Macro: {final_metrics['recall_macro']:.4f}

## Historial por epoca
| Epoca | Loss Train | F1 Macro (Val) | F1 Weighted (Val) |
|------:|-----------:|---------------:|------------------:|
"""
    for row in metrics_history:
        report += f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_metrics']['f1_macro']:.4f} | {row['val_metrics']['f1_weighted']:.4f} |\n"

    report += "\n## Clases\n"
    for i, c in enumerate(le.classes_):
        report += f"{i+1}. {c}\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def train():
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch no esta disponible.")
        return

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    df, le = load_data()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    train_loader, val_loader, test_loader = create_dataloaders(df, tokenizer)

    class_weights = None
    if CONFIG["use_class_weights"]:
        counts = np.bincount(df["label_encoded"].values)
        total = counts.sum()
        weights = total / (len(counts) * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = SingleHeadBERT(CONFIG["model_name"], num_classes=len(le.classes_)).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    metrics_history = []
    best_f1 = 0.0
    best_state = None
    start = datetime.now()

    visualizer = None
    if CONFIG.get("enable_gif", False) and VIS_AVAILABLE:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visualizer = TrainingVisualizer(
            script_dir,
            prefix="prioridad",
            frame_every_steps=CONFIG.get("frame_every_steps", 50),
            smooth_window=CONFIG.get("smooth_window", 50),
        )

    no_improve = 0
    patience = CONFIG.get("early_stopping_patience", 0)

    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            class_weights,
            epoch=epoch + 1,
            visualizer=visualizer,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        })

        print(f"Epoca {epoch+1}/{CONFIG['epochs']} - Loss: {train_loss:.4f} | F1 Macro: {val_metrics['f1_macro']:.4f} | F1 W: {val_metrics['f1_weighted']:.4f}")
        if visualizer:
            visualizer.update_epoch(epoch + 1, train_loss, val_metrics)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f"Early stopping activado (patience={patience}).")
            break

    if best_state:
        model.load_state_dict(best_state)

    final_metrics, preds, true = evaluate(model, test_loader, device)
    print("Final Test Metrics:", final_metrics)

    report_path = generate_report(metrics_history, final_metrics, le, (datetime.now() - start).total_seconds() / 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, CONFIG["output_model"])
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "label_encoder": list(le.classes_),
        "final_metrics": final_metrics,
    }, model_path)

    print(classification_report(true, preds, target_names=le.classes_, zero_division=0))
    print(f"Modelo guardado en: {model_path}")
    print(f"Reporte guardado en: {report_path}")
    if visualizer:
        gif_path = visualizer.create_gif(fps=CONFIG.get("gif_fps", 2))
        if gif_path:
            print(f"GIF guardado en: {gif_path}")


if __name__ == "__main__":
    train()

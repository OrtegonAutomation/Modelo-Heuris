"""
FASE 3: ENTRENAMIENTO DEL MODELO BERT (BETO) MULTI-CABEZA
==========================================================
Sistema OCENSA-ML - Clasificacion de Modos y Causas de Falla ISO 14224

Este script entrena un modelo BETO (BERT en Espanol) con arquitectura multi-cabeza:
- Cabeza A: Clasificador de Modo de Falla (32 categorias)
- Cabeza B: Clasificador de Causa de Falla (45 categorias)  
- Cabeza C: Estimador de Prioridad (4 categorias)

Objetivo: F1-Score > 0.95 sobre data sintetica (linea base de comprension ISO)

Referencia: README.md Seccion 5 (Arquitectura) y 9.3 (Fase 3)
"""

import os
import sys
import json
import warnings
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)

# Configurar estilo visual
plt.style.use('dark_background')
sns.set_palette("husl")

class TrainingVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_history = []  # Datos por epoca (F1, Val Loss)
        self.batch_history = []    # Datos por batch (Train Loss)
        self.frames = []
        self.step_counter = 0
        
        # Crear carpeta de frames
        self.frames_dir = os.path.join(output_dir, "frames_temp")
        os.makedirs(self.frames_dir, exist_ok=True)
        
    def update_batch(self, loss_value):
        self.step_counter += 1
        self.batch_history.append({'step': self.step_counter, 'loss': loss_value})
        
        # Generar frame cada 50 pasos (para no saturar disco)
        if self.step_counter % 50 == 0:
            self._plot_frame()

    def update_epoch(self, epoch_data):
        self.metrics_history.append(epoch_data)
        # Forzar plot al final de epoca con datos de F1 actualizados
        self._plot_frame(end_of_epoch=True)
        
    def _plot_frame(self, end_of_epoch=False):
        # Crear figura con 2 subplots: Loss (izquierda) y F1 (derecha)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # -- Plot 1: Loss evolution (Batch level resolution) --
        df_batch = pd.DataFrame(self.batch_history)
        if not df_batch.empty:
            # Suavizado para que se vea bonito (rolling mean)
            window = 50 if len(df_batch) > 100 else 1
            smooth_loss = df_batch['loss'].rolling(window=window, min_periods=1).mean()
            
            ax1.plot(df_batch['step'], df_batch['loss'], alpha=0.3, color='#00f2ff', linewidth=1, label='Raw Loss')
            ax1.plot(df_batch['step'], smooth_loss, color='white', linewidth=2, label=f'Avg Loss ({window} steps)')
            
            ax1.set_title(f'Entrenamiento en Tiempo Real - Step {self.step_counter}', fontsize=12, color='white', fontweight='bold')
            ax1.set_xlabel('Steps (Lotes)')
            ax1.set_ylabel('Cross-Entropy Loss')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.1)
        
        # -- Plot 2: F1 Score Evolution (Epoch level) --
        if self.metrics_history:
            df_epoch = pd.DataFrame(self.metrics_history)
            epochs = df_epoch['epoch']
            
            f1_modo = [m['val_metrics']['modo']['f1'] for m in self.metrics_history]
            f1_causa = [m['val_metrics']['causa']['f1'] for m in self.metrics_history]
            f1_prio = [m['val_metrics']['prioridad']['f1'] for m in self.metrics_history]
            
            ax2.plot(epochs, f1_modo, marker='s', color='#ff00ff', linewidth=2, label='Modo')
            ax2.plot(epochs, f1_causa, marker='^', color='#ffff00', linewidth=2, label='Causa')
            ax2.plot(epochs, f1_prio, marker='o', color='#00ff00', linewidth=2, label='Prioridad')
            
            ax2.axhline(y=0.95, color='cyan', linestyle='--', alpha=0.8, label='Meta (0.95)')
            
            title_text = f"Metricas de Validacion (Epoca {len(self.metrics_history)})"
            if end_of_epoch:
                last_f1 = (f1_modo[-1] + f1_causa[-1]) / 2
                title_text += f"\nF1 Promedio Actual: {last_f1:.4f}"
                
            ax2.set_title(title_text, fontsize=12, color='white', fontweight='bold')
            ax2.set_xlabel('Epoca')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.1)
        else:
            ax2.text(0.5, 0.5, "Esperando primera epoca...", ha='center', va='center', color='gray')
        
        # Guardar frame
        filename = os.path.join(self.frames_dir, f"frame_{self.step_counter:06d}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        self.frames.append(filename)
        plt.close()
        
    def create_gifs(self):
        if not self.frames:
            print("No hay frames para generar GIF.")
            return

        print(f"Generando GIF cinemático con {len(self.frames)} frames...")
        gif_path = os.path.join(self.output_dir, 'training_cinematic.gif')
        
        # Leer imagenes y guardar gif
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer: # 10 fps
            for filename in self.frames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Limpiar carpeta temporal
        import shutil
        try:
            shutil.rmtree(self.frames_dir)
        except:
            pass
            
        print(f"GIF Cinemático guardado en: {gif_path}")

# Suprimir warnings de transformers
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Intentar importar PyTorch y Transformers
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer,
        AutoModel,
    get_linear_schedule_with_warmup
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ADVERTENCIA] PyTorch o Transformers no estan instalados.")
    print("Ejecute: pip install torch transformers")

# Intentar importar tqdm para barra de progreso
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable
    print("[INFO] tqdm no instalado, no se mostrara barra de progreso.")


# --- CONFIGURACION ---
CONFIG = {
    # Modelo base
    "model_name": "dccuchile/bert-base-spanish-wwm-cased",  # BETO
    "max_length": 256,  # Aumentado para capturar mas contexto técnico
    
    # Entrenamiento PRODUCCION
    "batch_size": 16,
    "epochs": 15,        # Aumentado para datos mas complejos
    "learning_rate": 3e-5, # Ligeramente mayor inicial
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "label_smoothing": 0.1, # Ayuda a generalizar y no sobreajustar
    "use_cause_class_weights": True,
    
    # Divisiones de datos
    "train_split": 0.8,
    "val_split": 0.1,       
    "test_split": 0.1,
    
    # Archivos
    "synthetic_data": "../fase1_generacion_sintetica/synthetic_training_data.csv",
    "output_model": "base_iso_model.bin",
    "output_report": "training_report.md",
    "label_encoders": "label_encoders.json",
    
    # Semilla
    "seed": 42,

    # Sanity checks / leakage guard
    "sanity_checks": True,
    "stratify_on": "auto",  # auto | causa | modo | combo
    "leakage_checks": {
        "enabled": True,
        "max_exact_overlap": 0.01,
        "max_normalized_overlap": 0.10,
        "max_text_purity": 0.98,
        "max_imbalance_ratio": 8.0,
    },
    "leakage_guard": {
        "enabled": True,
        "min_epoch": 2,
        "cause_f1_threshold": 0.98,
        "mode_f1_threshold": 0.60,
        "patience": 2,
    },
}

def set_seed(seed):
    """Configura semillas para reproducibilidad."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def normalize_for_leakage(text):
    """Normaliza texto para detectar duplicados por plantilla."""
    text = str(text).lower()
    text = re.sub(r"\b[a-z]{1,5}-?\d{2,6}[a-z]?\b", "tag", text)
    text = re.sub(r"\d+", "num", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_sanity_checks(df):
    """Chequear leakage y desbalances antes de entrenar."""
    if not CONFIG.get("sanity_checks", False):
        return

    checks = CONFIG.get("leakage_checks", {})
    if not checks.get("enabled", False):
        return

    print("\n[Sanity] Revisando leakage y desbalance...")

    # Desbalance de clases
    cause_counts = df['causa_falla'].value_counts()
    mode_counts = df['modo_falla'].value_counts()
    if not cause_counts.empty and not mode_counts.empty:
        cause_ratio = cause_counts.max() / max(cause_counts.min(), 1)
        mode_ratio = mode_counts.max() / max(mode_counts.min(), 1)
        if cause_ratio > checks.get("max_imbalance_ratio", 8.0):
            print(f"[ADVERTENCIA] Desbalance alto en causas (max/min={cause_ratio:.2f}).")
        if mode_ratio > checks.get("max_imbalance_ratio", 8.0):
            print(f"[ADVERTENCIA] Desbalance alto en modos (max/min={mode_ratio:.2f}).")

    # Heuristica de leakage: tokens exclusivos por causa
    text_norm = df['descripcion'].apply(normalize_for_leakage)
    token_cause_map = {}
    for cause, texts in df.groupby('causa_falla')['descripcion']:
        tokens = re.findall(r"\b[a-z]{4,}\b", " ".join(texts.astype(str)).lower())
        token_counts = Counter(tokens)
        top_tokens = [t for t, _ in token_counts.most_common(30)]
        for t in top_tokens:
            token_cause_map.setdefault(t, set()).add(cause)

    exclusive_tokens = {t for t, causes in token_cause_map.items() if len(causes) == 1}
    if exclusive_tokens:
        exclusive_rate = len(exclusive_tokens) / max(len(token_cause_map), 1)
        if exclusive_rate > checks.get("max_text_purity", 0.98):
            print(f"[ADVERTENCIA] Muchos tokens exclusivos por causa (rate={exclusive_rate:.3f}). Posible leakage.")

    # Overlap promedio de tokens entre causas
    cause_tokens = []
    for cause, texts in df.groupby('causa_falla')['descripcion']:
        tokens = re.findall(r"\b[a-z]{4,}\b", " ".join(texts.astype(str)).lower())
        token_counts = Counter(tokens)
        top_tokens = set([t for t, _ in token_counts.most_common(40)])
        cause_tokens.append(top_tokens)

    if len(cause_tokens) > 1:
        overlaps = []
        for i in range(len(cause_tokens)):
            for j in range(i + 1, len(cause_tokens)):
                a = cause_tokens[i]
                b = cause_tokens[j]
                denom = max(len(a | b), 1)
                overlaps.append(len(a & b) / denom)
        avg_overlap = sum(overlaps) / max(len(overlaps), 1)
        if avg_overlap < 0.06:
            print(f"[ADVERTENCIA] Overlap bajo entre tokens de causas (avg_jaccard={avg_overlap:.3f}). Posible leakage.")


def pick_stratify_labels(df):
    """Elige la estrategia de estratificacion segun configuracion y cardinalidad."""
    stratify_mode = CONFIG.get("stratify_on", "auto")
    if stratify_mode == "modo":
        return df['modo_falla'].astype(str).values
    if stratify_mode == "causa":
        return df['causa_falla'].astype(str).values
    if stratify_mode == "combo":
        combo = (df['modo_falla'].astype(str) + "||" + df['causa_falla'].astype(str))
        if combo.value_counts().min() < 2:
            return df['causa_falla'].astype(str).values
        return combo.values

    # auto
    combo = (df['modo_falla'].astype(str) + "||" + df['causa_falla'].astype(str))
    if combo.value_counts().min() >= 2:
        return combo.values
    return df['causa_falla'].astype(str).values


def check_split_overlap(df, train_idx, val_idx, test_idx):
    """Chequear textos repetidos entre splits para evitar leakage."""
    checks = CONFIG.get("leakage_checks", {})
    if not checks.get("enabled", False):
        return

    texts = df['texto_completo'].astype(str).values
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]

    def overlap_rate(a, b):
        set_a = set(a)
        set_b = set(b)
        if not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_b)

    exact_val = overlap_rate(train_texts, val_texts)
    exact_test = overlap_rate(train_texts, test_texts)
    if exact_val > checks.get("max_exact_overlap", 0.01):
        print(f"[ADVERTENCIA] Overlap exacto Train/Val alto: {exact_val:.2%}")
    if exact_test > checks.get("max_exact_overlap", 0.01):
        print(f"[ADVERTENCIA] Overlap exacto Train/Test alto: {exact_test:.2%}")

    norm_train = [normalize_for_leakage(t) for t in train_texts]
    norm_val = [normalize_for_leakage(t) for t in val_texts]
    norm_test = [normalize_for_leakage(t) for t in test_texts]
    norm_val_rate = overlap_rate(norm_train, norm_val)
    norm_test_rate = overlap_rate(norm_train, norm_test)
    if norm_val_rate > checks.get("max_normalized_overlap", 0.10):
        print(f"[ADVERTENCIA] Overlap normalizado Train/Val alto: {norm_val_rate:.2%}")
    if norm_test_rate > checks.get("max_normalized_overlap", 0.10):
        print(f"[ADVERTENCIA] Overlap normalizado Train/Test alto: {norm_test_rate:.2%}")


class MaintenanceDataset(Dataset):
    """Dataset personalizado para avisos de mantenimiento."""
    
    def __init__(self, texts, modo_labels, causa_labels, prioridad_labels, tokenizer, max_length):
        self.texts = texts
        self.modo_labels = modo_labels
        self.causa_labels = causa_labels
        self.prioridad_labels = prioridad_labels
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'modo_label': torch.tensor(self.modo_labels[idx], dtype=torch.long),
            'causa_label': torch.tensor(self.causa_labels[idx], dtype=torch.long),
            'prioridad_label': torch.tensor(self.prioridad_labels[idx], dtype=torch.long)
        }


class MultiHeadBERT(nn.Module):
    """
    Modelo BERT Multi-Cabeza para clasificacion de fallas.
    
    Arquitectura segun README Seccion 5:
    - Backbone: BETO (BERT Spanish)
    - Cabeza A: Modo de Falla
    - Cabeza B: Causa de Falla
    - Cabeza C: Prioridad
    """
    
    def __init__(self, model_name, num_modo_classes, num_causa_classes, num_prioridad_classes, dropout=0.3):
        super(MultiHeadBERT, self).__init__()
        
        # Backbone BERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Capa compartida con activacion GELU (segun README 5.2)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cabeza A: Modo de Falla (32 categorias)
        self.modo_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_modo_classes)
        )
        
        # Cabeza B: Causa de Falla (45 categorias)
        self.causa_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_causa_classes)
        )
        
        # Cabeza C: Prioridad (4 categorias)
        self.prioridad_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_prioridad_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Obtener embeddings de BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Usar el token [CLS] como representacion de la secuencia
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Capa compartida
        shared = self.shared_layer(cls_output)
        
        # Clasificaciones
        modo_logits = self.modo_classifier(shared)
        causa_logits = self.causa_classifier(shared)
        prioridad_logits = self.prioridad_classifier(shared)
        
        return modo_logits, causa_logits, prioridad_logits


def load_and_prepare_data():
    """Carga y prepara los datos sinteticos para entrenamiento."""
    print("\n[1/6] Cargando datos sinteticos...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, CONFIG["synthetic_data"])
    
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"   -> Registros cargados: {len(df):,}")
    
    # Verificar columnas requeridas
    required_cols = ['descripcion', 'modo_falla', 'causa_falla', 'prioridad']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en el dataset")
    
    # Crear texto combinado para mejor contexto
    df['texto_completo'] = df.apply(
        lambda row: f"{row['descripcion']}. Equipo: {row.get('equipo', '')}. Familia: {row.get('familia_equipo', '')}",
        axis=1
    )

    run_sanity_checks(df)
    
    # Codificar etiquetas
    le_modo = LabelEncoder()
    le_causa = LabelEncoder()
    le_prioridad = LabelEncoder()
    
    df['modo_encoded'] = le_modo.fit_transform(df['modo_falla'].astype(str))
    df['causa_encoded'] = le_causa.fit_transform(df['causa_falla'].astype(str))
    df['prioridad_encoded'] = le_prioridad.fit_transform(df['prioridad'].astype(str))
    
    print(f"   -> Modos de falla unicos: {len(le_modo.classes_)}")
    print(f"   -> Causas de falla unicas: {len(le_causa.classes_)}")
    print(f"   -> Niveles de prioridad: {len(le_prioridad.classes_)}")
    
    # Guardar encoders para inferencia posterior
    encoders = {
        'modo_falla': list(le_modo.classes_),
        'causa_falla': list(le_causa.classes_),
        'prioridad': list(le_prioridad.classes_)
    }
    
    encoders_path = os.path.join(script_dir, CONFIG["label_encoders"])
    with open(encoders_path, 'w', encoding='utf-8') as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)
    print(f"   -> Encoders guardados: {CONFIG['label_encoders']}")
    
    return df, le_modo, le_causa, le_prioridad


def create_data_loaders(df, tokenizer):
    """Crea los DataLoaders para entrenamiento, validacion y test."""
    print("\n[2/6] Preparando DataLoaders...")

    texts = df['texto_completo'].values
    modo_labels = df['modo_encoded'].values
    causa_labels = df['causa_encoded'].values
    prioridad_labels = df['prioridad_encoded'].values

    indices = np.arange(len(df))
    stratify_labels = pick_stratify_labels(df)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - CONFIG["train_split"]),
        random_state=CONFIG["seed"],
        stratify=stratify_labels
    )

    val_ratio = CONFIG["val_split"] / (CONFIG["val_split"] + CONFIG["test_split"])
    stratify_temp = np.array(stratify_labels)[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=CONFIG["seed"],
        stratify=stratify_temp
    )

    X_train = texts[train_idx]
    X_val = texts[val_idx]
    X_test = texts[test_idx]

    y_modo_train = modo_labels[train_idx]
    y_modo_val = modo_labels[val_idx]
    y_modo_test = modo_labels[test_idx]

    y_causa_train = causa_labels[train_idx]
    y_causa_val = causa_labels[val_idx]
    y_causa_test = causa_labels[test_idx]

    y_prio_train = prioridad_labels[train_idx]
    y_prio_val = prioridad_labels[val_idx]
    y_prio_test = prioridad_labels[test_idx]

    check_split_overlap(df, train_idx, val_idx, test_idx)
    
    print(f"   -> Train: {len(X_train):,} registros")
    print(f"   -> Validation: {len(X_val):,} registros")
    print(f"   -> Test: {len(X_test):,} registros")
    
    # Crear datasets
    train_dataset = MaintenanceDataset(
        X_train, y_modo_train, y_causa_train, y_prio_train,
        tokenizer, CONFIG["max_length"]
    )
    val_dataset = MaintenanceDataset(
        X_val, y_modo_val, y_causa_val, y_prio_val,
        tokenizer, CONFIG["max_length"]
    )
    test_dataset = MaintenanceDataset(
        X_test, y_modo_test, y_causa_test, y_prio_test,
        tokenizer, CONFIG["max_length"]
    )
    
    # Crear DataLoaders con optimizacion para CPU Intel (num_workers)
    # Windows requiere num_workers > 0 solo si se ejecuta protegida por if __name__ == "__main__"
    workers = 4 
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=workers, pin_memory=True)
    
    # Calcular pesos de clase para causas (opcional)
    num_causa_classes = int(df['causa_encoded'].nunique())
    cause_counts = np.bincount(y_causa_train, minlength=num_causa_classes)
    cause_weights = None
    if CONFIG.get("use_cause_class_weights", False):
        total = cause_counts.sum()
        cause_weights = total / (num_causa_classes * np.maximum(cause_counts, 1))
        cause_weights = torch.tensor(cause_weights, dtype=torch.float)

    return (
        train_loader,
        val_loader,
        test_loader,
        (X_test, y_modo_test, y_causa_test, y_prio_test),
        cause_weights,
    )


def train_epoch(model, data_loader, optimizer, scheduler, device, visualizer=None, causa_weights=None):
    """Entrena el modelo por una epoca con barra de progreso y actualizacion visual."""
    model.train()
    total_loss = 0
    
    # Usar Label Smoothing si está en config
    label_smoothing = CONFIG.get("label_smoothing", 0.0)
    criterion_modo = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if causa_weights is not None:
        causa_weights = causa_weights.to(device)
    criterion_causa = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=causa_weights)
    criterion_prioridad = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Barra de progreso con tqdm
    progress_bar = tqdm(data_loader, desc="Entrenando", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        modo_labels = batch['modo_label'].to(device)
        causa_labels = batch['causa_label'].to(device)
        prioridad_labels = batch['prioridad_label'].to(device)
        
        optimizer.zero_grad()
        
        modo_logits, causa_logits, prioridad_logits = model(input_ids, attention_mask)
        
        # Calcular perdidas con NUEVAS PONDERACIONES (Nivel 2)
        loss_modo = criterion_modo(modo_logits, modo_labels)
        loss_causa = criterion_causa(causa_logits, causa_labels)
        loss_prioridad = criterion_prioridad(prioridad_logits, prioridad_labels)
        
        # Causa: 0.6, Prioridad: 0.3, Modo: 0.1 (Enfoque en las cabezas mas dificiles)
        loss = 0.1 * loss_modo + 0.6 * loss_causa + 0.3 * loss_prioridad
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        
        # Actualizar barra
        progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        # Actualizar visualizador de alto detalle (Batch level)
        if visualizer:
            visualizer.update_batch(loss_val)
    
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """Evalua el modelo en un conjunto de datos."""
    model.eval()
    
    modo_preds, modo_true = [], []
    causa_preds, causa_true = [], []
    prioridad_preds, prioridad_true = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            modo_logits, causa_logits, prioridad_logits = model(input_ids, attention_mask)
            
            modo_preds.extend(torch.argmax(modo_logits, dim=1).cpu().numpy())
            causa_preds.extend(torch.argmax(causa_logits, dim=1).cpu().numpy())
            prioridad_preds.extend(torch.argmax(prioridad_logits, dim=1).cpu().numpy())
            
            modo_true.extend(batch['modo_label'].numpy())
            causa_true.extend(batch['causa_label'].numpy())
            prioridad_true.extend(batch['prioridad_label'].numpy())
    
    # Calcular metricas
    metrics = {
        'modo': {
            'accuracy': accuracy_score(modo_true, modo_preds),
            'f1': f1_score(modo_true, modo_preds, average='weighted'),
            'precision': precision_score(modo_true, modo_preds, average='weighted', zero_division=0),
            'recall': recall_score(modo_true, modo_preds, average='weighted', zero_division=0)
        },
        'causa': {
            'accuracy': accuracy_score(causa_true, causa_preds),
            'f1': f1_score(causa_true, causa_preds, average='weighted'),
            'precision': precision_score(causa_true, causa_preds, average='weighted', zero_division=0),
            'recall': recall_score(causa_true, causa_preds, average='weighted', zero_division=0)
        },
        'prioridad': {
            'accuracy': accuracy_score(prioridad_true, prioridad_preds),
            'f1': f1_score(prioridad_true, prioridad_preds, average='weighted'),
            'precision': precision_score(prioridad_true, prioridad_preds, average='weighted', zero_division=0),
            'recall': recall_score(prioridad_true, prioridad_preds, average='weighted', zero_division=0)
        }
    }
    
    return metrics, (modo_preds, causa_preds, prioridad_preds), (modo_true, causa_true, prioridad_true)


def generate_training_report(metrics_history, final_metrics, le_modo, le_causa, le_prioridad, training_time):
    """Genera un reporte detallado del entrenamiento."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, CONFIG["output_report"])
    
    report = f"""# REPORTE DE ENTRENAMIENTO - FASE 3
## Sistema OCENSA-ML - Modelo Base ISO 14224

**Fecha de entrenamiento:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Tiempo total:** {training_time:.2f} minutos

---

## 1. CONFIGURACION DEL MODELO

| Parametro | Valor |
|-----------|-------|
| Modelo Base | {CONFIG['model_name']} |
| Max Length | {CONFIG['max_length']} |
| Batch Size | {CONFIG['batch_size']} |
| Epochs | {CONFIG['epochs']} |
| Learning Rate | {CONFIG['learning_rate']} |
| Weight Decay | {CONFIG['weight_decay']} |

---

## 2. ARQUITECTURA MULTI-CABEZA

| Cabeza | Clases | Descripcion |
|--------|--------|-------------|
| Modo de Falla | {len(le_modo.classes_)} | ISO 14224 Failure Modes |
| Causa de Falla | {len(le_causa.classes_)} | ISO 14224 Failure Causes |
| Prioridad | {len(le_prioridad.classes_)} | Niveles de Urgencia |

---

## 3. METRICAS FINALES (Test Set)

### 3.1 Modo de Falla
| Metrica | Valor |
|---------|-------|
| **F1-Score** | {final_metrics['modo']['f1']:.4f} |
| Accuracy | {final_metrics['modo']['accuracy']:.4f} |
| Precision | {final_metrics['modo']['precision']:.4f} |
| Recall | {final_metrics['modo']['recall']:.4f} |

### 3.2 Causa de Falla
| Metrica | Valor |
|---------|-------|
| **F1-Score** | {final_metrics['causa']['f1']:.4f} |
| Accuracy | {final_metrics['causa']['accuracy']:.4f} |
| Precision | {final_metrics['causa']['precision']:.4f} |
| Recall | {final_metrics['causa']['recall']:.4f} |

### 3.3 Prioridad
| Metrica | Valor |
|---------|-------|
| **F1-Score** | {final_metrics['prioridad']['f1']:.4f} |
| Accuracy | {final_metrics['prioridad']['accuracy']:.4f} |
| Precision | {final_metrics['prioridad']['precision']:.4f} |
| Recall | {final_metrics['prioridad']['recall']:.4f} |

---

## 4. OBJETIVO DE FASE 3

| Objetivo | Requerido | Alcanzado | Status |
|----------|-----------|-----------|--------|
| F1-Score Modo de Falla | > 0.95 | {final_metrics['modo']['f1']:.4f} | {'CUMPLIDO' if final_metrics['modo']['f1'] > 0.95 else 'PENDIENTE'} |
| F1-Score Causa de Falla | > 0.95 | {final_metrics['causa']['f1']:.4f} | {'CUMPLIDO' if final_metrics['causa']['f1'] > 0.95 else 'PENDIENTE'} |

---

## 5. HISTORIAL DE ENTRENAMIENTO

| Epoca | Loss Train | F1 Modo (Val) | F1 Causa (Val) | F1 Prioridad (Val) |
|-------|------------|---------------|----------------|---------------------|
"""
    
    for epoch_data in metrics_history:
        report += f"| {epoch_data['epoch']} | {epoch_data['train_loss']:.4f} | {epoch_data['val_metrics']['modo']['f1']:.4f} | {epoch_data['val_metrics']['causa']['f1']:.4f} | {epoch_data['val_metrics']['prioridad']['f1']:.4f} |\n"
    
    report += f"""
---

## 6. CLASES DE MODO DE FALLA

"""
    for i, clase in enumerate(le_modo.classes_):
        report += f"{i+1}. {clase}\n"
    
    report += f"""
---

## 7. CLASES DE CAUSA DE FALLA

"""
    for i, clase in enumerate(le_causa.classes_):
        report += f"{i+1}. {clase}\n"
    
    report += f"""
---

## 8. ARCHIVOS GENERADOS

- `{CONFIG['output_model']}`: Modelo entrenado
- `{CONFIG['label_encoders']}`: Mapeo de etiquetas
- `{CONFIG['output_report']}`: Este reporte

---

*Generado automaticamente por Sistema OCENSA-ML - Fase 3*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def train_model():
    """Funcion principal de entrenamiento."""
    print("=" * 60)
    print("FASE 3: ENTRENAMIENTO DEL MODELO BERT - SISTEMA OCENSA-ML")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("\n[ERROR] PyTorch no esta disponible.")
        print("Instale con: pip install torch transformers scikit-learn")
        return None
    
    start_time = datetime.now()
    set_seed(CONFIG["seed"])
    
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n-> Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Cargar datos
    df, le_modo, le_causa, le_prioridad = load_and_prepare_data()
    
    # 2. Cargar tokenizer
    print("\n[2/6] Cargando tokenizer BETO...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    print(f"   -> Tokenizer cargado: {CONFIG['model_name']}")
    
    # 3. Crear DataLoaders
    train_loader, val_loader, test_loader, test_data, causa_weights = create_data_loaders(df, tokenizer)
    
    # 4. Crear modelo
    print("\n[3/6] Inicializando modelo Multi-Cabeza...")
    model = MultiHeadBERT(
        model_name=CONFIG["model_name"],
        num_modo_classes=len(le_modo.classes_),
        num_causa_classes=len(le_causa.classes_),
        num_prioridad_classes=len(le_prioridad.classes_)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> Parametros totales: {total_params:,}")
    print(f"   -> Parametros entrenables: {trainable_params:,}")
    
    # 5. Configurar optimizador y scheduler
    print("\n[4/6] Configurando optimizador...")
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"   -> Total steps: {total_steps:,}")
    print(f"   -> Warmup steps: {warmup_steps:,}")
    
    # 6. Entrenamiento
    print("\n[5/6] Iniciando entrenamiento...")
    print("-" * 60)
    
    metrics_history = []
    best_f1 = 0
    best_model_state = None
    leakage_hits = 0
    leakage_triggered = False
    
    # Inicializar visualizador visual de alto impacto (GIFs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualizer = TrainingVisualizer(script_dir)
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoca {epoch + 1}/{CONFIG['epochs']}")
        
        # Train (Pasamos visualizer para updates en tiempo real)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, visualizer, causa_weights)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        # Guardar historial
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_metrics': val_metrics
        }
        metrics_history.append(epoch_data)
        
        # Actualizar visualizacion fin de epoca
        visualizer.update_epoch(epoch_data)
        
        # Mostrar progreso
        print(f"   Loss: {train_loss:.4f}")
        print(f"   F1 Modo: {val_metrics['modo']['f1']:.4f} | F1 Causa: {val_metrics['causa']['f1']:.4f} | F1 Prio: {val_metrics['prioridad']['f1']:.4f}")

        # Guard de leakage para evitar entrenamientos largos en data facil
        leak_cfg = CONFIG.get("leakage_guard", {})
        if leak_cfg.get("enabled", False):
            min_epoch = leak_cfg.get("min_epoch", 2)
            cause_thr = leak_cfg.get("cause_f1_threshold", 0.98)
            mode_thr = leak_cfg.get("mode_f1_threshold", 0.60)
            patience = leak_cfg.get("patience", 2)

            if (epoch + 1) >= min_epoch and val_metrics['causa']['f1'] >= cause_thr and val_metrics['modo']['f1'] <= mode_thr:
                leakage_hits += 1
                print(f"   [ADVERTENCIA] F1 Causa muy alto con F1 Modo bajo ({leakage_hits}/{patience}). Posible leakage.")
            else:
                leakage_hits = 0

            if leakage_hits >= patience:
                leakage_triggered = True
                print("   [ALERTA] Entrenamiento detenido por sospecha de leakage en datos sinteticos.")
                print("   [SUGERENCIA] Regenerar dataset con mayor solapamiento de contextos/ruido.")
                break
        
        # Guardar mejor modelo
        avg_f1 = (val_metrics['modo']['f1'] + val_metrics['causa']['f1']) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_state = model.state_dict().copy()
            print(f"   -> Nuevo mejor modelo guardado (F1 promedio: {best_f1:.4f})")
    
    # Cargar mejor modelo para evaluacion final
    if leakage_triggered:
        return None, None
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 7. Evaluacion final en test set
    print("\n[6/6] Evaluacion final en Test Set...")
    print("-" * 60)
    
    final_metrics, predictions, ground_truth = evaluate(model, test_loader, device)
    
    print(f"\n   RESULTADOS FINALES:")
    print(f"   {'='*40}")
    print(f"   Modo de Falla:")
    print(f"      F1-Score:  {final_metrics['modo']['f1']:.4f}")
    print(f"      Accuracy:  {final_metrics['modo']['accuracy']:.4f}")
    print(f"   Causa de Falla:")
    print(f"      F1-Score:  {final_metrics['causa']['f1']:.4f}")
    print(f"      Accuracy:  {final_metrics['causa']['accuracy']:.4f}")
    print(f"   Prioridad:")
    print(f"      F1-Score:  {final_metrics['prioridad']['f1']:.4f}")
    print(f"      Accuracy:  {final_metrics['prioridad']['accuracy']:.4f}")
    
    # Verificar objetivo
    print(f"\n   VERIFICACION OBJETIVO FASE 3:")
    modo_ok = final_metrics['modo']['f1'] > 0.95
    causa_ok = final_metrics['causa']['f1'] > 0.95
    print(f"   F1 Modo > 0.95: {'SI' if modo_ok else 'NO'} ({final_metrics['modo']['f1']:.4f})")
    print(f"   F1 Causa > 0.95: {'SI' if causa_ok else 'NO'} ({final_metrics['causa']['f1']:.4f})")
    
    # Guardar modelo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, CONFIG["output_model"])
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'label_encoders': {
            'modo_falla': list(le_modo.classes_),
            'causa_falla': list(le_causa.classes_),
            'prioridad': list(le_prioridad.classes_)
        },
        'final_metrics': final_metrics
    }, model_path)
    print(f"\n   -> Modelo guardado: {CONFIG['output_model']}")
    
    # Calcular tiempo total
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60
    
    # Generar reporte
    report_path = generate_training_report(
        metrics_history, final_metrics, 
        le_modo, le_causa, le_prioridad,
        training_time
    )
    print(f"   -> Reporte guardado: {CONFIG['output_report']}")
    
    # Generar GIFs animados
    visualizer.create_gifs()
    
    print("\n" + "=" * 60)
    print("FASE 3 COMPLETADA")
    print("=" * 60)
    print(f"\nTiempo total: {training_time:.2f} minutos")
    print(f"\nArchivos generados:")
    print(f"  - {CONFIG['output_model']}")
    print(f"  - {CONFIG['label_encoders']}")
    print(f"  - {CONFIG['output_report']}")
    
    return model, final_metrics


if __name__ == "__main__":
    model, metrics = train_model()

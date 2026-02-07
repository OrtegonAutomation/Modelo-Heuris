"""
FASE 4: EVALUACION SOBRE TESTER REAL (PRUEBA DE FUEGO)
======================================================
Sistema OCENSA-ML - Inferencia Masiva y Auditoria de Incertidumbre

Este script aplica el modelo entrenado sobre los datos reales de IW69.
Objetivos:
1. Inferencia masiva sobre 10,000+ registros reales.
2. Identificacion de casos de baja confianza ("Duda del Modelo").
3. Generacion de "Reporte de Contraste" para auditoria experta.

Output:
- IW69_inference_results.csv: Resultados completos
- top_500_uncertainty_audit.csv: Casos para revision humana
- evaluation_report.md: Metricas de confianza y distribucion
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm

# --- CONFIGURACION ---
CONFIG = {
    "model_path": "../fase3_entrenamiento_bert/base_iso_model.bin",
    "data_path": "../fase2_limpieza_etl/IW69_preprocessed.csv",
    "output_dir": ".",
    "batch_size": 32,
    "confidence_threshold": 0.6  # Umbral para considerar "duda" (Least Confidence)
}

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
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
            'attention_mask': encoding['attention_mask'].flatten()
        }

class MultiHeadBERT(torch.nn.Module):
    """Misma arquitectura que en entrenamiento para poder cargar los pesos"""
    def __init__(self, model_name, num_modo_classes, num_causa_classes, num_prioridad_classes, dropout=0.3):
        super(MultiHeadBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        self.modo_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, num_modo_classes)
        )
        self.causa_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, num_causa_classes)
        )
        self.prioridad_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 4, num_prioridad_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        shared = self.shared_layer(cls_output)
        return self.modo_classifier(shared), self.causa_classifier(shared), self.prioridad_classifier(shared)

def load_model_and_metadata():
    print("[1/5] Cargando modelo y metadatos...")
    checkpoint = torch.load(CONFIG["model_path"], map_location=torch.device('cpu'))
    model_config = checkpoint['config']
    encoders = checkpoint['label_encoders']
    
    # Reconstruir modelo
    model = MultiHeadBERT(
        model_config['model_name'],
        len(encoders['modo_falla']),
        len(encoders['causa_falla']),
        len(encoders['prioridad'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    return model, tokenizer, encoders, model_config

def run_inference():
    # 1. Cargar recursos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, CONFIG["model_path"])
    
    if not os.path.exists(model_path):
        print(f"ERROR: No se encuentra el modelo en {model_path}")
        print("Asegurese de completar la Fase 3 primero.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    model, tokenizer, encoders, model_config = load_model_and_metadata()
    model.to(device)
    
    # 2. Cargar datos reales
    print("[2/5] Cargando datos reales...")
    data_path = os.path.join(script_dir, CONFIG["data_path"])
    df = pd.read_csv(data_path)
    print(f"   -> {len(df)} registros cargados")
    
    # Preparar texto de entrada (mismo formato que entrenamiento)
    # Usamos descripcion_normalizada si existe, sino la original
    text_col = 'descripcion_normalizada' if 'descripcion_normalizada' in df.columns else 'descripcion'
    
    # Crear texto enriquecido con contexto
    # NOTA: Usar 'familia_equipo_detectada' del ETL como 'familia_equipo'
    texts = df.apply(
        lambda row: f"{row.get(text_col, '')}. Equipo: {row.get('equipo', '')}. Familia: {row.get('familia_equipo_detectada', '')}",
        axis=1
    ).tolist()
    
    dataset = InferenceDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 3. Inferencia
    print("[3/5] Ejecutando inferencia masiva...")
    all_modo_probs = []
    all_causa_probs = []
    all_prio_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Procesando lotes"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            modo_logits, causa_logits, prio_logits = model(input_ids, attention_mask)
            
            # Softmax para probabilidades
            all_modo_probs.append(F.softmax(modo_logits, dim=1).cpu().numpy())
            all_causa_probs.append(F.softmax(causa_logits, dim=1).cpu().numpy())
            all_prio_probs.append(F.softmax(prio_logits, dim=1).cpu().numpy())
            
    # Concatenar resultados
    modo_probs = np.vstack(all_modo_probs)
    causa_probs = np.vstack(all_causa_probs)
    prio_probs = np.vstack(all_prio_probs)
    
    # 4. Procesar resultados y metricas de confianza
    print("[4/5] Analizando confianza y generando reporte...")
    
    # Obtener predicciones y scores de confianza (max probability)
    modo_pred_idxs = np.argmax(modo_probs, axis=1)
    modo_conf = np.max(modo_probs, axis=1)
    
    causa_pred_idxs = np.argmax(causa_probs, axis=1)
    causa_conf = np.max(causa_probs, axis=1)
    
    prio_pred_idxs = np.argmax(prio_probs, axis=1)
    prio_conf = np.max(prio_probs, axis=1)
    
    # Mapear indices a etiquetas
    df['pred_modo_falla'] = [encoders['modo_falla'][i] for i in modo_pred_idxs]
    df['conf_modo'] = modo_conf
    
    df['pred_causa_falla'] = [encoders['causa_falla'][i] for i in causa_pred_idxs]
    df['conf_causa'] = causa_conf
    
    df['pred_prioridad'] = [encoders['prioridad'][i] for i in prio_pred_idxs]
    df['conf_prioridad'] = prio_conf
    
    # Puntaje global de confianza (promedio)
    df['confianza_global'] = (df['conf_modo'] + df['conf_causa']) / 2
    
    # Identificar incertidumbre
    df['baja_confianza'] = df['confianza_global'] < CONFIG["confidence_threshold"]
    
    # Guardar resultados completos
    output_csv = os.path.join(script_dir, "IW69_inference_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"   -> Resultados guardados en: {output_csv}")
    
    # Extraer Top 500 para auditoria (menor confianza)
    audit_df = df.sort_values('confianza_global', ascending=True).head(500)
    audit_csv = os.path.join(script_dir, "top_500_uncertainty_audit.csv")
    
    # Seleccionar columnas relevantes para auditoria
    cols_audit = [
        'id_aviso', 'equipo', 'descripcion', 'familia_equipo_detectada',
        'pred_modo_falla', 'conf_modo',
        'pred_causa_falla', 'conf_causa',
        'confianza_global'
    ]
    # Filtrar solo columnas existentes
    cols_audit = [c for c in cols_audit if c in df.columns]
    
    audit_df[cols_audit].to_csv(audit_csv, index=False)
    print(f"   -> Top 500 para auditoria: {audit_csv}")
    
    # Generar reporte Markdown
    generate_report(df, audit_df, script_dir)
    print("[5/5] Fase 4 completada exitosamente.")

def generate_report(df, audit_df, output_dir):
    avg_conf = df['confianza_global'].mean()
    high_conf_pct = (len(df[df['confianza_global'] >= 0.8]) / len(df)) * 100
    low_conf_pct = (len(df[df['confianza_global'] < 0.6]) / len(df)) * 100
    
    top_modos = df['pred_modo_falla'].value_counts().head(5)
    top_causas = df['pred_causa_falla'].value_counts().head(5)
    
    report = f"""# REPORTE DE EVALUACION MASIVA - FASE 4
## Auditoria de Inferencia sobre Datos Reales (IW69)

**Total Registros:** {len(df):,}
**Confianza Promedio del Modelo:** {avg_conf:.2%}

---

## 1. DISTRIBUCION DE CONFIANZA

El modelo se siente seguro en sus predicciones?

- **Alta Confianza (> 80%):** {high_conf_pct:.1f}% de los casos.
- **Incertidumbre (< 60%):** {low_conf_pct:.1f}% de los casos. -> **Requieren Revision Humana**

*Nota: Una confianza alta indica que el aviso real se parece mucho a los datos sinteticos entrenados. Una confianza baja indica "Drift" o situaciones nuevas no vistas.*

---

## 2. PREDICCIONES MAS FRECUENTES

### Top 5 Modos de Falla Detectados
"""
    for label, count in top_modos.items():
        report += f"- **{label}**: {count:,} ({count/len(df):.1%})\n"

    report += """
### Top 5 Causas de Falla Detectadas
"""
    for label, count in top_causas.items():
        report += f"- **{label}**: {count:,} ({count/len(df):.1%})\n"
        
    report += f"""
---

## 3. AUDITORIA DE INCERTIDUMBRE (ACTIVE LEARNING)

Se han extraido los **500 casos mas dificiles** para el modelo en el archivo:
`top_500_uncertainty_audit.csv`

### Lote de Muestra para Ingenieria (Top 5 mas confusos):
"""
    # Mostrar 5 ejemplos del audit file
    sample = audit_df.head(5)
    for _, row in sample.iterrows():
        desc = row.get('descripcion', 'N/A')[:50] + "..."
        report += f"1. Aviso {row.get('id_aviso', '?')}: *'{desc}'*\n"
        report += f"   - Prediccion: **{row['pred_modo_falla']}** (Conf: {row['conf_modo']:.2f})\n"
        report += f"   - Causa: **{row['pred_causa_falla']}** (Conf: {row['conf_causa']:.2f})\n\n"

    report += """
---
*Generado por Sistema OCENSA-ML - Fase 4*
"""
    
    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    run_inference()

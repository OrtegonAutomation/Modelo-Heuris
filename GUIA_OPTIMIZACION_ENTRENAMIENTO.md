# Guía de Optimización del Entrenamiento — OCENSA-ML

> ¿Tu modelo BERT (BETO) se demora demasiado en tu computadora personal?  
> Aquí tienes las estrategias más efectivas para acelerar el entrenamiento sin sacrificar calidad.

---

## Resumen Rápido

| Técnica | Ahorro estimado de tiempo | Dificultad | Impacto en calidad |
|---------|--------------------------|------------|-------------------|
| Entrenamiento con **GPU en la nube** (Colab) | 5-10x más rápido | ⭐ Fácil | Sin impacto |
| **Mixed Precision** (fp16) | 2-3x más rápido | ⭐ Fácil | Mínimo |
| Reducir **max_length** de 256 a 128 | ~2x más rápido | ⭐ Fácil | Bajo-medio |
| **Congelar capas** de BERT | 1.5-2x más rápido | ⭐⭐ Medio | Bajo |
| **Gradient Accumulation** | Permite batch más grande sin más RAM | ⭐⭐ Medio | Positivo |
| Usar modelo más pequeño (**DistilBETO**) | 2-3x más rápido | ⭐⭐ Medio | Bajo-medio |
| **Early Stopping** | Evita épocas innecesarias | ⭐ Fácil | Positivo |
| Reducir **épocas** (15 → 8-10) | ~40% menos tiempo | ⭐ Fácil | Monitorear |

---

## 1. Usa Google Colab (GPU Gratis) ⭐ RECOMENDADO

La forma más rápida de acelerar tu entrenamiento es usar una GPU en la nube. Google Colab ofrece GPUs Tesla T4 gratis.

### Pasos:

1. Ve a [Google Colab](https://colab.research.google.com)
2. Sube tu proyecto o clónalo desde GitHub:
   ```python
   !git clone https://github.com/tu-usuario/Modelo-Heuris.git
   %cd Modelo-Heuris
   ```
3. Activa la GPU: **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)**
4. Instala dependencias:
   ```python
   !pip install torch transformers tqdm scikit-learn pandas matplotlib seaborn imageio
   ```
5. Ejecuta el entrenamiento:
   ```python
   %cd fase3_entrenamiento_bert
   !python train_bert_model.py
   ```

> 💡 **Tiempo estimado**: En CPU local → 24-48 horas. En GPU Colab T4 → 2-4 horas.

---

## 2. Mixed Precision Training (fp16) ⭐ RECOMENDADO

Entrena con números de 16 bits en lugar de 32 bits. Reduce uso de memoria y acelera los cálculos en GPUs modernas.

### Cambios en `train_bert_model.py` (Fase 3):

```python
# Agregar al inicio del archivo
from torch.cuda.amp import autocast, GradScaler

# Crear scaler UNA SOLA VEZ antes del loop de épocas
scaler = GradScaler()
```

Modificar la función `train_epoch` para recibir el `scaler` como parámetro:

```python
def train_epoch(model, data_loader, optimizer, scheduler, device, scaler, visualizer=None, causa_weights=None):
    model.train()
    total_loss = 0

    label_smoothing = CONFIG.get("label_smoothing", 0.0)
    criterion_modo = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if causa_weights is not None:
        causa_weights = causa_weights.to(device)
    criterion_causa = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=causa_weights)
    criterion_prioridad = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    progress_bar = tqdm(data_loader, desc="Entrenando", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        modo_labels = batch['modo_label'].to(device)
        causa_labels = batch['causa_label'].to(device)
        prioridad_labels = batch['prioridad_label'].to(device)

        optimizer.zero_grad()

        # ✅ Mixed Precision
        with autocast():
            modo_logits, causa_logits, prioridad_logits = model(input_ids, attention_mask)
            loss_modo = criterion_modo(modo_logits, modo_labels)
            loss_causa = criterion_causa(causa_logits, causa_labels)
            loss_prioridad = criterion_prioridad(prioridad_logits, prioridad_labels)
            loss = 0.1 * loss_modo + 0.6 * loss_causa + 0.3 * loss_prioridad

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})

        if visualizer:
            visualizer.update_batch(loss_val)

    return total_loss / len(data_loader)
```

> 💡 **Resultado**: ~2x más rápido en GPU, reduce memoria VRAM ~40%.

---

## 3. Reducir `max_length` (256 → 128)

Tu configuración actual usa `max_length: 256`. La mayoría de las descripciones de falla industrial son cortas. Reducir a 128 tokens corta el tiempo casi a la mitad.

### Cambio en el CONFIG:

```python
# En fase3_entrenamiento_bert/train_bert_model.py
CONFIG = {
    "max_length": 128,  # Antes: 256 — La mayoría de textos caben en 128
    # ... resto igual
}
```

### Cómo verificar que no pierdes información:

```python
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
df = pd.read_csv("data/synthetic_training_data.csv")

# Ver la distribución de longitudes de tokens
texts = df['descripcion'].astype(str).tolist()
encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
lengths = pd.Series([len(ids) for ids in encoded['input_ids']])
print(f"Percentil 95: {lengths.quantile(0.95):.0f} tokens")
print(f"Percentil 99: {lengths.quantile(0.99):.0f} tokens")
print(f"Máximo: {lengths.max()} tokens")
```

Si el percentil 95 es menor a 128, puedes usar 128 sin problema.

> 💡 **Resultado**: ~2x más rápido (la complejidad de BERT es O(n²) con respecto a la longitud).

---

## 4. Congelar Capas de BERT

BERT tiene 12 capas de transformer. Las capas inferiores capturan gramática general y rara vez necesitan re-entrenarse. Congela las primeras 8-10 capas y solo entrena las últimas 2-4.

### Agregar después de crear el modelo:

```python
# Congelar embeddings y las primeras 10 capas del encoder
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for i, layer in enumerate(model.bert.encoder.layer):
    if i < 10:  # Congelar capas 0-9, entrenar solo 10-11
        for param in layer.parameters():
            param.requires_grad = False

# Verificar parámetros entrenables
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros: {trainable:,} entrenables de {total:,} totales ({trainable/total*100:.1f}%)")
```

> 💡 **Resultado**: ~1.5-2x más rápido, menos memoria. Solo entrena ~15% de los parámetros.

---

## 5. Gradient Accumulation

Si tu GPU/CPU no tiene suficiente memoria para `batch_size=16`, puedes simular un batch grande acumulando gradientes en varios pasos pequeños.

### Ejemplo (simular batch 16 con mini-batches de 4):

```python
CONFIG = {
    "batch_size": 4,                 # Batch real en memoria
    "gradient_accumulation_steps": 4, # 4 * 4 = 16 batch efectivo
    # ...
}

# En el loop de entrenamiento:
accumulation_steps = CONFIG.get("gradient_accumulation_steps", 1)
optimizer.zero_grad()

for step, batch in enumerate(progress_bar):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    # ... forward pass y cálculo de loss (igual que en train_epoch) ...
    loss = loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

> 💡 **Resultado**: Permite entrenar en GPUs con poca VRAM (4-6 GB) o incluso en CPU con menos RAM.

---

## 6. Usar un Modelo Más Pequeño (DistilBETO)

Si el tiempo sigue siendo un problema, considera usar una versión destilada de BERT en español que es ~40% más pequeña y 60% más rápida.

### Cambio en CONFIG:

```python
CONFIG = {
    # Opción 1: DistilBERT multilingüe (incluye español)
    "model_name": "distilbert-base-multilingual-cased",

    # Opción 2: BETO original (actual)
    # "model_name": "dccuchile/bert-base-spanish-wwm-cased",
}
```

> ⚠️ **Nota**: DistilBERT multilingüe puede ser ligeramente menos preciso que BETO para texto técnico en español, pero la diferencia suele ser menor al 2-3% en F1.

---

## 7. Early Stopping (Parada Anticipada)

Evita entrenar épocas innecesarias si el modelo ya dejó de mejorar. Fase 5 ya lo tiene, pero Fase 3 no.

### Agregar al loop principal de Fase 3:

```python
best_val_f1 = 0
patience_counter = 0
patience_limit = 3  # Detener si no mejora en 3 épocas

for epoch in range(CONFIG["epochs"]):
    train_loss = train_epoch(...)
    val_results = evaluate(...)
    
    current_f1 = val_results["causa_f1"]  # Monitorear la métrica más importante
    
    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        patience_counter = 0
        # Guardar el mejor modelo
        torch.save(model.state_dict(), "best_model.bin")
    else:
        patience_counter += 1
    
    if patience_counter >= patience_limit:
        print(f"[Early Stopping] Sin mejora en {patience_limit} épocas. Deteniendo.")
        break
```

> 💡 **Resultado**: Si el modelo converge en la época 8, te ahorras 7 épocas (~47% del tiempo).

---

## 8. Reducir Épocas y Monitorear

En tu configuración actual tienes `epochs: 15`. Para iteración rápida:

```python
CONFIG = {
    "epochs": 5,  # Para pruebas rápidas
    # "epochs": 15,  # Para entrenamiento final
}
```

Consejo: Entrena primero 5 épocas para verificar que todo funciona, luego escala a 15.

---

## 9. Optimización del DataLoader

Mejora la velocidad de carga de datos con más workers y memoria pinned:

```python
# En la creación del DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=4,        # Usar 4 workers para cargar datos en paralelo
    pin_memory=True,      # Carga más rápida a GPU
    persistent_workers=True,  # No reiniciar workers entre épocas (requiere num_workers > 0)
)
```

> 💡 En **Windows**, `num_workers > 0` puede causar errores. Si es tu caso, usa `num_workers=0` y elimina `persistent_workers`.

---

## 10. Resumen de Cambios Recomendados por Prioridad

### 🟢 Prioridad Alta (Haz esto primero)

| # | Cambio | Archivo | Tiempo de implementación |
|---|--------|---------|-------------------------|
| 1 | Usa Google Colab con GPU T4 | N/A (infraestructura) | 10 minutos |
| 2 | Reduce `max_length` a 128 | `fase3/.../train_bert_model.py` | 1 minuto |
| 3 | Agrega Early Stopping | `fase3/.../train_bert_model.py` | 15 minutos |

### 🟡 Prioridad Media (Si necesitas más velocidad)

| # | Cambio | Archivo | Tiempo de implementación |
|---|--------|---------|-------------------------|
| 4 | Activa Mixed Precision (fp16) | `fase3/.../train_bert_model.py` | 20 minutos |
| 5 | Congela capas 0-9 de BERT | `fase3/.../train_bert_model.py` | 10 minutos |
| 6 | Gradient Accumulation (batch=4, accum=4) | `fase3/.../train_bert_model.py` | 15 minutos |

### 🔵 Prioridad Baja (Para experimentación)

| # | Cambio | Archivo | Tiempo de implementación |
|---|--------|---------|-------------------------|
| 7 | Usar DistilBERT multilingüe | CONFIG en cada script | 5 minutos |
| 8 | Optimizar DataLoader workers | Cada script de entrenamiento | 5 minutos |

---

## Comparación de Tiempos Estimados

| Configuración | Tiempo Fase 3 (15 épocas) | Tiempo Total (3 modelos) |
|---------------|--------------------------|-------------------------|
| CPU sin optimizar (actual) | 8-16 horas | 24-48 horas |
| CPU + max_len=128 + freeze | 3-6 horas | 9-18 horas |
| GPU Colab T4 sin optimizar | 1-2 horas | 3-6 horas |
| GPU Colab T4 + fp16 + freeze | 30-60 min | 1.5-3 horas |
| GPU Colab T4 + fp16 + DistilBERT | 20-40 min | 1-2 horas |

---

## Recursos Adicionales

- [Google Colab](https://colab.research.google.com) — GPU gratis
- [Kaggle Notebooks](https://www.kaggle.com/code) — GPU P100 gratis (30h/semana)
- [Hugging Face Mixed Precision](https://huggingface.co/docs/transformers/perf_train_gpu_one#fp16-training) — Documentación oficial
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) — Automatic Mixed Precision

---

> 📝 **Nota**: Todas estas optimizaciones son compatibles entre sí. Puedes combinar Colab + fp16 + freeze + max_len=128 para obtener el máximo beneficio.

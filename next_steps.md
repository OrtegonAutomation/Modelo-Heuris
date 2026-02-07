# Proximos Pasos - Sistema OCENSA-ML

## Estado Actual del Proyecto

| Modelo | F1-Score Actual | Meta | Status |
|--------|----------------|------|--------|
| Modo de Falla (Fase 3 multi-cabeza) | 0.9929 | > 0.95 | ✅ CUMPLIDO |
| Causa de Falla (Fase 3 multi-cabeza) | 0.2758 | > 0.95 | ❌ PENDIENTE |
| Causa ISO (Fase 5 single-head) | 0.4488 macro / 0.5124 weighted | > 0.95 | ❌ PENDIENTE |
| Prioridad (Fase 3 multi-cabeza) | 0.4826 | > 0.95 | ❌ PENDIENTE |
| Prioridad (Fase 5 single-head) | Sin entrenar aun | > 0.95 | ❌ PENDIENTE |

---

## PARTE 1: Como Mejorar F1 en Causa y Prioridad

### 1.1 Problema Raiz Identificado

El modelo multi-cabeza de Fase 3 tenia los pesos de la funcion de perdida invertidos:

```python
# ANTES (incorrecto) — 70% del esfuerzo en Modo que ya estaba en 0.99
loss = 0.7 * loss_modo + 0.1 * loss_causa + 0.2 * loss_prioridad

# CORREGIDO — Enfoque en las cabezas dificiles
loss = 0.1 * loss_modo + 0.6 * loss_causa + 0.3 * loss_prioridad
```

Esto ya fue corregido en el commit mas reciente. Al re-entrenar Fase 3, el F1 de Causa y Prioridad deberia subir significativamente.

### 1.2 Pasos para Mejorar Fase 3 (Multi-Cabeza)

1. **Re-entrenar `fase3_entrenamiento_bert/train_bert_model.py`** con los pesos corregidos
   ```bash
   cd fase3_entrenamiento_bert
   python train_bert_model.py
   ```
2. **Verificar** que F1 Causa sube por encima de 0.70 en las primeras epocas
3. Si despues de 15 epocas no llega a 0.95, ajustar:
   - Aumentar `epochs` a 20-25
   - Probar `label_smoothing: 0.05` (actualmente 0.1)
   - Agregar Focal Loss para Causa (como ya existe en Fase 5)

### 1.3 Pasos para Mejorar Fase 5 - Causa ISO (Single-Head)

El modelo de Causa ISO en Fase 5 ya fue optimizado con estos cambios:

| Parametro | Antes | Despues | Razon |
|-----------|-------|---------|-------|
| `min_class_count` | 100 | 30 | Mantiene mas clases (11 → ~18 clases) |
| `use_class_weights` | False | True | Compensa desbalance severo |
| `epochs` | 8 | 15 | Mas tiempo de convergencia |
| `early_stopping_patience` | 2 | 5 | Evita parada prematura |
| `learning_rate` | 3e-5 | 2e-5 | Convergencia mas estable |

**Para entrenar:**
```bash
cd fase5_fine_tuning
python train_causa_iso.py
```

**Si no llega a 0.95 con estos cambios, probar adicionalmente:**

1. **Data Augmentation**: Generar mas registros sinteticos para clases minoritarias usando `fase1_generacion_sintetica/synthetic_gen.py` con enfoque en las causas que tienen menos de 100 ejemplos
2. **Back-translation**: Traducir descripciones al ingles y de vuelta al espanol para crear variantes
3. **Reducir `min_class_count` a 15**: Si las clases con 30+ registros siguen siendo difíciles
4. **Mezclar datos sinteticos + reales**: Usar los datos de `Consolidado_de_Resultados_Basados_en_IA_ISO_v2.xlsx` junto con `synthetic_training_data.csv` para aumentar el volumen

### 1.4 Pasos para Mejorar Fase 5 - Prioridad (Single-Head)

El modelo de Prioridad ya fue optimizado:

| Parametro | Antes | Despues | Razon |
|-----------|-------|---------|-------|
| `epochs` | 6 | 12 | Mas convergencia |
| `early_stopping_patience` | 2 | 5 | Evita parada prematura |

**Para entrenar:**
```bash
cd fase5_fine_tuning
python train_prioridad_iso.py
```

**Nota importante sobre Prioridad:** Este modelo es diferente porque la etiqueta se genera con una regla determinista basada en `Indicador ABC`, `Parada` y `duracion_horas`. Como es una regla fija, el modelo deberia aprender a replicarla con F1 muy alto (>0.95) si se le dan suficientes epocas y si los features estan bien incluidos en el texto de entrada (ya lo estan).

### 1.5 Orden Recomendado de Entrenamiento

```
1. Fase 3 (Multi-cabeza) → Re-entrenar con pesos corregidos
2. Fase 5 - Causa ISO    → Entrenar modelo independiente  
3. Fase 5 - Prioridad    → Entrenar modelo independiente
4. Fase 4 (Evaluacion)   → Correr inferencia sobre datos reales
```

---

## PARTE 2: Como Juntar los Modelos Independientes para Prediccion

### 2.1 Arquitectura de Prediccion Unificada

Hay **dos estrategias** para combinar los modelos entrenados independientemente:

#### Estrategia A: Pipeline Secuencial (Recomendada)

Cada modelo corre por separado sobre el mismo texto de entrada. Esta es la mejor opcion porque cada modelo single-head esta optimizado para su tarea.

```
CSV de Entrada
    │
    ├──► Modelo Causa ISO  → causa_predicha + confianza_causa
    │    (fase5/causa_iso_model.bin)
    │
    ├──► Modelo Prioridad  → prioridad_predicha + confianza_prioridad
    │    (fase5/prioridad_model.bin)
    │
    └──► (Opcional) Modelo Multi-Cabeza → modo_predicho + confianza_modo
         (fase3/base_iso_model.bin)
    │
    ▼
CSV de Salida con todas las predicciones
```

#### Estrategia B: Modelo Multi-Cabeza Unico (Ya existe en Fase 3)

Usar el modelo de Fase 3 que predice las 3 cosas a la vez. Ventaja: un solo modelo. Desventaja: optimizar una cabeza puede afectar las otras.

### 2.2 Script de Prediccion Unificada

El archivo que une todo seria un script `predict.py` en la raiz del proyecto que:

1. Carga los modelos entrenados (`.bin`) y los label encoders (`.json`)
2. Lee un CSV de entrada
3. Construye el texto de cada registro (igual que en entrenamiento)
4. Pasa el texto por cada modelo
5. Combina las predicciones en un DataFrame
6. Exporta el CSV enriquecido

**Ejemplo de uso:**

```bash
python predict.py --input datos_nuevos.csv --output predicciones.csv
```

**Estructura del CSV de salida:**

| Columnas Originales | causa_predicha | confianza_causa | prioridad_predicha | confianza_prioridad |
|---------------------|----------------|-----------------|---------------------|---------------------|
| (datos del aviso)   | Mechanical wear| 0.87            | 2-Alta              | 0.92                |

### 2.3 Como Cargar los Modelos Independientes

Cada modelo `.bin` contiene:

```python
# Causa ISO (fase5)
checkpoint = torch.load("fase5_fine_tuning/causa_iso_model.bin")
# checkpoint["model_state_dict"]  → pesos del modelo
# checkpoint["config"]            → hiperparametros usados
# checkpoint["label_encoder"]     → lista de clases ["Actuator failure", ...]
# checkpoint["final_metrics"]     → metricas del test set

# Prioridad (fase5)  
checkpoint = torch.load("fase5_fine_tuning/prioridad_model.bin")
# Misma estructura

# Multi-cabeza (fase3)
checkpoint = torch.load("fase3_entrenamiento_bert/base_iso_model.bin")
# checkpoint["label_encoders"]    → dict con 3 listas: modo_falla, causa_falla, prioridad
```

Para reconstruir un modelo Single-Head:
```python
from transformers import AutoModel
import torch.nn as nn

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

# Cargar
checkpoint = torch.load("causa_iso_model.bin", map_location="cpu")
model = SingleHeadBERT("dccuchile/bert-base-spanish-wwm-cased", num_classes=len(checkpoint["label_encoder"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## PARTE 3: Especificacion de la GUI

### 3.1 Vision General

La GUI tendra **dos secciones principales**:

```
┌─────────────────────────────────────────────────────┐
│              OCENSA-ML Dashboard                     │
├──────────────────┬──────────────────────────────────┤
│  📊 Entrenamiento │  🔮 Prediccion                   │
│  (Tab 1)          │  (Tab 2)                         │
└──────────────────┴──────────────────────────────────┘
```

### 3.2 Tab 1: Entrenamiento y Metricas

**Funcionalidad:**
- Ver el estado actual de cada modelo (entrenado o no)
- Ver las metricas de cada modelo (F1, Precision, Recall, Accuracy)
- Boton para lanzar entrenamiento de cada modelo
- Grafica en tiempo real del progreso de entrenamiento (Loss y F1 por epoca)
- Historial de entrenamientos anteriores

**Wireframe:**

```
┌─────────────────────────────────────────────────────────────┐
│  📊 PANEL DE ENTRENAMIENTO                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Estado de Modelos:                                          │
│  ┌────────────────┬──────────┬─────────┬──────────────────┐ │
│  │ Modelo         │ Status   │ F1      │ Ultima Fecha     │ │
│  ├────────────────┼──────────┼─────────┼──────────────────┤ │
│  │ Causa ISO      │ ✅ Listo │ 0.4488  │ 2026-01-31       │ │
│  │ Prioridad      │ ⏳ Pend. │ --      │ --               │ │
│  │ Multi-Cabeza   │ ✅ Listo │ 0.9929* │ 2026-01-29       │ │
│  └────────────────┴──────────┴─────────┴──────────────────┘ │
│  * F1 de Modo; Causa=0.27                                    │
│                                                              │
│  [▶ Entrenar Causa ISO] [▶ Entrenar Prioridad] [▶ Multi]    │
│                                                              │
│  ┌───────────────────────────────────┐                       │
│  │  Grafica: F1 por Epoca            │                       │
│  │  (se actualiza en tiempo real)    │                       │
│  │  ████████████████████             │                       │
│  └───────────────────────────────────┘                       │
│                                                              │
│  Classification Report:                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Clase          │ Precision │ Recall │ F1    │ Support │  │
│  │ Mechanical wear│ 0.76      │ 0.52   │ 0.61  │ 143     │  │
│  │ ...            │ ...       │ ...    │ ...   │ ...     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Tab 2: Prediccion con Nuevos Datos

**Funcionalidad:**
- Cargar un CSV con datos nuevos (avisos de mantenimiento)
- Seleccionar que modelos usar para prediccion
- Ejecutar prediccion
- Ver tabla de resultados con predicciones y confianza
- Descargar CSV enriquecido con las columnas de prediccion

**Wireframe:**

```
┌─────────────────────────────────────────────────────────────┐
│  🔮 PANEL DE PREDICCION                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Modelos disponibles para prediccion:                        │
│  ☑ Causa ISO (F1: 0.XX)                                     │
│  ☑ Prioridad (F1: 0.XX)                                     │
│  ☐ Multi-Cabeza - Modo (F1: 0.99)                           │
│                                                              │
│  Cargar CSV: [📁 Seleccionar archivo...]                     │
│  Archivo: datos_nuevos.csv (2,345 registros)                 │
│                                                              │
│  [🔮 Ejecutar Prediccion]                                    │
│                                                              │
│  Resultados:                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Aviso  │ Descripcion     │ Causa Pred.  │ Conf. │ Pr.│   │
│  │ 236515 │ pase en valvu...│ Seal failure │ 0.87  │ 2  │   │
│  │ 236512 │ falla trasmi... │ Wiring fault │ 0.72  │ 1  │   │
│  │ ...    │ ...             │ ...          │ ...   │ ...│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  [📥 Descargar CSV con Predicciones]                         │
│                                                              │
│  Resumen:                                                    │
│  - Total registros: 2,345                                    │
│  - Confianza promedio causa: 0.78                            │
│  - Confianza promedio prioridad: 0.85                        │
│  - Registros con baja confianza (<0.6): 234 (10%)           │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Tecnologia Recomendada para la GUI

| Opcion | Pros | Contras | Recomendacion |
|--------|------|---------|---------------|
| **Streamlit** | Rapido de implementar, Python puro, buenas graficas | Limitado en customizacion | ⭐ **Recomendado para MVP** |
| **Gradio** | Muy facil para modelos ML, interfaz lista | Menos flexible para dashboards | Bueno para demo rapida |
| **Flask + React** | Maxima flexibilidad, produccion | Mas tiempo de desarrollo | Para version final |

### 3.5 Estructura de Archivos para la GUI (Streamlit)

```
gui/
├── app.py                 # Aplicacion principal Streamlit
├── pages/
│   ├── entrenamiento.py   # Tab de entrenamiento y metricas
│   └── prediccion.py      # Tab de prediccion con CSV
├── utils/
│   ├── model_loader.py    # Carga de modelos .bin
│   ├── predictor.py       # Pipeline de prediccion unificado
│   └── metrics_reader.py  # Leer reportes .md y .txt existentes
└── requirements.txt       # streamlit, torch, transformers, pandas, plotly
```

### 3.6 Flujo de la GUI

```
                    INICIO
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
     Tab Entrenamiento    Tab Prediccion
            │                   │
     Ver metricas        Cargar CSV
     actuales                   │
            │             Verificar que
     Lanzar             los modelos .bin
     entrenamiento       existen
            │                   │
     Monitorear          Ejecutar
     progreso            inferencia
            │                   │
     Guardar             Mostrar tabla
     modelo .bin         de resultados
            │                   │
     Actualizar          Descargar CSV
     metricas            enriquecido
```

---

## PARTE 4: Resumen de Acciones Inmediatas

### Checklist de Ejecucion

- [ ] **Re-entrenar Fase 3** (`train_bert_model.py`) con pesos corregidos → verificar F1 Causa sube
- [ ] **Entrenar Fase 5 Causa ISO** (`train_causa_iso.py`) con hiperparametros optimizados
- [ ] **Entrenar Fase 5 Prioridad** (`train_prioridad_iso.py`) con hiperparametros optimizados
- [ ] **Evaluar** con Fase 4 (`evaluate_real_data.py`) sobre datos reales
- [ ] **Crear `predict.py`** — Script unificado de prediccion que carga los 2-3 modelos
- [ ] **Crear GUI (Streamlit)** — Dashboard de entrenamiento + prediccion
- [ ] **Probar GUI** con datos reales de IW69

### Tiempo Estimado

| Tarea | Tiempo |
|-------|--------|
| Re-entrenar modelos (3 scripts) | 6-12 horas (GPU) / 24-48h (CPU) |
| Crear `predict.py` | 2-3 horas |
| Crear GUI Streamlit basica | 4-6 horas |
| Pruebas de integracion | 2-3 horas |
| **Total** | **~2-3 dias** |

---

## PARTE 5: Notas Tecnicas Importantes

### Consistencia del Texto de Entrada

Cada modelo espera un formato de texto especifico. Es **critico** usar el mismo formato en prediccion que en entrenamiento:

```python
# Causa ISO (fase5) espera:
f"{descripcion}. Problema: {TextoCódProblem}. Modo: {modo_iso_label}. Equipo: {equipo}. Denominacion: {Denominación}."

# Prioridad (fase5) espera:
f"{descripcion}. Problema: {TextoCódProblem}. Parada: {Parada}. ABC: {Indicador ABC}. DuracionHoras: {duracion_horas}."

# Multi-cabeza (fase3) espera:
f"{descripcion}. Equipo: {equipo}. Familia: {familia_equipo}"
```

### Compatibilidad de Modelos

Los modelos de Fase 5 (single-head) usan la clase `SingleHeadBERT` mientras que el de Fase 3 usa `MultiHeadBERT`. El script de prediccion debe saber cual cargar segun el tipo.

### GPU vs CPU

- Entrenamiento: **GPU recomendada** (reduce de ~24h a ~6h por modelo)
- Prediccion/Inferencia: **CPU es suficiente** para lotes de hasta 10,000 registros

---

*Documento generado para el equipo de OCENSA-ML como guia de proximos pasos.*

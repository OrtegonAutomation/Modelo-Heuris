# Guia: Entrenar Modelos OCENSA-ML en Google Colab

Esta guia explica como subir el proyecto a Google Colab, entrenar los modelos con GPU gratuita y descargar los resultados a tu maquina local.

---

## Requisitos Previos

- Cuenta de Google (para acceder a Google Colab y Google Drive)
- Repositorio clonado localmente o acceso al repositorio en GitHub

---

## Paso 1: Subir el Proyecto a Google Drive

### Opcion A — Subir manualmente desde tu maquina

1. Abre [Google Drive](https://drive.google.com)
2. Crea una carpeta llamada `Modelo-Heuris`
3. Sube **toda** la estructura del proyecto manteniendo las carpetas:

```
Modelo-Heuris/
├── data/
│   ├── IW69_limpio.csv
│   ├── Consolidado_de_Resultados_Basados_en_IA_ISO_v2.xlsx
│   └── ... (demas archivos .csv y .xlsx)
├── fase1_generacion_sintetica/
│   └── synthetic_training_data.csv
├── fase2_limpieza_etl/
│   └── IW69_preprocessed.csv
├── fase3_entrenamiento_bert/
│   └── train_bert_model.py
├── fase4_evaluacion_real/
│   └── evaluate_real_data.py
└── fase5_fine_tuning/
    ├── train_causa_iso.py
    └── train_prioridad_iso.py
```

> **Tip:** No necesitas subir los archivos `.bin` (modelos ya entrenados) ni `.gif` si solo quieres re-entrenar. Esto ahorra tiempo de subida.

### Opcion B — Clonar directamente desde GitHub en Colab

Si el repositorio es publico o tienes un token de acceso, puedes clonarlo directamente desde Colab (ver Paso 3).

---

## Paso 2: Crear un Notebook en Google Colab

1. Abre [Google Colab](https://colab.research.google.com)
2. Click en **Archivo > Nuevo cuaderno**
3. **Activar GPU:** Ve a **Entorno de ejecucion > Cambiar tipo de entorno de ejecucion** y selecciona:
   - **Tipo de hardware:** `GPU` (T4 en la version gratuita)
   - Click en **Guardar**

4. Verifica que la GPU esta activa ejecutando en una celda:

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

Deberia mostrar algo como:
```
GPU disponible: True
GPU: Tesla T4
Memoria GPU: 15.8 GB
```

---

## Paso 3: Montar Google Drive y Configurar el Entorno

### 3.1 Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Esto te pedira autorizar el acceso. Acepta y tu Drive quedara en `/content/drive/MyDrive/`.

### 3.2 Navegar al proyecto

Si subiste el proyecto manualmente (Opcion A):

```python
import os
PROJECT_DIR = '/content/drive/MyDrive/Modelo-Heuris'
os.chdir(PROJECT_DIR)
print(f"Directorio actual: {os.getcwd()}")
!ls -la
```

Si prefieres clonar desde GitHub (Opcion B):

```python
# Clonar repositorio (ajusta la URL si es privado)
!git clone https://github.com/OrtegonAutomation/Modelo-Heuris.git /content/Modelo-Heuris

PROJECT_DIR = '/content/Modelo-Heuris'
os.chdir(PROJECT_DIR)
print(f"Directorio actual: {os.getcwd()}")
!ls -la
```

> **Nota sobre Opcion B:** Si clonas directamente a `/content/`, los archivos se pierden al cerrar la sesion de Colab. Para persistirlos, clona dentro de tu Drive:
> ```python
> !git clone https://github.com/OrtegonAutomation/Modelo-Heuris.git /content/drive/MyDrive/Modelo-Heuris
> ```

### 3.3 Instalar dependencias

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers scikit-learn pandas numpy matplotlib seaborn imageio tqdm openpyxl
```

Verifica la instalacion:

```python
import torch
import transformers
import sklearn
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU activa: {torch.cuda.is_available()}")
```

---

## Paso 4: Entrenar los Modelos

### 4.1 Entrenar Fase 3 — Modelo Multi-Cabeza (Modo + Causa + Prioridad)

```python
os.chdir(os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert'))
!python train_bert_model.py
```

**Que genera:**
| Archivo | Descripcion |
|---------|-------------|
| `base_iso_model.bin` (~425 MB) | Modelo multi-cabeza entrenado |
| `label_encoders.json` | Mapeo de etiquetas para las 3 tareas |
| `training_report.md` | Reporte con metricas por epoca |
| `training_cinematic.gif` | Animacion del progreso de entrenamiento |

**Tiempo estimado:** ~30-45 min con GPU T4 (vs ~3-4 horas en CPU).

### 4.2 Entrenar Fase 5 — Causa ISO (Single-Head)

```python
os.chdir(os.path.join(PROJECT_DIR, 'fase5_fine_tuning'))
!python train_causa_iso.py
```

**Que genera:**
| Archivo | Descripcion |
|---------|-------------|
| `causa_iso_model.bin` (~421 MB) | Modelo de causa ISO entrenado |
| `label_encoder_causa_iso.json` | Mapeo de etiquetas de causa ISO |
| `training_report_causa_iso.md` | Reporte detallado de entrenamiento |
| `classification_report_causa_iso.txt` | Precision/recall/F1 por clase |
| `training_causa_iso.gif` | Animacion del entrenamiento |

**Tiempo estimado:** ~20-30 min con GPU T4.

### 4.3 Entrenar Fase 5 — Prioridad (Single-Head)

```python
os.chdir(os.path.join(PROJECT_DIR, 'fase5_fine_tuning'))
!python train_prioridad_iso.py
```

**Que genera:**
| Archivo | Descripcion |
|---------|-------------|
| `prioridad_model.bin` (~421 MB) | Modelo de prioridad entrenado |
| `label_encoder_prioridad.json` | Mapeo de etiquetas de prioridad |
| `training_report_prioridad.md` | Reporte detallado de entrenamiento |

**Tiempo estimado:** ~20-30 min con GPU T4.

### 4.4 Entrenar todos los modelos en secuencia (opcional)

Si quieres ejecutar todo de una vez:

```python
import subprocess, os

scripts = [
    ("fase3_entrenamiento_bert", "train_bert_model.py"),
    ("fase5_fine_tuning",        "train_causa_iso.py"),
    ("fase5_fine_tuning",        "train_prioridad_iso.py"),
]

for folder, script in scripts:
    print(f"\n{'='*60}")
    print(f"  Entrenando: {folder}/{script}")
    print(f"{'='*60}\n")
    script_dir = os.path.join(PROJECT_DIR, folder)
    os.chdir(script_dir)
    result = subprocess.run(["python", script], capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR en {script} (codigo {result.returncode})")
    else:
        print(f"  {script} completado exitosamente")
```

---

## Paso 5: Verificar Resultados en Colab

### 5.1 Leer los reportes de entrenamiento

```python
os.chdir(PROJECT_DIR)

# Reporte Fase 3
with open('fase3_entrenamiento_bert/training_report.md', 'r') as f:
    print(f.read())
```

```python
# Reporte Fase 5 - Causa ISO
with open('fase5_fine_tuning/training_report_causa_iso.md', 'r') as f:
    print(f.read())
```

### 5.2 Verificar que los modelos se guardaron

```python
import os

archivos_esperados = [
    'fase3_entrenamiento_bert/base_iso_model.bin',
    'fase3_entrenamiento_bert/label_encoders.json',
    'fase3_entrenamiento_bert/training_report.md',
    'fase5_fine_tuning/causa_iso_model.bin',
    'fase5_fine_tuning/label_encoder_causa_iso.json',
    'fase5_fine_tuning/training_report_causa_iso.md',
]

print("Archivos generados:")
for archivo in archivos_esperados:
    ruta = os.path.join(PROJECT_DIR, archivo)
    if os.path.exists(ruta):
        size_mb = os.path.getsize(ruta) / (1024 * 1024)
        print(f"  ✅ {archivo} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {archivo} — NO ENCONTRADO")
```

### 5.3 Ver las metricas F1 del modelo entrenado

```python
import torch

# Cargar checkpoint de Fase 3 y mostrar metricas guardadas
checkpoint = torch.load(
    os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert/base_iso_model.bin'),
    map_location='cpu'
)
if 'best_metrics' in checkpoint:
    print("Metricas Fase 3:")
    for k, v in checkpoint['best_metrics'].items():
        print(f"  {k}: {v:.4f}")
```

---

## Paso 6: Descargar Modelos y Resultados a tu Maquina

### Opcion A — Descargar desde Google Drive (recomendado)

Si montaste Google Drive en el Paso 3, los archivos ya estan sincronizados automaticamente en tu Drive. Solo ve a [drive.google.com](https://drive.google.com) y descargalos desde la carpeta `Modelo-Heuris/`.

### Opcion B — Descargar directamente desde Colab

```python
from google.colab import files

# Descargar modelo Fase 3
files.download(os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert/base_iso_model.bin'))
files.download(os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert/label_encoders.json'))
files.download(os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert/training_report.md'))

# Descargar modelo Fase 5 - Causa ISO
files.download(os.path.join(PROJECT_DIR, 'fase5_fine_tuning/causa_iso_model.bin'))
files.download(os.path.join(PROJECT_DIR, 'fase5_fine_tuning/label_encoder_causa_iso.json'))
files.download(os.path.join(PROJECT_DIR, 'fase5_fine_tuning/training_report_causa_iso.md'))
files.download(os.path.join(PROJECT_DIR, 'fase5_fine_tuning/classification_report_causa_iso.txt'))
```

> **Nota:** Los archivos `.bin` son grandes (~421 MB cada uno). La descarga directa puede ser lenta. Se recomienda la Opcion A via Drive.

### Opcion C — Comprimir y descargar como ZIP

Para descargar todo de una vez:

```python
import shutil

# Comprimir modelos y resultados en un ZIP
archivos_a_comprimir = [
    'fase3_entrenamiento_bert/base_iso_model.bin',
    'fase3_entrenamiento_bert/label_encoders.json',
    'fase3_entrenamiento_bert/training_report.md',
    'fase3_entrenamiento_bert/training_cinematic.gif',
    'fase5_fine_tuning/causa_iso_model.bin',
    'fase5_fine_tuning/label_encoder_causa_iso.json',
    'fase5_fine_tuning/training_report_causa_iso.md',
    'fase5_fine_tuning/classification_report_causa_iso.txt',
    'fase5_fine_tuning/training_causa_iso.gif',
]

# Crear carpeta temporal con los resultados
os.makedirs('/tmp/resultados_colab', exist_ok=True)
for archivo in archivos_a_comprimir:
    src = os.path.join(PROJECT_DIR, archivo)
    if os.path.exists(src):
        dst_dir = os.path.join('/tmp/resultados_colab', os.path.dirname(archivo))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(archivo)))

shutil.make_archive('/tmp/resultados_entrenamiento', 'zip', '/tmp/resultados_colab')

from google.colab import files
files.download('/tmp/resultados_entrenamiento.zip')
```

### Opcion D — Push de resultados a GitHub desde Colab

Si clonaste el repositorio en Colab y quieres guardar los modelos con Git LFS:

```python
os.chdir(PROJECT_DIR)

# Configurar Git
!git config user.email "tu-email@example.com"
!git config user.name "Tu Nombre"

# Los archivos .bin ya estan configurados con Git LFS (.gitattributes)
!git add fase3_entrenamiento_bert/base_iso_model.bin
!git add fase3_entrenamiento_bert/label_encoders.json
!git add fase3_entrenamiento_bert/training_report.md
!git add fase5_fine_tuning/causa_iso_model.bin
!git add fase5_fine_tuning/label_encoder_causa_iso.json
!git add fase5_fine_tuning/training_report_causa_iso.md

!git commit -m "Modelos re-entrenados en Google Colab con GPU T4"

# Push (necesitaras un token de acceso personal)
# !git push https://<TOKEN>@github.com/OrtegonAutomation/Modelo-Heuris.git main
```

---

## Paso 7: Verificar Modelos Localmente

Una vez descargados los archivos, colocalos en las carpetas correspondientes de tu proyecto local y ejecuta la evaluacion con datos reales:

```bash
cd fase4_evaluacion_real
python evaluate_real_data.py
```

Esto cargara los modelos entrenados y generara predicciones sobre el dataset real (`IW69_preprocessed.csv`), permitiendote verificar que los modelos funcionan correctamente en tu maquina.

---

## Tips y Solucion de Problemas

### La sesion de Colab se desconecta

- Colab gratuito desconecta sesiones inactivas (~90 min) o largas (~12 horas)
- **Solucion:** Trabaja con Google Drive montado (Opcion A) para no perder archivos
- Puedes reconectar y continuar donde te quedaste

### Se acaba la memoria GPU (CUDA out of memory)

Si ves errores como `RuntimeError: CUDA out of memory`, reduce el batch size editando la configuracion del script antes de ejecutar:

```python
# Antes de ejecutar el script, modifica el CONFIG
# En train_bert_model.py linea ~202 o en train_causa_iso.py linea ~69
# Cambiar batch_size de 16 a 8:

import json
# Ejemplo: modificar batch size para Fase 3
config_path = os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert/train_bert_model.py')
with open(config_path, 'r') as f:
    content = f.read()
content = content.replace('"batch_size": 16', '"batch_size": 8')
with open(config_path, 'w') as f:
    f.write(content)
```

### Los scripts no encuentran los datos

Verifica que las rutas relativas esten correctas. Los scripts usan rutas relativas como `../data/` y `../fase1_generacion_sintetica/`. Asegurate de ejecutar cada script desde su propia carpeta:

```python
# Correcto
os.chdir(os.path.join(PROJECT_DIR, 'fase3_entrenamiento_bert'))
!python train_bert_model.py

# Incorrecto (las rutas relativas fallaran)
# !python fase3_entrenamiento_bert/train_bert_model.py
```

### Quiero usar Colab Pro para GPU mas rapida

Con Colab Pro/Pro+ obtienes acceso a GPUs mas potentes (A100, V100). El proceso es identico; solo cambia la GPU en **Entorno de ejecucion > Cambiar tipo de entorno de ejecucion**.

| GPU | Memoria | Velocidad Estimada |
|-----|---------|-------------------|
| T4 (gratuito) | 15 GB | ~30-45 min por modelo |
| V100 (Pro) | 16 GB | ~15-25 min por modelo |
| A100 (Pro+) | 40 GB | ~10-15 min por modelo |

---

## Resumen de Archivos Clave

| Archivo | Ubicacion | Descripcion |
|---------|-----------|-------------|
| `base_iso_model.bin` | `fase3_entrenamiento_bert/` | Modelo multi-cabeza (Modo + Causa + Prioridad) |
| `label_encoders.json` | `fase3_entrenamiento_bert/` | Etiquetas de las 3 clasificaciones |
| `training_report.md` | `fase3_entrenamiento_bert/` | Metricas de entrenamiento Fase 3 |
| `causa_iso_model.bin` | `fase5_fine_tuning/` | Modelo single-head de Causa ISO |
| `label_encoder_causa_iso.json` | `fase5_fine_tuning/` | Etiquetas de causa ISO |
| `training_report_causa_iso.md` | `fase5_fine_tuning/` | Metricas de entrenamiento Causa ISO |
| `prioridad_model.bin` | `fase5_fine_tuning/` | Modelo single-head de Prioridad |
| `label_encoder_prioridad.json` | `fase5_fine_tuning/` | Etiquetas de prioridad |
| `training_report_prioridad.md` | `fase5_fine_tuning/` | Metricas de entrenamiento Prioridad |

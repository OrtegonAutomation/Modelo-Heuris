"""
FASE 2: PREPROCESAMIENTO ETL DEL TESTER REAL (IW69_limpio.csv)
===============================================================
Sistema OCENSA-ML - Limpieza Técnica según Sección 3.1 del README

Este script aplica las siguientes transformaciones:
1. Lowercasing: Convertir todo a minúsculas
2. Punctuation Stripping: Eliminar caracteres especiales, manteniendo puntos y comas
3. Identificación de TAGS: Preservar identificadores de activos (P-101, V-05, etc.)
4. Normalización de texto industrial
5. NO tocar las etiquetas originales (causa_falla, Código averia, etc.)

Columnas clave identificadas:
- equipo: Identificador numérico del equipo SAP
- descripcion: Texto libre con información de la falla
- Denominación: Descripción del sistema/ubicación
- Denominación.1: Descripción específica del activo
"""

import pandas as pd
import re
import unicodedata
import os
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS ---
INPUT_FILE = "../data/IW69_limpio.csv"
OUTPUT_FILE = "IW69_preprocessed.csv"
REPORT_FILE = "preprocessing_report.md"

# --- DICCIONARIO DE ABREVIATURAS INDUSTRIALES ---
INDUSTRIAL_ABBREVIATIONS = {
    # Equipos y sistemas
    "mov": "motor operated valve",
    "psv": "pressure safety valve", 
    "pcv": "pressure control valve",
    "lcv": "level control valve",
    "fcv": "flow control valve",
    "hv": "hand valve",
    "esdv": "emergency shutdown valve",
    "ccm": "centro control motores",
    "vfd": "variable frequency drive",
    "ups": "uninterruptible power supply",
    "bpc": "bomba principal crudo",
    "sci": "sistema contra incendio",
    "ptar": "planta tratamiento aguas residuales",
    "api": "american petroleum institute",
    "dcs": "distributed control system",
    "plc": "programmable logic controller",
    "hmi": "human machine interface",
    
    # Medidas y unidades
    "psig": "pounds per square inch gauge",
    "kva": "kilovolt amperes",
    "kv": "kilovoltios",
    "mw": "megawatts",
    
    # Estaciones
    "epo": "estacion porvenir",
    "cva": "estacion coveñas",
    "vsa": "estacion vasconia",
    "cus": "estacion cusiana",
    "pae": "estacion paez",
    "cqo": "estacion chiquillo",
    "gra": "estacion granjita",
    "cca": "caucacia",
    "cup": "cupiagua",
    "mrf": "miraflores",
    "lbe": "la belleza",
}

# --- PATRONES DE TAGS DE EQUIPOS ---
# Estos patrones identifican equipos que deben preservarse
EQUIPMENT_TAG_PATTERNS = [
    r'\b[A-Z]{1,4}[-_]?\d{3,6}[A-Z]?\b',  # MOV-370112, TK12020, BC-52100
    r'\b[A-Z]{2,3}\d{4,5}\b',              # TC52080, TP302
    r'\bGE[-_]?\d{2,5}\b',                 # GE-010, GE22130
    r'\bTG[-_]?\d{4,5}\b',                 # TG83000, TG-82000
    r'\bTR[-_]?\d{3}\b',                   # TR-301
    r'\bBA[-_]?\d{3}\b',                   # BA-304, BA310
    r'\bBC[-_]?\d{4,5}\b',                 # BC-52100
    r'\bBT[-_]?\d{4,5}\b',                 # BT-20010
    r'\bBU[-_]?\d{4,5}\b',                 # BU-53020
    r'\bTP[-_]?\d{3}\b',                   # TP-302, TP304
    r'\bTC[-_]?\d{3,5}\b',                 # TC-310, TC52080
    r'\bTK[-_]?\d{4,5}\b',                 # TK-12020, TK7312
    r'\bPSV[-_]?\d{3,4}[A-Z]?\b',          # PSV-2307A, PSV2309
    r'\bPCV[-_]?\d{5,6}\b',                # PCV-300001
    r'\bLCV[-_]?\d{5,6}\b',                # LCV300801
    r'\bFCV[-_]?\d{5,6}\b',                # FCV-370607
    r'\bMOV[-_]?\d{5,6}\b',                # MOV-370112
    r'\bESDV[-_]?\d{5,6}\b',               # ESDV-820024
    r'\bPI[-_]?\d{4,5}\b',                 # PI-3661
    r'\bPIT[-_]?\d{5,6}\b',                # PIT-820013
    r'\bLIT[-_]?\d{5,6}\b',                # LIT-520212B
    r'\bFQI[-_]?\d{5,6}\b',                # FQI-300102
    r'\bFQE[-_]?\d{4,5}\b',                # FQE-36050
    r'\bTFL[-_]?\d{5,6}\b',                # TFL-3660
    r'\bPDI[-_]?\d{4}\b',                  # PDI-3010
    r'\bDV[-_]?\d{6}\b',                   # DV-572001
    r'\bHMA[-_]?\d{2}\b',                  # HMA-40, HMA-28
    r'\bAC[-_]?\d{3}[-_]?\d{5}[A-Z]?\b',   # AC920-50600C
    r'\bMS[-_]?\d{4,5}\b',                 # MS-3600, MS36000
    r'\bBE[-_]?\d{4,5}\b',                 # BE3610, BE-36040
    r'\bBS[-_]?\d{4,5}\b',                 # BS37010, BS38060
    r'\bPR[-_]?\d{4,5}\b',                 # PR3010, PR38100
    r'\bCA[-_]?\d{4,5}\b',                 # CA38000, CA36000
    r'\bCO[-_]?\d{4,5}\b',                 # CO-54010
    r'\bSWG[-_]?\d{5}\b',                  # SWG-20300, SWG20600
]


def extract_and_preserve_tags(text):
    """
    Extrae y preserva TAGs de equipos del texto antes de la normalización.
    Retorna el texto original y un diccionario de TAGs encontrados.
    """
    if pd.isna(text):
        return "", []
    
    text_upper = str(text).upper()
    found_tags = []
    
    for pattern in EQUIPMENT_TAG_PATTERNS:
        matches = re.findall(pattern, text_upper)
        found_tags.extend(matches)
    
    return text, list(set(found_tags))


def normalize_text(text):
    """
    Aplica normalización de texto industrial según Sección 3.1 del README.
    
    1. Lowercasing
    2. Mantener puntos y comas (separadores de síntomas)
    3. Eliminar caracteres especiales no relevantes
    4. Preservar TAGs de equipos
    5. Normalizar espacios en blanco
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    text = str(text)
    
    # 1. Preservar TAGs antes de lowercasing (los convertiremos de vuelta)
    original_text, found_tags = extract_and_preserve_tags(text)
    
    # 2. Convertir a minúsculas
    text = text.lower()
    
    # 3. Normalizar caracteres Unicode (acentos, etc.)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # 4. Mantener caracteres alfanuméricos, puntos, comas, guiones y espacios
    # Los guiones son importantes para TAGs como MOV-370112
    text = re.sub(r'[^a-z0-9\s\.\,\-\_\/\#]', ' ', text)
    
    # 5. Normalizar múltiples espacios a uno solo
    text = re.sub(r'\s+', ' ', text)
    
    # 6. Eliminar espacios al inicio y final
    text = text.strip()
    
    # 7. Restaurar TAGs en mayúsculas (importante para el modelo)
    for tag in found_tags:
        tag_lower = tag.lower()
        # Reemplazar versiones en minúsculas con la versión original
        text = re.sub(rf'\b{re.escape(tag_lower)}\b', tag, text, flags=re.IGNORECASE)
    
    return text


def identify_equipment_family(denominacion, descripcion):
    """
    Identifica la familia de equipo basándose en la denominación y descripción.
    Esto ayuda a mapear con las categorías ISO 14224.
    """
    if pd.isna(denominacion):
        denominacion = ""
    if pd.isna(descripcion):
        descripcion = ""
    
    text = (str(denominacion) + " " + str(descripcion)).lower()
    
    # Mapeo de familias según ISO 14224
    families = {
        "BOMBA": ["bomba", "pump", "centrifuga", "sumergible", "dosifica", "jockey", "hidroflo"],
        "VALVULA": ["valvula", "valve", "mov", "psv", "pcv", "lcv", "fcv", "hv", "esdv", "actuador"],
        "MOTOR": ["motor", "generador", "turbina", "vfd", "arrancador", "excitador"],
        "COMPRESOR": ["compresor", "compressor", "soplador", "ventilador"],
        "INSTRUMENTO": ["transmisor", "medidor", "sensor", "indicador", "switch", "radar", "detector", "analizador"],
        "TANQUE": ["tanque", "tank", "recipiente", "separador", "sumidero"],
        "TUBERIA": ["tuberia", "linea", "ducto", "cabezal", "multiple"],
        "ELECTRICO": ["transformador", "tablero", "ccm", "ups", "interruptor", "celda", "banco"],
        "SISTEMA_CONTROL": ["plc", "dcs", "hmi", "control", "comunicacion", "rack"],
        "CONTRAINCENDIO": ["sci", "fire", "hidrante", "espuma", "extincion", "diluvio"],
    }
    
    for family, keywords in families.items():
        if any(kw in text for kw in keywords):
            return family
    
    return "OTRO"


def extract_failure_indicators(descripcion):
    """
    Extrae indicadores de falla del texto de descripción.
    Estos son síntomas clave que el modelo debe aprender a reconocer.
    """
    if pd.isna(descripcion):
        return []
    
    text = str(descripcion).lower()
    
    # Indicadores comunes de falla
    indicators = []
    
    # Fugas y escapes
    if any(word in text for word in ["fuga", "goteo", "escape", "perdida", "humedec"]):
        indicators.append("FUGA")
    
    # Fallas eléctricas y de control
    if any(word in text for word in ["falla", "error", "no funciona", "no arranca", "no cierra", "no abre"]):
        indicators.append("FALLA_FUNCION")
    
    # Problemas mecánicos
    if any(word in text for word in ["vibracion", "ruido", "desgaste", "rotura", "ruptura", "atascamiento"]):
        indicators.append("MECANICO")
    
    # Problemas térmicos
    if any(word in text for word in ["temperatura", "caliente", "calor", "sobrecalentamiento"]):
        indicators.append("TERMICO")
    
    # Problemas de presión
    if any(word in text for word in ["presion", "diferencial", "alta presion", "baja presion"]):
        indicators.append("PRESION")
    
    # Obstrucciones
    if any(word in text for word in ["obstruccion", "taponamiento", "bloqueo", "atascado"]):
        indicators.append("OBSTRUCCION")
    
    # Comunicación/señal
    if any(word in text for word in ["comunicacion", "señal", "alarma", "false", "erratic"]):
        indicators.append("SEÑAL")
    
    # Corrosión/daño
    if any(word in text for word in ["corrosion", "daño", "deterioro", "grieta", "fisura"]):
        indicators.append("CORROSION_DAÑO")
    
    return indicators


def generate_preprocessing_report(df_original, df_processed, output_path):
    """
    Genera un reporte detallado del preprocesamiento realizado.
    """
    report = f"""# REPORTE DE PREPROCESAMIENTO - FASE 2
## Sistema OCENSA-ML - ETL del Tester Real

**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. RESUMEN GENERAL

| Métrica | Valor |
|---------|-------|
| Registros originales | {len(df_original):,} |
| Registros procesados | {len(df_processed):,} |
| Columnas originales | {len(df_original.columns)} |
| Columnas finales | {len(df_processed.columns)} |

---

## 2. COLUMNAS CLAVE IDENTIFICADAS

### Campos de Texto (para el modelo NLP)
- **descripcion**: Texto libre con síntomas y problemas
- **descripcion_normalizada**: Texto preprocesado para el modelo
- **Denominación**: Sistema/ubicación del equipo
- **Denominación.1**: Descripción específica del activo

### Campos de Identificación
- **equipo**: Código SAP del equipo
- **id_aviso**: Identificador único del aviso
- **Ubicac.técnica**: Ubicación técnica estructurada

### Campos de Clasificación Existentes (NO MODIFICADOS)
- **Código averia**: Código de tipo de falla
- **TextoCódProblem**: Descripción del problema
- **causa_falla**: Causa de falla registrada

---

## 3. DISTRIBUCIÓN DE FAMILIAS DE EQUIPO

"""
    # Distribución de familias
    if 'familia_equipo_detectada' in df_processed.columns:
        family_dist = df_processed['familia_equipo_detectada'].value_counts()
        for family, count in family_dist.items():
            pct = (count / len(df_processed)) * 100
            report += f"| {family} | {count:,} | {pct:.1f}% |\n"
    
    report += f"""
---

## 4. INDICADORES DE FALLA DETECTADOS

"""
    # Conteo de indicadores
    if 'indicadores_falla' in df_processed.columns:
        all_indicators = []
        for indicators in df_processed['indicadores_falla']:
            if isinstance(indicators, str):
                all_indicators.extend(indicators.split('|'))
        
        if all_indicators:
            from collections import Counter
            indicator_counts = Counter(all_indicators)
            for indicator, count in indicator_counts.most_common(10):
                if indicator:
                    report += f"- **{indicator}**: {count:,} ocurrencias\n"
    
    report += f"""
---

## 5. CALIDAD DE DATOS

### Campos con valores faltantes
"""
    # Análisis de valores faltantes
    missing = df_processed.isnull().sum()
    missing_pct = (missing / len(df_processed)) * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            report += f"- {col}: {pct:.1f}% faltantes\n"
    
    report += f"""
---

## 6. TRANSFORMACIONES APLICADAS

1. ✅ **Lowercasing**: Texto convertido a minúsculas
2. ✅ **Normalización Unicode**: Eliminación de acentos y caracteres especiales
3. ✅ **Preservación de TAGs**: Identificadores de equipos mantenidos
4. ✅ **Identificación de familias**: Categorización automática de equipos
5. ✅ **Extracción de indicadores**: Síntomas clave identificados
6. ✅ **Campos originales preservados**: Etiquetas existentes no modificadas

---

## 7. ARCHIVOS GENERADOS

- `IW69_preprocessed.csv`: Dataset preprocesado listo para evaluación
- `preprocessing_report.md`: Este reporte

---

*Generado automáticamente por el sistema OCENSA-ML - Fase 2 ETL*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    """
    Función principal del pipeline de preprocesamiento.
    """
    print("=" * 60)
    print("FASE 2: PREPROCESAMIENTO ETL - SISTEMA OCENSA-ML")
    print("=" * 60)
    
    # 1. Cargar datos originales
    print("\n[1/6] Cargando datos originales...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    print(f"   -> Registros cargados: {len(df):,}")
    print(f"   -> Columnas: {len(df.columns)}")
    
    # Guardar copia original para el reporte
    df_original = df.copy()
    
    # 2. Identificar columnas clave
    print("\n[2/6] Identificando columnas clave...")
    key_columns = {
        'descripcion': 'descripcion',
        'equipo': 'equipo',
        'denominacion': 'Denominación',
        'denominacion_activo': 'Denominación.1',
        'ubicacion_tecnica': 'Ubicac.técnica',
        'codigo_averia': 'Código averia',
        'texto_problema': 'TextoCódProblem',
        'causa_falla': 'causa_falla',
        'id_aviso': 'id_aviso',
    }
    
    for key, col in key_columns.items():
        if col in df.columns:
            print(f"   [OK] {key}: '{col}' encontrada")
        else:
            print(f"   [X] {key}: '{col}' NO encontrada")
    
    # 3. Aplicar normalización de texto
    print("\n[3/6] Aplicando normalización de texto...")
    
    # Normalizar descripción
    if 'descripcion' in df.columns:
        df['descripcion_normalizada'] = df['descripcion'].apply(normalize_text)
        print(f"   -> Campo 'descripcion_normalizada' creado")
    
    # Normalizar denominaciones
    if 'Denominación' in df.columns:
        df['denominacion_normalizada'] = df['Denominación'].apply(normalize_text)
    if 'Denominación.1' in df.columns:
        df['denominacion_activo_normalizada'] = df['Denominación.1'].apply(normalize_text)
    
    # 4. Identificar familias de equipo
    print("\n[4/6] Identificando familias de equipo...")
    df['familia_equipo_detectada'] = df.apply(
        lambda row: identify_equipment_family(
            row.get('Denominación.1', ''),
            row.get('descripcion', '')
        ),
        axis=1
    )
    
    family_dist = df['familia_equipo_detectada'].value_counts()
    for family, count in family_dist.head(5).items():
        print(f"   -> {family}: {count:,} registros")
    
    # 5. Extraer indicadores de falla
    print("\n[5/6] Extrayendo indicadores de falla...")
    df['indicadores_falla'] = df['descripcion'].apply(
        lambda x: '|'.join(extract_failure_indicators(x))
    )
    
    # Conteo de indicadores más comunes
    indicator_stats = {}
    for indicators in df['indicadores_falla']:
        if isinstance(indicators, str):
            for ind in indicators.split('|'):
                if ind:
                    indicator_stats[ind] = indicator_stats.get(ind, 0) + 1
    
    print("   Indicadores más frecuentes:")
    for ind, count in sorted(indicator_stats.items(), key=lambda x: -x[1])[:5]:
        print(f"   -> {ind}: {count:,} ocurrencias")
    
    # 6. Guardar resultados
    print("\n[6/6] Guardando resultados...")
    
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"   -> Dataset guardado: {OUTPUT_FILE}")
    
    # Generar reporte
    report_path = os.path.join(script_dir, REPORT_FILE)
    generate_preprocessing_report(df_original, df, report_path)
    print(f"   -> Reporte guardado: {REPORT_FILE}")
    
    print("\n" + "=" * 60)
    print("FASE 2 COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print(f"\nResultados:")
    print(f"  - Registros procesados: {len(df):,}")
    print(f"  - Familias identificadas: {len(family_dist)}")
    print(f"  - Columnas totales: {len(df.columns)}")
    print(f"\nArchivos generados en: {script_dir}")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {REPORT_FILE}")
    
    return df


if __name__ == "__main__":
    df_processed = main()

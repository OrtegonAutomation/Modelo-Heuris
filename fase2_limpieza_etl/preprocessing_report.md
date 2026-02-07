# REPORTE DE PREPROCESAMIENTO - FASE 2
## Sistema OCENSA-ML - ETL del Tester Real


---

## 1. RESUMEN GENERAL

| Métrica | Valor |
|---------|-------|
| Registros originales | 10,120 |
| Registros procesados | 10,120 |
| Columnas originales | 34 |
| Columnas finales | 39 |

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

| VALVULA | 2,723 | 26.9% |
| BOMBA | 1,594 | 15.8% |
| INSTRUMENTO | 1,574 | 15.6% |
| OTRO | 1,495 | 14.8% |
| MOTOR | 1,119 | 11.1% |
| TANQUE | 441 | 4.4% |
| ELECTRICO | 408 | 4.0% |
| COMPRESOR | 225 | 2.2% |
| TUBERIA | 211 | 2.1% |
| SISTEMA_CONTROL | 176 | 1.7% |
| CONTRAINCENDIO | 154 | 1.5% |

---

## 4. INDICADORES DE FALLA DETECTADOS

- **FALLA_FUNCION**: 4,513 ocurrencias
- **FUGA**: 1,711 ocurrencias
- **SEÑAL**: 497 ocurrencias
- **MECANICO**: 379 ocurrencias
- **PRESION**: 233 ocurrencias
- **CORROSION_DAÑO**: 213 ocurrencias
- **TERMICO**: 105 ocurrencias
- **OBSTRUCCION**: 67 ocurrencias

---

## 5. CALIDAD DE DATOS

### Campos con valores faltantes
- Centro planif.: 0.0% faltantes
- Activo fijo: 65.3% faltantes
- Orden: 7.1% faltantes
- TextoGrpPartObj: 0.4% faltantes
- TextoCódPartObj: 0.0% faltantes
- Causas avería: 72.9% faltantes
- Txt. cód. mot.: 72.9% faltantes
- causa_falla: 73.2% faltantes
- fecha_fin: 32.0% faltantes
- Parada: 29.7% faltantes
- Indicador ABC: 12.4% faltantes
- fecha_fin_dt: 32.0% faltantes
- duracion_horas: 10.9% faltantes

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
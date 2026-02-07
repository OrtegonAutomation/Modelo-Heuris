# REPORTE DE ENTRENAMIENTO - FASE 3
## Sistema OCENSA-ML - Modelo Base ISO 14224

**Fecha de entrenamiento:** 2026-01-29 12:00:33  
**Tiempo total:** 1522.67 minutos

---

## 1. CONFIGURACION DEL MODELO

| Parametro | Valor |
|-----------|-------|
| Modelo Base | dccuchile/bert-base-spanish-wwm-cased |
| Max Length | 256 |
| Batch Size | 16 |
| Epochs | 10 |
| Learning Rate | 3e-05 |
| Weight Decay | 0.01 |

---

## 2. ARQUITECTURA MULTI-CABEZA

| Cabeza | Clases | Descripcion |
|--------|--------|-------------|
| Modo de Falla | 32 | ISO 14224 Failure Modes |
| Causa de Falla | 45 | ISO 14224 Failure Causes |
| Prioridad | 4 | Niveles de Urgencia |

---

## 3. METRICAS FINALES (Test Set)

### 3.1 Modo de Falla
| Metrica | Valor |
|---------|-------|
| **F1-Score** | 0.9929 |
| Accuracy | 0.9930 |
| Precision | 0.9935 |
| Recall | 0.9930 |

### 3.2 Causa de Falla
| Metrica | Valor |
|---------|-------|
| **F1-Score** | 0.2758 |
| Accuracy | 0.2895 |
| Precision | 0.3333 |
| Recall | 0.2895 |

### 3.3 Prioridad
| Metrica | Valor |
|---------|-------|
| **F1-Score** | 0.4826 |
| Accuracy | 0.4975 |
| Precision | 0.4898 |
| Recall | 0.4975 |

---

## 4. OBJETIVO DE FASE 3

| Objetivo | Requerido | Alcanzado | Status |
|----------|-----------|-----------|--------|
| F1-Score Modo de Falla | > 0.95 | 0.9929 | CUMPLIDO |
| F1-Score Causa de Falla | > 0.95 | 0.2758 | PENDIENTE |

---

## 5. HISTORIAL DE ENTRENAMIENTO

| Epoca | Loss Train | F1 Modo (Val) | F1 Causa (Val) | F1 Prioridad (Val) |
|-------|------------|---------------|----------------|---------------------|
| 1 | 2.6898 | 0.9816 | 0.0640 | 0.4852 |
| 2 | 1.0585 | 0.9940 | 0.0968 | 0.5062 |
| 3 | 1.0127 | 0.9922 | 0.1014 | 0.5003 |
| 4 | 1.0055 | 0.9917 | 0.1398 | 0.5089 |
| 5 | 0.9854 | 0.9917 | 0.2187 | 0.5090 |
| 6 | 0.9594 | 0.9949 | 0.2555 | 0.5031 |
| 7 | 0.9428 | 0.9922 | 0.2683 | 0.4980 |
| 8 | 0.9325 | 0.9935 | 0.2718 | 0.4976 |
| 9 | 0.9241 | 0.9935 | 0.2766 | 0.5137 |
| 10 | 0.9182 | 0.9934 | 0.2793 | 0.5107 |

---

## 6. CLASES DE MODO DE FALLA

1. Blockage (BLO)
2. Calibration error (CAL)
3. Cavitation (CAV)
4. Contamination (CNT)
5. Control failure (CON)
6. Corrosion (COR)
7. Deformation (DEF)
8. Erratic reading (ERR)
9. External leakage - process (ELP)
10. External leakage - utility (ELU)
11. Fail to start (FTS)
12. Fail to stop (STP)
13. False alarm (FAL)
14. Fracture (FRA)
15. High output (HIO)
16. Insulation failure (INS)
17. Internal leakage (INL)
18. Loose part (LOO)
19. Low output (LOW)
20. Mechanical failure (MEC)
21. No output (NONE)
22. Other (OTH)
23. Overheating (HIW)
24. Parameter deviation (DEV)
25. Plugged (PLU)
26. Seizure (SEI)
27. Short circuit (SHT)
28. Signal failure (SIG)
29. Spontaneous shutdown (SHU)
30. Structural deficiency (STR)
31. Vibration (VIB)
32. Wear (WEA)

---

## 7. CLASES DE CAUSA DE FALLA

1. Actuator failure
2. Bearing failure
3. Breaker trip
4. Calibration drift
5. Cavitation erosion
6. Connection failure
7. Corrosion
8. Design error
9. Electrical surge
10. End of life
11. Erosion
12. External environment
13. Fatiga
14. Fluid contamination
15. Foreign object damage
16. Fouling
17. Fuse blown
18. Gasket failure
19. Hydrogen embrittlement
20. Impeller damage
21. Inadequate procedure
22. Installation error
23. Looseness
24. Lubrication failure
25. Maintenance error
26. Manufacturing defect
27. Material defect
28. Mechanical wear
29. Microbial corrosion
30. Operator error
31. Overload
32. Packing wear
33. Power loss
34. Pressure extreme
35. Process change
36. Scale buildup
37. Seal failure
38. Shaft misalignment
39. Software bug
40. Solenoid failure
41. Spare part quality
42. Temperature extreme
43. Tool damage
44. Vandalism
45. Wiring fault

---

## 8. ARCHIVOS GENERADOS

- `base_iso_model.bin`: Modelo entrenado
- `label_encoders.json`: Mapeo de etiquetas
- `training_report.md`: Este reporte

---

*Generado automaticamente por Sistema OCENSA-ML - Fase 3*

# REPORTE ENTRENAMIENTO CAUSA ISO
Fecha: 2026-01-31 00:01:51
Tiempo total: 340.70 minutos

## Configuracion
- Modelo: dccuchile/bert-base-spanish-wwm-cased
- Max length: 256
- Epochs: 8
- Batch size: 16
- Learning rate: 3e-05
- Min class count: 100
- Drop Other: True

## Metricas finales (Test)
- Accuracy: 0.5102
- F1 Macro: 0.4488
- F1 Weighted: 0.5124
- Precision Macro: 0.4412
- Recall Macro: 0.4663

## Historial por epoca
| Epoca | Loss Train | F1 Macro (Val) | F1 Weighted (Val) |
|------:|-----------:|---------------:|------------------:|
| 1 | 1.4684 | 0.3793 | 0.4654 |
| 2 | 0.9080 | 0.3895 | 0.4584 |
| 3 | 0.7025 | 0.4209 | 0.4897 |
| 4 | 0.5242 | 0.4381 | 0.5032 |
| 5 | 0.3931 | 0.4407 | 0.4973 |
| 6 | 0.2910 | 0.4304 | 0.4979 |
| 7 | 0.2326 | 0.4110 | 0.4781 |

## Clases
1. Actuator failure
2. Calibration drift
3. Connection failure
4. Corrosion
5. Fouling
6. Gasket failure
7. Mechanical wear
8. Packing wear
9. Seal failure
10. Software bug
11. Wiring fault

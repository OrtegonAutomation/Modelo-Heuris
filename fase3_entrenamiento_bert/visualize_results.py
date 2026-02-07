"""
VISUALIZADOR DE ENTRENAMIENTO - SISTEMA OCENSA-ML
=================================================
Este script genera graficas de alto impacto visual a partir del modelo entrenado.
Se debe ejecutar DESPUES de que train_bert_model.py termine.

Graficas generadas:
1. training_loss_evolution.png (Curva de aprendizaje)
2. f1_score_comparison.png (Rendimiento por cabeza)
3. combined_metrics_dashboard.png (Resumen ejecutivo)
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# Estilo "Cyberpunk" / Premium
plt.style.use('dark_background')
sns.set_palette("husl")

def load_history(model_path):
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra {model_path}")
        return None
        
    print(f"Cargando historial desde {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # El script de entrenamiento original guardaba 'metrics_history' en una variable local
    # pero NO en el checkpoint diccionario explícitamente en la versión anterior.
    # REVISIÓN: En la última versión del script, ¿guardé metrics_history en el .bin?
    # Revisando el código... 
    # torch.save({ ..., 'model_state_dict': ..., 'final_metrics': ... })
    # ¡UPS! No guardé el historial completo en el diccionario, solo las métricas finales.
    
    # PERO, el reporte training_report.md TIENE los datos en una tabla Markdown.
    # Vamos a hacer un parser inteligente que lea el markdown si el bin no tiene la info.
    
    return parse_markdown_report("training_report.md")

def parse_markdown_report(report_path):
    if not os.path.exists(report_path):
        print("No se encuentra el reporte de entrenamiento.")
        return None
        
    print("Extrayendo datos de training_report.md...")
    data = []
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    parsing_table = False
    for line in lines:
        if "| Epoca | Loss Train |" in line:
            parsing_table = True
            continue
        if parsing_table and line.strip().startswith("|") and "---" not in line:
            # Ejemplo: | 1 | 3.1885 | 0.0052 | 0.0044 | 0.1223 |
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5:
                try:
                    data.append({
                        "Epoca": int(parts[0]),
                        "Loss": float(parts[1]),
                        "F1_Modo": float(parts[2]),
                        "F1_Causa": float(parts[3]),
                        "F1_Prio": float(parts[4])
                    })
                except ValueError:
                    pass
        if parsing_table and line.strip() == "":
            parsing_table = False
            
    return pd.DataFrame(data)

def plot_curves(df):
    if df is None or df.empty:
        print("No hay datos para graficar.")
        return

    # 1. Grafica de LOSS (Gradiente Azul)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoca'], df['Loss'], marker='o', linewidth=3, color='#00f2ff', label='Training Loss')
    plt.fill_between(df['Epoca'], df['Loss'], alpha=0.3, color='#00f2ff')
    plt.title('Evolucion de la Funcion de Perdida (Loss)', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend()
    plt.savefig('training_loss_evolution.png', dpi=300, bbox_inches='tight')
    print("Generado: training_loss_evolution.png")
    
    # 2. Grafica de F1 Score (Comparativa Multi-Cabeza)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Epoca'], df['F1_Modo'], marker='s', linewidth=2, color='#ff00ff', label='Modo de Falla')
    plt.plot(df['Epoca'], df['F1_Causa'], marker='^', linewidth=2, color='#ffff00', label='Causa de Falla')
    plt.plot(df['Epoca'], df['F1_Prio'], marker='D', linewidth=2, color='#00ff00', label='Prioridad')
    
    # Linea objetivo 0.95
    plt.axhline(y=0.95, color='white', linestyle='--', alpha=0.5, label='Objetivo (0.95)')
    
    plt.title('Evolucion del Aprendizaje (F1-Score)', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('F1 Score (Weighted)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig('f1_score_evolution.png', dpi=300, bbox_inches='tight')
    print("Generado: f1_score_evolution.png")

if __name__ == "__main__":
    df = load_history("base_iso_model.bin")
    plot_curves(df)
    

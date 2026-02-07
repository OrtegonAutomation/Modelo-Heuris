"""
Normaliza salidas LLM a ISO 14224 con reglas refinadas.
Genera:
- Consolidado_de_Resultados_Basados_en_IA_ISO_v2.xlsx
- iso_mapping_report.md
- iso_modo_oth_top.csv
- iso_causa_other_top.csv
"""

import os
import re
import unicodedata
import pandas as pd

INPUT_XLSX = "../data/Consolidado_de_Resultados_Basados_en_IA.xlsx"
SHEET = "Consolidado Base"

OUTPUT_XLSX = "../data/Consolidado_de_Resultados_Basados_en_IA_ISO_v2.xlsx"
REPORT_MD = "../data/iso_mapping_report.md"
MODO_OTH_CSV = "../data/iso_modo_oth_top.csv"
CAUSA_OTHER_CSV = "../data/iso_causa_other_top.csv"

MODE_LABELS = {
    "ELP": "External leakage - process (ELP)",
    "ELU": "External leakage - utility (ELU)",
    "INL": "Internal leakage (INL)",
    "VIB": "Vibration (VIB)",
    "HIW": "Overheating (HIW)",
    "SHU": "Spontaneous shutdown (SHU)",
    "FTS": "Fail to start (FTS)",
    "STP": "Fail to stop (STP)",
    "LOW": "Low output (LOW)",
    "HIO": "High output (HIO)",
    "NONE": "No output (NONE)",
    "PLU": "Plugged (PLU)",
    "CAV": "Cavitation (CAV)",
    "SHT": "Short circuit (SHT)",
    "INS": "Insulation failure (INS)",
    "COR": "Corrosion (COR)",
    "WEA": "Wear (WEA)",
    "DEF": "Deformation (DEF)",
    "FRA": "Fracture (FRA)",
    "LOO": "Loose part (LOO)",
    "CON": "Control failure (CON)",
    "SIG": "Signal failure (SIG)",
    "MEC": "Mechanical failure (MEC)",
    "ERR": "Erratic reading (ERR)",
    "FAL": "False alarm (FAL)",
    "CAL": "Calibration error (CAL)",
    "STR": "Structural deficiency (STR)",
    "BLO": "Blockage (BLO)",
    "SEI": "Seizure (SEI)",
    "CNT": "Contamination (CNT)",
    "DEV": "Parameter deviation (DEV)",
    "OTH": "Other (OTH)",
}


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def map_mode(raw: str) -> str:
    t = normalize_text(raw)
    if t in ["", "nan", "none"]:
        return "OTH"

    if re.search(r"no especificad|sin especificar|informacion insuficiente|falta de informacion", t):
        return "OTH"

    if re.search(r"(pase interno|fuga interna|interno|bypass)", t):
        return "INL"

    if re.search(r"(fuga|fugas|goteo|escape|perdida|pase)", t):
        if re.search(r"(agua|aire|refrigerante|lubric|aceite|hidraul|vapor|gas)", t):
            return "ELU"
        return "ELP"

    if re.search(r"cavit", t):
        return "CAV"

    if re.search(r"(ruido anormal|ruido fuera de lo normal|ruido excesivo)", t):
        return "VIB"
    if re.search(r"vibr", t):
        return "VIB"

    if re.search(r"(sobrecalent|alta temperatura|temperatura alta|calentamiento|calor excesivo)", t):
        return "HIW"

    if re.search(r"(parada inesperada|paro inesperado|apagado|se apaga|trip|disparo|shutdown|parada por dano)", t):
        return "SHU"

    if re.search(r"(falla de arranque|no arranca|no inicia|no arranque|no parte)", t):
        return "FTS"

    if re.search(r"(no cierra|no cierre)", t):
        return "STP"
    if re.search(r"(no abre|no apertura)", t):
        return "FTS"

    if re.search(r"(no responde|no resp|no respuesta|no opera|falla de operacion|fallo de operacion)", t):
        return "CON"

    if re.search(r"(sin salida|no hay salida|sin flujo|no hay flujo|no hay caudal|salida nula|sin presion)", t):
        return "NONE"
    if re.search(r"(falla de iluminacion|falla de iluminación|iluminacion)", t):
        return "NONE"

    if re.search(r"(bajo caudal|baja presion|bajo flujo|bajo rendimiento|salida baja|baja salida)", t):
        return "LOW"
    if re.search(r"(alto caudal|alta presion|sobreflujo|salida alta|sobrepresion|alto nivel)", t):
        return "HIO" if "caudal" in t else "DEV"

    if re.search(r"(atasc|trabado|agarrot|se traba|pegado)", t):
        return "SEI"
    if re.search(r"(bloqueo|bloqueado|restriccion|paso restringido)", t):
        return "BLO"
    if re.search(r"(obstruccion|taponamiento|tapon|tapado)", t):
        return "PLU"

    if re.search(r"(cortocircuit|corto circuito|corto|chisp)", t):
        return "SHT"
    if re.search(r"(aislamiento|fuga a tierra)", t):
        return "INS"

    if re.search(r"(corrosion|oxid|herrumbre)", t):
        return "COR"
    if re.search(r"(fisur|grieta|fractur|rotura|quiebre)", t):
        return "FRA"
    if re.search(r"(deform|aboll|doblam)", t):
        return "DEF"
    if re.search(r"(desgaste|degradacion|deterioro)", t):
        return "WEA"
    if re.search(r"(dano material|daño material|dano)", t):
        return "MEC"

    if re.search(r"(pieza suelta|holgura|perneria floja|suelto|afloj)", t):
        return "LOO"

    if re.search(r"(control|respuesta tardia|falla de control)", t):
        return "CON"

    if re.search(r"(falsa alarma|falsa indicacion|indicacion defectuosa|falla de alarma)", t):
        return "FAL"
    if re.search(r"(senal erratica|lectura erratica|lectura anormal|oscilante|drift|señal erronea|senal erronea)", t):
        return "ERR"
    if re.search(r"(sin senal|no hay senal|perdida de senal|senal perdida|falla de comunicacion|falla de señal|falta de señal)", t):
        return "SIG"

    if re.search(r"(calibracion|descalibr)", t):
        return "CAL"

    if re.search(r"(estructura|structural|soporte|base|fundacion)", t):
        return "STR"

    if re.search(r"(contamin)", t):
        return "CNT"

    if re.search(r"(desviacion|fuera de rango|setpoint|parametro)", t):
        return "DEV"

    if re.search(r"(falla mecanica|mecanica|mecanico)", t):
        return "MEC"

    return "OTH"


def map_cause(raw: str) -> str:
    t = normalize_text(raw)
    if t in ["", "nan", "none"]:
        return "Other"

    if re.search(r"no especificad|sin especificar|informacion insuficiente|falta de informacion|causa no especificada|indefinida", t):
        return "Other"

    if re.search(r"(vandal|sabot|robo)", t):
        return "Vandalism"
    if re.search(r"(objeto extr|cuerpo extr|ingreso de objeto|piedra|basura)", t):
        return "Foreign object damage"

    if re.search(r"microb|bacter", t):
        return "Microbial corrosion"
    if re.search(r"hidrogen", t):
        return "Hydrogen embrittlement"
    if re.search(r"cavit", t):
        return "Cavitation erosion"
    if re.search(r"corrosion|oxid|herrumbre", t):
        return "Corrosion"
    if re.search(r"erosion|abrasion", t):
        return "Erosion"
    if re.search(r"fatiga", t):
        return "Fatiga"

    if re.search(r"(prensaestopa|empaquetadura|packing)", t):
        return "Packing wear"
    if re.search(r"(empaque|junta|gasket)", t):
        return "Gasket failure"
    if re.search(r"(sello|sellado|reten)", t):
        return "Seal failure"

    if re.search(r"(rodamiento|balin)", t):
        return "Bearing failure"
    if re.search(r"(impulsor|rodete|alabe|aspas)", t):
        return "Impeller damage"
    if re.search(r"(desaline|alineacion|acople|acoplamiento)", t):
        return "Shaft misalignment"

    if re.search(r"actuador|mecanismo de apertura|mecanismo de cierre|mecanismo de la valvula|falla en la valvula", t):
        return "Actuator failure"
    if re.search(r"solenoide", t):
        return "Solenoid failure"

    if re.search(r"(cable|cableado|circuito|luminaria|falla electrica|fallo electrico|sistema electrico)", t):
        return "Wiring fault"
    if re.search(r"(conexion|conector|terminal|enchufe|borne|comunicacion|falla de senal|falla de señal)", t):
        return "Connection failure"

    if re.search(r"fusible", t):
        return "Fuse blown"
    if re.search(r"(breaker|termomagnet|interruptor)", t):
        return "Breaker trip"
    if re.search(r"(sin energia|falta de energia|apag|corte de energia|falla de bateria|power loss|sistema de arranque)", t):
        return "Power loss"

    if re.search(r"(sobretension|pico de tension|surge|descarga)", t):
        return "Electrical surge"

    if re.search(r"(calibracion|descalibr|sensor|instrumentacion|indicador defectuoso|sistema de medicion|sistema de deteccion)", t):
        return "Calibration drift"
    if re.search(r"(sistema de control|control)", t):
        return "Software bug"

    if re.search(r"(procedimiento|instruccion|metodo)", t):
        return "Inadequate procedure"
    if re.search(r"mantenimiento", t):
        return "Maintenance error"
    if re.search(r"(instalacion|montaje)", t):
        return "Installation error"
    if re.search(r"(operador|operacion incorrecta|error de operacion)", t):
        return "Operator error"

    if re.search(r"(sobrecarga|sobre carga|sobreesfuerzo|torque)", t):
        return "Overload"

    if re.search(r"(cambio de proceso|variacion de proceso|cambio de condiciones)", t):
        return "Process change"
    if re.search(r"(temperatura extrema|sobretemperatura|calor extremo|frio extremo|refrigeracion)", t):
        return "Temperature extreme"
    if re.search(r"(sobrepresion|presion extrema|alta presion|baja presion)", t):
        return "Pressure extreme"

    if re.search(r"(sarro|cal|incrustacion)", t):
        return "Scale buildup"
    if re.search(r"(obstruccion|taponamiento|fouling|ensuc|incrust)", t):
        return "Fouling"
    if re.search(r"(contamin|particul|suciedad|lodos|impureza)", t):
        return "Fluid contamination"

    if re.search(r"(ambiente|ambiental|humedad|polvo|salinidad|clima)", t):
        return "External environment"

    if re.search(r"(defecto de material|material defectuoso|no conforme|daño material|dano material)", t):
        return "Material defect"
    if re.search(r"(defecto de fabric|fabrica|ensamble|manufact)", t):
        return "Manufacturing defect"
    if re.search(r"(error de diseno|diseño|diseno)", t):
        return "Design error"
    if re.search(r"repuesto", t):
        return "Spare part quality"

    if re.search(r"(herramienta|golpe|impacto)", t):
        return "Tool damage"

    if re.search(r"(holgura|flojo|suelto|perneria floja)", t):
        return "Looseness"

    if re.search(r"(vida util|fin de vida|obsolesc)", t):
        return "End of life"

    if re.search(r"(lubric|engrase|aceite)", t):
        return "Lubrication failure"

    if re.search(r"(desgaste|degradacion|deterioro|fallo mecanico|falla mecanica|falla en componentes)", t):
        return "Mechanical wear"

    return "Other"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_XLSX)
    df = pd.read_excel(input_path, sheet_name=SHEET)

    df["modo_iso_code"] = df["Modo de falla predecido"].apply(map_mode)
    df["modo_iso_label"] = df["modo_iso_code"].map(MODE_LABELS).fillna("Other (OTH)")
    df["causa_iso_label"] = df["Causa de falla predecida"].apply(map_cause)

    # Report OTH/Other
    modo_oth = df[df["modo_iso_code"] == "OTH"]["Modo de falla predecido"].value_counts().reset_index()
    modo_oth.columns = ["label", "count"]
    causa_other = df[df["causa_iso_label"] == "Other"]["Causa de falla predecida"].value_counts().reset_index()
    causa_other.columns = ["label", "count"]

    modo_oth_path = os.path.join(script_dir, MODO_OTH_CSV)
    causa_other_path = os.path.join(script_dir, CAUSA_OTHER_CSV)
    modo_oth.to_csv(modo_oth_path, index=False)
    causa_other.to_csv(causa_other_path, index=False)

    report_path = os.path.join(script_dir, REPORT_MD)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# REPORTE NORMALIZACION ISO (v2)\n\n")
        f.write(f"- Total registros: {len(df)}\n")
        f.write(f"- OTH modo: {(df['modo_iso_code'] == 'OTH').mean()*100:.1f}%\n")
        f.write(f"- Other causa: {(df['causa_iso_label'] == 'Other').mean()*100:.1f}%\n\n")
        f.write("## Top 30 Modo OTH\n\n")
        f.write(modo_oth.head(30).to_markdown(index=False))
        f.write("\n\n## Top 30 Causa Other\n\n")
        f.write(causa_other.head(30).to_markdown(index=False))
        f.write("\n")

    # Save new file
    out_path = os.path.join(script_dir, OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Consolidado Base ISO v2", index=False)

    print("Archivo generado:", out_path)
    print("Reporte:", report_path)
    print("CSV modo OTH:", modo_oth_path)
    print("CSV causa Other:", causa_other_path)


if __name__ == "__main__":
    main()

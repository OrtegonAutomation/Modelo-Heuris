import pandas as pd
import random
import datetime
import re

# --- CONFIGURACION DE TAXONOMIA ISO 14224 ---

EQUIPMENT_FAMILIES = {
    "BOMBA": [
        "Bomba Centrifuga",
        "Bomba de Desplazamiento Positivo",
        "Bomba Multietapa",
        "Bomba de Cavidad Progresiva",
        "Bomba Jockey",
    ],
    "MOTOR": [
        "Motor Electrico",
        "Motor de Combustion",
        "Servomotor",
        "Motor de Induccion",
        "Motor Sincrono",
    ],
    "VALVULA": [
        "Valvula de Control",
        "Valvula de Seguridad",
        "Valvula de Bola",
        "Valvula de Compuerta",
        "Valvula de Retencion",
        "Valvula Mariposa",
    ],
    "COMPRESOR": [
        "Compresor Reciprocante",
        "Compresor de Tornillo",
        "Compresor Centrifugo",
        "Compresor de Lobulos",
    ],
    "INSTRUMENTO": [
        "Transmisor de Presion",
        "Sensor de Flujo",
        "Termocupla",
        "Analizador de Gas",
        "Transmisor de Nivel",
    ],
    "FILTRO": [
        "Filtro de Felpa",
        "Filtro de Canasta",
        "Filtro de Cartucho",
        "Filtro Coalescente",
        "Filtro Prensa",
    ],
}

FAMILY_WEIGHTS = {
    "BOMBA": 0.25,
    "MOTOR": 0.20,
    "VALVULA": 0.20,
    "COMPRESOR": 0.15,
    "INSTRUMENTO": 0.12,
    "FILTRO": 0.08,
}

FAILURE_MODES = [
    "External leakage - process (ELP)",
    "External leakage - utility (ELU)",
    "Internal leakage (INL)",
    "Vibration (VIB)",
    "Overheating (HIW)",
    "Spontaneous shutdown (SHU)",
    "Fail to start (FTS)",
    "Fail to stop (STP)",
    "Low output (LOW)",
    "High output (HIO)",
    "No output (NONE)",
    "Plugged (PLU)",
    "Cavitation (CAV)",
    "Short circuit (SHT)",
    "Insulation failure (INS)",
    "Corrosion (COR)",
    "Wear (WEA)",
    "Deformation (DEF)",
    "Fracture (FRA)",
    "Loose part (LOO)",
    "Control failure (CON)",
    "Signal failure (SIG)",
    "Mechanical failure (MEC)",
    "Erratic reading (ERR)",
    "False alarm (FAL)",
    "Calibration error (CAL)",
    "Structural deficiency (STR)",
    "Blockage (BLO)",
    "Seizure (SEI)",
    "Contamination (CNT)",
    "Parameter deviation (DEV)",
    "Other (OTH)",
]

FAILURE_CAUSES = [
    "Mechanical wear",
    "Corrosion",
    "Erosion",
    "Fatiga",
    "Overload",
    "Lubrication failure",
    "Material defect",
    "Design error",
    "Installation error",
    "Maintenance error",
    "Operator error",
    "External environment",
    "Fluid contamination",
    "Electrical surge",
    "Software bug",
    "Calibration drift",
    "Fouling",
    "Scale buildup",
    "Seal failure",
    "Bearing failure",
    "Impeller damage",
    "Shaft misalignment",
    "Gasket failure",
    "Packing wear",
    "Solenoid failure",
    "Actuator failure",
    "Wiring fault",
    "Connection failure",
    "Fuse blown",
    "Breaker trip",
    "Power loss",
    "Process change",
    "Temperature extreme",
    "Pressure extreme",
    "Cavitation erosion",
    "Hydrogen embrittlement",
    "Microbial corrosion",
    "Looseness",
    "Foreign object damage",
    "Vandalism",
    "End of life",
    "Manufacturing defect",
    "Spare part quality",
    "Tool damage",
    "Inadequate procedure",
]

CAUSE_CONTEXT = {
    "Mechanical wear": ["friccion metalica", "desgaste por rozamiento", "vida util agotada", "evidencia de roce"],
    "Corrosion": ["oxidacion avanzada", "presencia de herrumbre", "ataque quimico en metal", "picaduras por ambiente"],
    "Erosion": ["desgaste por flujo", "adelgazamiento de pared", "impacto de particulas", "cavidades por arrastre"],
    "Fatiga": ["fisura por ciclos", "rotura por fatiga", "fractura fragil por esfuerzo", "agrietamiento termico"],
    "Overload": ["exceso de torque", "sobreesfuerzo mecanico", "operacion arriba de nominal", "demanda de potencia excesiva"],
    "Lubrication failure": ["falta de engrase", "aceite degradado", "nivel critico de lubricante", "obstruccion en linea de aceite"],
    "Material defect": ["falla de fundicion", "porosidad interna", "impurezas en aleacion", "defecto de fabricacion"],
    "Design error": ["subdimensionamiento", "error en calculo de carga", "geometria inadecuada", "deficiencia proyectual"],
    "Installation error": ["mal montaje", "ajuste incorrecto", "falta de torque en pernos", "alineacion deficiente inicial"],
    "Maintenance error": ["procedimiento mal aplicado", "ajuste fuera de norma", "omision de paso en checklist", "reapriete inadecuado"],
    "Operator error": ["maniobra incorrecta", "setpoint errado", "arranque brusco", "sobrepaso de limites operativos"],
    "External environment": ["humedad extrema", "polvo ambiental", "salinidad del aire", "condiciones climaticas adversas"],
    "Fluid contamination": ["presencia de solidos", "mezcla de fluidos", "particulas abrasivas", "lodos en linea"],
    "Electrical surge": ["pico de tension", "transitorio electrico", "sobretension en linea", "descarga atmosferica"],
    "Software bug": ["error de logica", "falla en firmware", "congelamiento de pantalla", "bug de sistema de control"],
    "Calibration drift": ["desviacion en cero", "desajuste de span", "deriva de lectura", "falta de precision en sensor"],
    "Fouling": ["incrustaciones biologicas", "acumulacion de sedimentos", "suciedad en intercambiador", "capa de sarro"],
    "Scale buildup": ["depositos de cal", "carbonato en paredes", "sarro endurecido", "obstruccion por sales"],
    "Seal failure": ["fuga por caras", "dano en reten", "desgaste de junta torica", "perdida de estanqueidad"],
    "Bearing failure": ["ruido en rodamientos", "vibracion en apoyos", "balineras picadas", "juego radial excesivo"],
    "Impeller damage": ["alabes doblados", "erosion en rodete", "desbalanceo por impacto", "dano en aspas"],
    "Shaft misalignment": ["eje desalineado", "desviacion angular", "vibracion axial alta", "falla de acoplamiento"],
    "Gasket failure": ["empaque soplado", "fisura en junta", "perdida por brida", "degradacion de elastomero"],
    "Packing wear": ["goteo en prensaestopas", "empaquetadura seca", "desgaste de fibras", "falta de apriete en prensa"],
    "Solenoid failure": ["bobina quemada", "valvula pegada", "falla electromagnetica", "actuacion intermitente"],
    "Actuator failure": ["piston trabado", "fuga en diafragma", "falta de aire de instrumentos", "fallo de posicionador"],
    "Wiring fault": ["cable sulfatado", "aislamiento roto", "corto circuito en bornera", "conexion suelta"],
    "Connection failure": ["falla de terminal", "enchufe flojo", "soldadura fria", "falla de pin"],
    "Fuse blown": ["fusible abierto", "proteccion quemada", "interrupcion por sobrecorriente", "falla de fusible"],
    "Breaker trip": ["disparo de termomagnetico", "proteccion activada", "caida de breaker", "reset necesario"],
    "Power loss": ["falta de suministro", "apagon local", "falla en UPS", "caida de fase"],
    "Process change": ["cambio de densidad", "variacion de caudal", "golpe de ariete", "cambio en composicion"],
    "Temperature extreme": ["calor abrasador", "frio extremo", "choque termico", "exceso de radiacion"],
    "Pressure extreme": ["sobrepresion", "vacio no deseado", "pico de presion", "colapso por succion"],
    "Cavitation erosion": ["implosion de burbujas", "picaduras en succion", "ruido de piedras", "dano por vacio"],
    "Hydrogen embrittlement": ["fragilizacion por hidrogeno", "rotura subita", "perdida de ductilidad", "ataque por H2S"],
    "Microbial corrosion": ["ataque por bacteria", "biocapa corrosiva", "corrosion inducida microbiologicamente", "lodos acidos"],
    "Looseness": ["perneria floja", "desajuste de base", "movimiento estructural", "falta de apriete"],
    "Foreign object damage": ["ingreso de piedra", "presencia de trapo", "impacto de tuerca suelta", "objeto extrano en camara"],
    "Vandalism": ["dano provocado", "sabotaje", "robo de cable", "intervencion no autorizada"],
    "End of life": ["desgaste natural excesivo", "obsolescencia", "fatiga acumulada", "fin de ciclo util"],
    "Manufacturing defect": ["pobre control de calidad", "error de ensamble fabrica", "material no conforme", "tolerancia errada"],
    "Spare part quality": ["repuesto alternativo deficiente", "falla prematura de repuesto", "calidad inferior", "lote de repuestos malo"],
    "Tool damage": ["golpe por llave", "dano por herramienta", "marca de impacto", "intervencion brusca"],
    "Inadequate procedure": ["instruccion poco clara", "metodo de trabajo errado", "falta de capacitacion", "secuencia incorrecta"],
}

CAUSE_GROUPS = {
    "Mechanical wear": "mechanical",
    "Fatiga": "mechanical",
    "Overload": "mechanical",
    "Bearing failure": "mechanical",
    "Impeller damage": "mechanical",
    "Shaft misalignment": "mechanical",
    "Looseness": "mechanical",
    "End of life": "mechanical",
    "Tool damage": "mechanical",
    "Foreign object damage": "external",
    "Vandalism": "external",
    "Corrosion": "corrosion",
    "Erosion": "corrosion",
    "Microbial corrosion": "corrosion",
    "Hydrogen embrittlement": "corrosion",
    "Cavitation erosion": "corrosion",
    "Seal failure": "sealing",
    "Gasket failure": "sealing",
    "Packing wear": "sealing",
    "Electrical surge": "electrical",
    "Wiring fault": "electrical",
    "Connection failure": "electrical",
    "Fuse blown": "electrical",
    "Breaker trip": "electrical",
    "Power loss": "electrical",
    "Software bug": "control",
    "Calibration drift": "control",
    "Actuator failure": "control",
    "Solenoid failure": "control",
    "Process change": "process",
    "Temperature extreme": "process",
    "Pressure extreme": "process",
    "Fluid contamination": "process",
    "Fouling": "process",
    "Scale buildup": "process",
    "External environment": "process",
    "Operator error": "human",
    "Maintenance error": "human",
    "Installation error": "human",
    "Inadequate procedure": "human",
    "Design error": "quality",
    "Material defect": "quality",
    "Manufacturing defect": "quality",
    "Spare part quality": "quality",
}

GROUP_CONTEXT = {
    "mechanical": [
        "vibracion en servicio",
        "ruido mecanico sostenido",
        "juego excesivo en componente",
        "desajuste en acople",
        "desbalance detectado",
    ],
    "corrosion": [
        "superficie con picaduras",
        "deterioro por ambiente",
        "material fragilizado",
        "ataque localizado",
        "pitting visible",
    ],
    "sealing": [
        "perdida de estanqueidad",
        "fuga por sello",
        "humedad en empaque",
        "junta reseca",
        "perdida en brida",
    ],
    "electrical": [
        "disparo de proteccion",
        "variacion de tension",
        "alarma electrica",
        "sobreconsumo",
        "aislamiento degradado",
    ],
    "control": [
        "lazo inestable",
        "lecturas inconsistentes",
        "senal intermitente",
        "recalibracion pendiente",
        "respuesta tardia",
    ],
    "process": [
        "cambio de condiciones",
        "presion fuera de rango",
        "flujo irregular",
        "temperatura variable",
        "contaminantes en linea",
    ],
    "human": [
        "intervencion reciente",
        "ajuste manual",
        "procedimiento no estandar",
        "operacion fuera de rutina",
        "registro incompleto",
    ],
    "quality": [
        "material no conforme",
        "tolerancia fuera de especificacion",
        "acabado deficiente",
        "fallo prematuro",
        "defecto de origen",
    ],
    "external": [
        "impacto externo",
        "manipulacion indebida",
        "condicion ambiental severa",
        "objeto ajeno en equipo",
        "intervencion no autorizada",
    ],
}

CONTEXT_BLEND = {
    "cause_prob": 0.20,
    "group_prob": 0.40,
    "generic_prob": 0.40,
    "secondary_prob": 0.65,
    "cross_cause_prob": 0.40,
    "swap_cause_prob": 0.50,
}

ALL_CAUSE_CONTEXTS = [c for contexts in CAUSE_CONTEXT.values() for c in contexts]
ALL_GROUP_CONTEXTS = [c for contexts in GROUP_CONTEXT.values() for c in contexts]

SYMPTOMS = {
    "ELP": ["fuga de crudo", "goteo de producto", "perdida de fluido", "mancha de aceite"],
    "ELU": ["fuga de refrigerante", "escape de aire", "goteo de agua", "perdida de lubricante"],
    "INL": ["paso interno", "falla de sello interno", "bypass involuntario", "comunicacion de camaras"],
    "VIB": ["vibracion excesiva", "ruido anormal", "trepidacion", "golpeteo metalico"],
    "HIW": ["alta temperatura", "sobrecalentamiento", "puntos calientes", "termografia elevada"],
    "SHU": ["disparo subito", "parada inesperada", "trip de sistema", "enclavamiento activado"],
    "FTS": ["no arranca", "bloqueo en el inicio", "falla de encendido", "arranque fallido"],
    "STP": ["no se detiene", "bloqueo al parar", "parada no efectiva", "cierre incompleto"],
    "LOW": ["caudal bajo", "salida reducida", "bajo rendimiento", "presion baja"],
    "HIO": ["caudal alto", "salida elevada", "presion alta", "sobreflujo"],
    "NONE": ["sin salida", "sin caudal", "cero flujo", "salida nula"],
    "PLU": ["obstruccion total", "taponamiento excesivo", "flujo restringido", "caida de presion"],
    "CAV": ["ruido tipo piedras", "vacio en succion", "golpeteo en impulsor", "vibracion por burbujas"],
    "SHT": ["corto circuito", "disparo electrico", "chispazo", "proteccion quemada"],
    "INS": ["aislamiento degradado", "fuga a tierra", "resistencia baja", "deriva de aislamiento"],
    "COR": ["oxidacion visible", "herrumbre", "picadura en metal", "ataque quimico"],
    "WEA": ["desgaste visible", "holgura excesiva", "juego radial", "superficie gastada"],
    "DEF": ["doblamiento", "abolladura", "torsion en pieza", "deformacion visible"],
    "FRA": ["rotura de pieza", "grieta en cuerpo", "quiebre", "fisura abierta"],
    "LOO": ["pieza suelta", "holgura en fijacion", "perneria floja", "vibracion por juego"],
    "CON": ["respuesta erratica de control", "actuacion tardia", "no responde a setpoint", "control intermitente"],
    "SIG": ["perdida de senal", "senal intermitente", "4-20mA perdido", "comunicacion perdida"],
    "MEC": ["dano mecanico visible", "rotura detectada", "desajuste estructural", "fisura en cuerpo"],
    "ERR": ["lectura erratica", "oscilacion en senal", "valor fuera de rango", "fluctuacion constante"],
    "FAL": ["alarma falsa", "indicacion incorrecta", "falso disparo", "alerta sin evento"],
    "CAL": ["desajuste de calibracion", "deriva de cero", "desviacion de span", "mala exactitud"],
    "STR": ["fisura en base", "aflojamiento estructural", "debilidad de soporte", "deformacion en estructura"],
    "BLO": ["bloqueo parcial", "restriccion de paso", "paso obstruido", "caida de presion"],
    "SEI": ["eje trabado", "rotor pegado", "bloqueo mecanico", "movimiento detenido"],
    "CNT": ["contaminacion en fluido", "presencia de particulas", "suciedad en linea", "mezcla no deseada"],
    "DEV": ["desviacion de parametro", "valor fuera de especificacion", "setpoint fuera de rango", "tendencia anomala"],
    "OTH": ["anomalia general", "evento no clasificado", "falla reportada", "comportamiento atipico"],
}

FAMILY_COMPONENTS = {
    "BOMBA": [("rodamiento", 3), ("sello mecanico", 3), ("eje", 2), ("impulsor", 3), ("carcasa", 2), ("empaquetadura", 2)],
    "MOTOR": [("rodamiento", 3), ("bobinado", 3), ("eje", 2), ("ventilador", 2), ("tarjeta electronica", 2), ("cableado", 2)],
    "VALVULA": [("valvula solenoide", 3), ("empaquetadura", 3), ("sello mecanico", 2), ("actuador", 2), ("conector", 1)],
    "COMPRESOR": [("rodamiento", 3), ("impulsor", 2), ("eje", 2), ("carcasa", 2), ("acople", 1)],
    "INSTRUMENTO": [("transmisor", 3), ("sensor", 3), ("tarjeta electronica", 2), ("cableado", 2), ("conector", 1)],
    "FILTRO": [("elemento filtrante", 4), ("carcasa", 2), ("empaquetadura", 2), ("conector", 1)],
}

FAMILY_TAG_RULES = {
    "BOMBA": [("P-", 3, 4), ("BP-", 4, 5)],
    "MOTOR": [("M-", 3, 4), ("ME-", 3, 4)],
    "VALVULA": [("V-", 3, 4), ("MOV-", 5, 6), ("PCV-", 5, 6), ("PSV-", 3, 4)],
    "COMPRESOR": [("K-", 3, 4), ("C-", 3, 4)],
    "INSTRUMENTO": [("PI-", 4, 5), ("TI-", 4, 5), ("FT-", 4, 5), ("LT-", 4, 5)],
    "FILTRO": [("F-", 3, 4), ("FL-", 3, 4)],
}

FAMILY_MODE_WEIGHTS = {
    "BOMBA": [("ELP", 2), ("ELU", 1), ("INL", 1), ("VIB", 2), ("HIW", 2), ("LOW", 2), ("CAV", 2), ("WEA", 2)],
    "MOTOR": [("VIB", 2), ("HIW", 2), ("FTS", 2), ("SHT", 2), ("INS", 2), ("SHU", 2)],
    "VALVULA": [("ELP", 2), ("ELU", 2), ("INL", 2), ("FTS", 2), ("STP", 2), ("CON", 2)],
    "COMPRESOR": [("VIB", 2), ("HIW", 2), ("LOW", 2), ("CAV", 2), ("WEA", 1), ("SEI", 1)],
    "INSTRUMENTO": [("ERR", 3), ("SIG", 3), ("CAL", 2), ("DEV", 2), ("FAL", 1), ("CON", 2)],
    "FILTRO": [("PLU", 3), ("BLO", 3), ("LOW", 2), ("CNT", 2)],
}

COMPONENT_MODE_MAP = {
    "rodamiento": ["VIB", "HIW", "WEA", "SEI", "LOO"],
    "sello mecanico": ["ELP", "ELU", "INL"],
    "empaquetadura": ["ELP", "ELU", "INL"],
    "impulsor": ["VIB", "LOW", "HIO", "CAV", "NONE"],
    "carcasa": ["COR", "DEF", "FRA", "STR", "ELP"],
    "eje": ["VIB", "LOO", "DEF", "FRA", "WEA"],
    "bobinado": ["HIW", "SHT", "INS", "SHU"],
    "tarjeta electronica": ["SHT", "INS", "ERR", "SIG", "CON", "FAL"],
    "cableado": ["SIG", "SHT", "INS", "CON", "ERR"],
    "conector": ["SIG", "ERR", "CON"],
    "valvula solenoide": ["FTS", "STP", "SHU", "CON"],
    "actuador": ["FTS", "STP", "CON"],
    "ventilador": ["VIB", "LOW", "NONE"],
    "elemento filtrante": ["PLU", "BLO", "LOW"],
    "transmisor": ["ERR", "SIG", "CAL", "DEV", "FAL"],
    "sensor": ["ERR", "SIG", "CAL", "DEV", "FAL"],
    "acople": ["VIB", "LOO", "DEF", "FRA"],
}

MODE_CAUSE_MAP = {
    "ELP": ["Seal failure", "Gasket failure", "Packing wear", "Corrosion", "Installation error", "Maintenance error"],
    "ELU": ["Seal failure", "Gasket failure", "Packing wear", "Corrosion", "Installation error", "Maintenance error"],
    "INL": ["Seal failure", "Gasket failure", "Packing wear", "Design error", "Material defect"],
    "VIB": ["Bearing failure", "Shaft misalignment", "Looseness", "Impeller damage", "Mechanical wear", "Installation error"],
    "HIW": ["Lubrication failure", "Overload", "Bearing failure", "Electrical surge", "Temperature extreme"],
    "SHU": ["Electrical surge", "Power loss", "Breaker trip", "Software bug", "Wiring fault"],
    "FTS": ["Power loss", "Wiring fault", "Fuse blown", "Breaker trip", "Actuator failure", "Solenoid failure"],
    "STP": ["Actuator failure", "Solenoid failure", "Software bug", "Wiring fault"],
    "LOW": ["Impeller damage", "Fouling", "Scale buildup", "Process change", "Fluid contamination"],
    "HIO": ["Process change", "Pressure extreme", "Calibration drift", "Software bug", "Operator error"],
    "NONE": ["Power loss", "Fuse blown", "Breaker trip", "Wiring fault", "Actuator failure"],
    "PLU": ["Fouling", "Scale buildup", "Fluid contamination", "Foreign object damage"],
    "CAV": ["Cavitation erosion", "Pressure extreme", "Process change", "Fluid contamination"],
    "SHT": ["Electrical surge", "Wiring fault", "Manufacturing defect", "Material defect"],
    "INS": ["Electrical surge", "Wiring fault", "Manufacturing defect", "Material defect"],
    "COR": ["Corrosion", "External environment", "Microbial corrosion", "Hydrogen embrittlement"],
    "WEA": ["Mechanical wear", "End of life", "Lubrication failure"],
    "DEF": ["Overload", "Tool damage", "Foreign object damage", "Material defect"],
    "FRA": ["Overload", "Foreign object damage", "Material defect", "Manufacturing defect"],
    "LOO": ["Looseness", "Installation error", "Maintenance error"],
    "CON": ["Software bug", "Wiring fault", "Actuator failure", "Connection failure"],
    "SIG": ["Wiring fault", "Connection failure", "Software bug"],
    "MEC": ["Mechanical wear", "Looseness", "Installation error", "Maintenance error"],
    "ERR": ["Calibration drift", "Software bug", "Connection failure", "Wiring fault"],
    "FAL": ["Calibration drift", "Software bug", "Connection failure"],
    "CAL": ["Calibration drift", "Maintenance error", "Inadequate procedure"],
    "STR": ["Design error", "Material defect", "Manufacturing defect", "Overload"],
    "BLO": ["Fouling", "Scale buildup", "Fluid contamination", "Foreign object damage"],
    "SEI": ["Lubrication failure", "Bearing failure", "Overload", "Foreign object damage"],
    "CNT": ["Fluid contamination", "Fouling", "Scale buildup", "External environment"],
    "DEV": ["Calibration drift", "Process change", "Software bug", "Inadequate procedure"],
    "OTH": [],
}

COMPONENT_CAUSE_MAP = {
    "rodamiento": ["Bearing failure", "Lubrication failure", "Mechanical wear"],
    "sello mecanico": ["Seal failure", "Gasket failure", "Packing wear"],
    "empaquetadura": ["Packing wear", "Gasket failure", "Seal failure"],
    "impulsor": ["Impeller damage", "Cavitation erosion", "Foreign object damage"],
    "carcasa": ["Corrosion", "Material defect", "Manufacturing defect"],
    "eje": ["Shaft misalignment", "Mechanical wear", "Looseness"],
    "bobinado": ["Electrical surge", "Manufacturing defect", "Wiring fault"],
    "tarjeta electronica": ["Software bug", "Wiring fault", "Connection failure"],
    "cableado": ["Wiring fault", "Connection failure", "Electrical surge"],
    "conector": ["Connection failure", "Wiring fault"],
    "valvula solenoide": ["Solenoid failure", "Actuator failure", "Wiring fault"],
    "actuador": ["Actuator failure", "Installation error", "Maintenance error"],
    "elemento filtrante": ["Fouling", "Scale buildup", "Fluid contamination"],
    "transmisor": ["Calibration drift", "Software bug", "Connection failure"],
    "sensor": ["Calibration drift", "Software bug", "Connection failure"],
}

GENERIC_CONTEXT = [
    "se observa residuo en superficie",
    "sin evidencia de ajuste reciente",
    "historial de intervencion reciente",
    "condicion operativa variable",
    "indicio de impacto leve",
]

SECONDARY_SYMPTOMS = ["ruido leve", "temperatura elevada", "oscilacion de lectura", "fuga leve", "presion inestable"]
LOCATIONS = ["estacion vasconia", "estacion porvenir", "estacion cusiana", "estacion covenas", "estacion cupiagua"]
DETECTORS = ["Se reporta", "Se detecta", "Se evidencia", "Operador reporta", "Aviso informa"]
INSPECTIONS = ["inspeccion rutinaria", "ronda de campo", "prueba funcional", "inspeccion en sitio"]
TIME_HINTS = ["arranque", "operacion continua", "cambio de turno", "parada de linea", "prueba operacional"]
EVENTS = ["cambio de carga", "arranque", "parada", "prueba de lazo", "cambio de proceso"]
ACTIONS = ["rev tecnica", "ajuste", "limpieza", "reapriete", "verificacion", "mant correctivo"]
MEASUREMENTS = ["vib 12 mm/s", "temp 85 c", "pres 220 psi", "corr 35 A", "flujo 40 %"]

TEMPLATES = [
    "{detector} {symptom} en {component} de {equipment} {tag}. Se observa {context}.",
    "{equipment} {tag} presenta {symptom} durante {time_hint}. Componente: {component}. {context}.",
    "Fallo en {component} de {equipment} {tag}: {symptom}. Posible antecedente de {context}.",
    "Durante {inspection} se evidencia {symptom} en {equipment} {tag}. Afecta {component}. {context}.",
    "{equipment} {tag} fuera de servicio por {symptom}. Revision preliminar indica {context} en {component}.",
    "Bajo rendimiento en {equipment} {tag}. {symptom} detectado en {component}. {context}.",
    "En {location}, {equipment} {tag} presenta {symptom}; {context}. {action} recomendada.",
    "Se detecta {symptom} tras {event} en {equipment} {tag}. {context} en {component}.",
    "OT {ot}: {equipment} {tag}. {symptom}. {context} en {component}.",
    "Registro indica {symptom} y {secondary_symptom} en {equipment} {tag}. {context}.",
]

PRIORITIES = ["1-Critica", "2-Alta", "3-Media", "4-Baja"]

MISSPELLINGS = {
    "vibracion": "vibrasion",
    "fuga": "fubga",
    "mecanico": "mecnaico",
    "bomba": "bba",
    "valvula": "vlv",
    "presion": "presn",
    "temperatura": "tempratura",
    "rodamiento": "rodamnto",
    "inspeccion": "inspc",
}

ABBREVIATIONS = {
    "revision": "rev",
    "mantenimiento": "mant",
    "tecnica": "tec",
    "inspeccion": "insp",
    "operacion": "op",
    "temperatura": "temp",
    "presion": "pres",
    "comunicacion": "com",
    "vibracion": "vib",
    "falla": "flla",
}


def _mode_code(mode):
    if "(" in mode:
        return mode.split(" (")[1].replace(")", "").strip()
    return "MEC"


MODE_CODE_TO_MODE = {_mode_code(m): m for m in FAILURE_MODES}
ALL_MODE_CODES = list(MODE_CODE_TO_MODE.keys())


def validate_taxonomy():
    codes = [_mode_code(m) for m in FAILURE_MODES]
    if len(codes) != len(set(codes)):
        dup = sorted([c for c in set(codes) if codes.count(c) > 1])
        raise ValueError(f"Duplicate mode codes detected: {dup}")
    missing_symptoms = [c for c in codes if c not in SYMPTOMS]
    if missing_symptoms:
        raise ValueError(f"Missing symptoms for codes: {missing_symptoms}")
    missing_causes = [c for c in FAILURE_CAUSES if c not in CAUSE_CONTEXT]
    if missing_causes:
        raise ValueError(f"CAUSE_CONTEXT missing causes: {missing_causes}")


def pick_weighted(weighted_items):
    items = [i for i, _ in weighted_items]
    weights = [w for _, w in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def choose_family():
    return pick_weighted(list(FAMILY_WEIGHTS.items()))


def generate_tag(family):
    rules = FAMILY_TAG_RULES.get(family, [("EQ-", 3, 5)])
    prefix, min_d, max_d = random.choice(rules)
    number = random.randint(10 ** (min_d - 1), (10**max_d) - 1)
    suffix = random.choice(["", "A", "B"]) if random.random() < 0.15 else ""
    return f"{prefix}{number}{suffix}"


def choose_component(family):
    components = FAMILY_COMPONENTS.get(family)
    if components:
        return pick_weighted(components)
    all_components = [c for items in FAMILY_COMPONENTS.values() for c, _ in items]
    return random.choice(all_components)


def choose_mode_code(component, family):
    pool = []
    weights = []

    comp_modes = COMPONENT_MODE_MAP.get(component, [])
    if comp_modes:
        pool.extend(comp_modes)
        weights.extend([4] * len(comp_modes))

    fam_modes = FAMILY_MODE_WEIGHTS.get(family, [])
    if fam_modes:
        pool.extend([m for m, _ in fam_modes])
        weights.extend([w for _, w in fam_modes])

    pool.extend(ALL_MODE_CODES)
    weights.extend([1] * len(ALL_MODE_CODES))

    return random.choices(pool, weights=weights, k=1)[0]


def choose_cause(mode_code, component):
    pool = []
    weights = []

    mode_causes = MODE_CAUSE_MAP.get(mode_code, [])
    if mode_causes:
        pool.extend(mode_causes)
        weights.extend([3] * len(mode_causes))

    comp_causes = COMPONENT_CAUSE_MAP.get(component, [])
    if comp_causes:
        pool.extend(comp_causes)
        weights.extend([2] * len(comp_causes))

    random_causes = random.sample(FAILURE_CAUSES, k=min(6, len(FAILURE_CAUSES)))
    pool.extend(random_causes)
    weights.extend([1] * len(random_causes))

    return random.choices(pool, weights=weights, k=1)[0]


def choose_symptom(mode_code):
    return random.choice(SYMPTOMS.get(mode_code, SYMPTOMS["MEC"]))


def build_context(cause):
    group = CAUSE_GROUPS.get(cause)
    group_contexts = GROUP_CONTEXT.get(group, [])

    roll = random.random()
    if roll < CONTEXT_BLEND["cause_prob"]:
        primary = random.choice(CAUSE_CONTEXT.get(cause, ["falla tecnica reportada"]))
        if random.random() < CONTEXT_BLEND["swap_cause_prob"]:
            primary = random.choice(ALL_CAUSE_CONTEXTS)
    elif roll < (CONTEXT_BLEND["cause_prob"] + CONTEXT_BLEND["group_prob"]) and group_contexts:
        primary = random.choice(group_contexts)
    else:
        primary = random.choice(GENERIC_CONTEXT)

    parts = [primary]

    if random.random() < CONTEXT_BLEND["secondary_prob"]:
        secondary_pool = []
        if group_contexts:
            secondary_pool.extend(group_contexts)
        secondary_pool.extend(GENERIC_CONTEXT)
        parts.append(random.choice(secondary_pool))

    # Inyectar contexto cruzado para evitar que la causa sea unica por texto
    if random.random() < CONTEXT_BLEND["cross_cause_prob"]:
        cross_pool = []
        cross_pool.extend(ALL_GROUP_CONTEXTS)
        cross_pool.extend(ALL_CAUSE_CONTEXTS)
        parts.append(random.choice(cross_pool))

    unique_parts = list(dict.fromkeys(parts))
    return "; ".join(unique_parts)


def choose_priority(mode_code):
    critical_modes = {"SHU", "FTS", "NONE", "SHT", "INS", "SEI"}
    high_modes = {"ELP", "ELU", "INL", "HIW", "VIB", "CAV", "PLU", "BLO"}

    if mode_code in critical_modes:
        return random.choices(["1-Critica", "2-Alta"], weights=[0.6, 0.4], k=1)[0]
    if mode_code in high_modes:
        return random.choices(["2-Alta", "3-Media"], weights=[0.5, 0.5], k=1)[0]
    return random.choices(["3-Media", "4-Baja"], weights=[0.5, 0.5], k=1)[0]


def sanitize_description(text):
    for cause in FAILURE_CAUSES:
        pattern = re.compile(rf"\b{re.escape(cause)}\b", flags=re.IGNORECASE)
        text = pattern.sub("fallo reportado", text)
    for mode in FAILURE_MODES:
        mode_name = mode.split(" (")[0]
        pattern = re.compile(rf"\b{re.escape(mode_name)}\b", flags=re.IGNORECASE)
        text = pattern.sub("averia detectada", text)
    return text


def clean_text(text):
    text = text.replace(" .", ".").replace(" ,", ",")
    text = text.replace("( )", "").replace("()", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def inject_noise(text):
    if random.random() < 0.25:
        for word, abbr in ABBREVIATIONS.items():
            if random.random() < 0.20:
                text = re.sub(rf"\b{word}\b", abbr, text, flags=re.IGNORECASE)

    for word, missp in MISSPELLINGS.items():
        if random.random() < 0.08:
            text = re.sub(rf"\b{word}\b", missp, text, flags=re.IGNORECASE)

    if random.random() < 0.45:
        code = random.choice(["SAP", "OT", "PM"])
        text = f"{text} ({code}-{random.randint(100000, 999999)})"

    if random.random() < 0.10:
        text = text.replace(".", "")
    return text


def generate_dataset(n_records=20000, seed=None):
    if seed is not None:
        random.seed(seed)
    validate_taxonomy()

    data = []
    print(f"Generando {n_records} registros con coherencia tecnica y anti-leakage...")

    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    delta_days = (end_date - start_date).days

    for i in range(n_records):
        family = choose_family()
        equipment = random.choice(EQUIPMENT_FAMILIES[family])
        component = choose_component(family)

        mode_code = choose_mode_code(component, family)
        mode = MODE_CODE_TO_MODE.get(mode_code, random.choice(FAILURE_MODES))
        cause = choose_cause(mode_code, component)

        symptom = choose_symptom(mode_code)
        secondary_symptom = random.choice(SECONDARY_SYMPTOMS)
        context = build_context(cause)

        tag = generate_tag(family)
        tag_text = tag if random.random() < 0.90 else ""

        description = random.choice(TEMPLATES).format(
            symptom=symptom,
            secondary_symptom=secondary_symptom,
            component=component,
            equipment=equipment,
            tag=tag_text,
            context=context,
            detector=random.choice(DETECTORS),
            time_hint=random.choice(TIME_HINTS),
            inspection=random.choice(INSPECTIONS),
            event=random.choice(EVENTS),
            action=random.choice(ACTIONS),
            location=random.choice(LOCATIONS),
            measurement=random.choice(MEASUREMENTS),
            ot=random.randint(100000, 999999),
        )

        description = sanitize_description(description)
        description = clean_text(description)
        description = inject_noise(description)
        description = clean_text(description)

        priority = choose_priority(mode_code)
        fecha = start_date + datetime.timedelta(days=random.randint(0, delta_days))

        data.append(
            {
                "id_aviso": 1000000 + i,
                "descripcion": description,
                "equipo": tag,
                "familia_equipo": family,
                "modo_falla": mode,
                "causa_falla": cause,
                "prioridad": priority,
                "fecha_aviso": fecha.strftime("%Y-%m-%d"),
                "componente_afectado": component,
            }
        )

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_dataset(20000, seed=42)
    df.to_csv("synthetic_training_data.csv", index=False)
    print("Dataset sintetico regenerado con coherencia y anti-leakage.")

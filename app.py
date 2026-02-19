"""
Aplicacion Streamlit para desambiguacion de terminos historicos.
Version 2: integra similitud fonetica y semantica (embeddings).
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from io import BytesIO
import zipfile

try:
    from metaphone import doublemetaphone as _doublemetaphone
    _DM_DISPONIBLE = True
except ImportError:
    _DM_DISPONIBLE = False

try:
    import jellyfish as _jellyfish
    _JELLYFISH_DISPONIBLE = True
except ImportError:
    _JELLYFISH_DISPONIBLE = False

try:
    from text2vec import SentenceModel as _Text2VecModel
    _TEXT2VEC_DISPONIBLE = True
except ImportError:
    _TEXT2VEC_DISPONIBLE = False

try:
    from sentence_transformers import SentenceTransformer as _STModel
    _ST_DISPONIBLE = True
except ImportError:
    _ST_DISPONIBLE = False

# =============================================================================
# CONFIGURACION DE LA APP
# =============================================================================

st.set_page_config(
    page_title="Desambiguador de terminos v2",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directorio base
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def normalize_name(name: str) -> str:
    """Normaliza el nombre a una forma canonica basica."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFD", name)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = without_marks.lower()
    cleaned = re.sub(r"[^a-z\s-]", " ", lowered)
    return " ".join(cleaned.split())


def cargar_voces_lista(archivo: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Carga todas las voces de un archivo de lista.
    Retorna:
    - Lista de voces
    - Diccionario voz_normalizada -> entidad
    """
    voces = []
    voz_a_entidad = {}  # mapeo voz_normalizada -> entidad
    entidad_actual = None

    with open(archivo, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.rstrip()
            if not linea.strip():
                continue
            # Linea de entidad (NOMBRE:)
            if linea.strip().endswith(":") and not linea.startswith(" "):
                entidad_actual = linea.strip()[:-1]  # quitar el ":"
                continue
            # Linea de voz (- voz)
            if linea.strip().startswith("-"):
                voz = linea.strip()[1:].strip()
                if voz:
                    voces.append(voz)
                    voz_norm = normalize_name(voz)
                    if voz_norm and entidad_actual:
                        voz_a_entidad[voz_norm] = entidad_actual
    return voces, voz_a_entidad


# =============================================================================
# ALGORITMOS DE SIMILITUD
# =============================================================================

# Grupos de confusion OCR
OCR_CONFUSION_GROUPS = [
    {"c", "e"}, {"p", "n", "r"}, {"a", "o"}, {"l", "i", "1"},
    {"m", "n"}, {"u", "v"}, {"g", "q"}, {"h", "b"},
    {"d", "cl"}, {"rn", "m"}, {"f", "t"}, {"s", "5"},
]

_OCR_CONFUSION_PAIRS = set()
for group in OCR_CONFUSION_GROUPS:
    items = list(group)
    for i, x in enumerate(items):
        for y in items[i+1:]:
            _OCR_CONFUSION_PAIRS.add((x, y))
            _OCR_CONFUSION_PAIRS.add((y, x))


def ocr_substitution_cost(ca: str, cb: str, confusion_cost: float = 0.4) -> float:
    """Costo de sustitucion para OCR: penaliza menos confusiones tipicas."""
    if ca == cb:
        return 0.0
    if (ca, cb) in _OCR_CONFUSION_PAIRS:
        return confusion_cost
    return 1.0


def levenshtein_distance(a: str, b: str) -> int:
    """Distancia de edicion clasica (Levenshtein) entre dos cadenas."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def levenshtein_ratio_raw(a: str, b: str) -> float:
    """Similitud 0-1 basada en Levenshtein sin normalizacion previa."""
    max_len = max(len(a), len(b), 1)
    return 1.0 - levenshtein_distance(a, b) / max_len


def levenshtein_ratio(a: str, b: str) -> float:
    """Similitud 0-1 basada en Levenshtein sobre texto normalizado."""
    na, nb = normalize_name(a), normalize_name(b)
    max_len = max(len(na), len(nb), 1)
    return 1.0 - levenshtein_distance(na, nb) / max_len


def levenshtein_distance_ocr(a: str, b: str, confusion_cost: float = 0.4) -> float:
    """Distancia Levenshtein con costo reducido para confusiones OCR."""
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    previous = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        current = [float(i)]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j-1] + 1.0, previous[j] + 1.0,
                             previous[j-1] + ocr_substitution_cost(ca, cb, confusion_cost)))
        previous = current
    return previous[-1]


def levenshtein_ratio_ocr(a: str, b: str) -> float:
    """Similitud 0-1 que usa Levenshtein con ajustes OCR para textos cortos."""
    na, nb = normalize_name(a), normalize_name(b)
    if len(na) <= 5 and len(nb) <= 5:
        max_len = max(len(na), len(nb), 1)
        return 1.0 - levenshtein_distance_ocr(na, nb) / max_len
    return levenshtein_ratio(a, b)


def jaro_winkler_similarity(a: str, b: str, prefix_weight: float = 0.1) -> float:
    """Similitud Jaro-Winkler 0-1: pondera coincidencias y prefijos."""
    na, nb = normalize_name(a), normalize_name(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0

    max_distance = max(len(na), len(nb)) // 2 - 1
    if max_distance < 0:
        max_distance = 0

    matches_a = [False] * len(na)
    matches_b = [False] * len(nb)
    matches = 0

    for i, char_a in enumerate(na):
        start = max(0, i - max_distance)
        end = min(i + max_distance + 1, len(nb))
        for j in range(start, end):
            if matches_b[j] or char_a != nb[j]:
                continue
            matches_a[i] = True
            matches_b[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i, matched in enumerate(matches_a):
        if not matched:
            continue
        while not matches_b[k]:
            k += 1
        if na[i] != nb[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    jaro = ((matches / len(na)) + (matches / len(nb)) + ((matches - transpositions) / matches)) / 3

    prefix_len = 0
    for ca, cb in zip(na, nb):
        if ca != cb or prefix_len == 4:
            break
        prefix_len += 1

    return jaro + prefix_len * prefix_weight * (1 - jaro)


def ngram_similarity(a: str, b: str, n: int = 2) -> float:
    """Similitud 0-1 por Jaccard de n-gramas de caracteres."""
    na = normalize_name(a).replace(" ", "")
    nb = normalize_name(b).replace(" ", "")

    def get_ngrams(text, n):
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    ngrams_a, ngrams_b = get_ngrams(na, n), get_ngrams(nb, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def ngram_similarity_2(a: str, b: str) -> float:
    """Atajo para similitud por bigramas (n=2)."""
    return ngram_similarity(a, b, n=2)


def ngram_similarity_3(a: str, b: str) -> float:
    """Atajo para similitud por trigramas (n=3)."""
    return ngram_similarity(a, b, n=3)


# ---- Similitud fonetica (Double Metaphone) ---------------------------------

def _phonetic_codes(texto: str) -> List[str]:
    """Genera codigos foneticos (Double Metaphone o fallback Jellyfish)."""
    norm = normalize_name(texto).replace(" ", "")
    if not norm:
        return []
    if _DM_DISPONIBLE:
        codes = _doublemetaphone(norm)
        return [c for c in codes if c]
    if _JELLYFISH_DISPONIBLE and hasattr(_jellyfish, "metaphone"):
        code = _jellyfish.metaphone(norm)
        return [code] if code else []
    return []


def phonetic_similarity(a: str, b: str) -> float:
    """Similitud fonetica 0-1 comparando codigos foneticos."""
    codes_a = _phonetic_codes(a)
    codes_b = _phonetic_codes(b)
    if not codes_a or not codes_b:
        return 0.0
    if any(ca == cb for ca in codes_a for cb in codes_b if ca and cb):
        return 1.0
    best = 0.0
    for ca in codes_a:
        for cb in codes_b:
            if not ca or not cb:
                continue
            best = max(best, levenshtein_ratio_raw(ca, cb))
    return best


# ---- Similitud semantica (embeddings) --------------------------------------

_ARTICULOS = re.compile(
    r"\b(de\s+la|de\s+los|del|de|la|le|les|du|des|von|van|hms|el|los|las)\b",
    re.IGNORECASE
)
_SANTOS = re.compile(
    r"\b(san|santa|saint|sainte|st)\b",
    re.IGNORECASE
)
_TRATAMIENTOS = re.compile(
    r"\b(cpt|capt|cap|cptn|capitan|capitan|capitaine|captain|cmdr|comdr|"
    r"don|dn|sr|mr|msr|sieur|senor|senor|monsieur|m)\b\.?\s*",
    re.IGNORECASE
)


def _quitar_diacriticos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def normalizar_semantico(nombre: str) -> List[str]:
    """Normaliza tokens para embeddings: limpia articulos, tratamientos y ruido."""
    s = _quitar_diacriticos(nombre).lower()
    s = _TRATAMIENTOS.sub(" ", s)
    s = _ARTICULOS.sub(" ", s)
    s = _SANTOS.sub("san", s)
    s = re.sub(r"\b[a-z]\.\s*", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if len(t) >= 2]


def normalizar_semantico_str(nombre: str) -> str:
    tokens = normalizar_semantico(nombre)
    return " ".join(tokens) if tokens else nombre.lower().strip()


class ModeloEmbeddings:
    def __init__(
        self,
        backend: str = "auto",
        modelo_texto: Optional[str] = None,
        ruta_local: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self._cache: Dict[str, np.ndarray] = {}
        self._dim = 0

        if backend == "auto":
            if _TEXT2VEC_DISPONIBLE:
                backend = "text2vec"
            elif _ST_DISPONIBLE:
                backend = "sentence_transformers"
            else:
                raise ImportError(
                    "Instala al menos uno: text2vec o sentence-transformers"
                )

        self.backend = backend
        fuente = ruta_local or modelo_texto

        if backend == "text2vec":
            if not _TEXT2VEC_DISPONIBLE:
                raise ImportError("pip install text2vec")
            fuente = fuente or "shibing624/text2vec-base-multilingual"
            self._modelo = _Text2VecModel(fuente)
            self.modelo = fuente

        elif backend == "sentence_transformers":
            if not _ST_DISPONIBLE:
                raise ImportError("pip install sentence-transformers")
            fuente = fuente or "paraphrase-multilingual-mpnet-base-v2"
            self._modelo = _STModel(fuente, device=device)
            self.modelo = fuente

        else:
            raise ValueError(f"Backend desconocido: {backend!r}")

    def encode(self, textos: List[str], batch_size: int = 256) -> np.ndarray:
        """Codifica textos en embeddings normalizados y cacheados."""
        nuevos = [t for t in textos if t not in self._cache]

        if nuevos:
            if self.backend == "text2vec":
                vecs = self._modelo.encode(
                    nuevos,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            else:
                vecs = self._modelo.encode(
                    nuevos,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-10)

            if self._dim == 0:
                self._dim = vecs.shape[1]

            for t, v in zip(nuevos, vecs):
                self._cache[t] = v

        return np.array([self._cache[t] for t in textos])

    def encode_uno(self, texto: str) -> np.ndarray:
        return self.encode([texto])[0]

    def similitud_cosine(self, texto1: str, texto2: str) -> float:
        """Similitud coseno 0-1 entre embeddings de texto."""
        v1 = self.encode_uno(texto1)
        v2 = self.encode_uno(texto2)
        raw = float(np.dot(v1, v2))
        return round((raw + 1.0) / 2.0, 4)

    @property
    def dimension(self) -> int:
        if self._dim == 0 and self._cache:
            self._dim = next(iter(self._cache.values())).shape[0]
        return self._dim


@st.cache_resource(show_spinner=False)
def cargar_modelo_embeddings(
    backend: str,
    modelo_texto: Optional[str],
    ruta_local: Optional[str],
    device: str,
) -> ModeloEmbeddings:
    return ModeloEmbeddings(
        backend=backend,
        modelo_texto=modelo_texto,
        ruta_local=ruta_local,
        device=device,
    )


def generar_matriz_similitud_pairwise(
    terminos: List[str],
    voces: List[str],
    funcion_similitud,
    progress_bar=None,
) -> pd.DataFrame:
    n_filas = len(terminos)
    n_cols = len(voces)
    matriz = np.zeros((n_filas, n_cols))

    total = n_filas
    for i, tf in enumerate(terminos):
        for j, voz in enumerate(voces):
            matriz[i, j] = funcion_similitud(tf, voz)
        if progress_bar:
            progress_bar.progress((i + 1) / total)

    return pd.DataFrame(matriz, index=terminos, columns=voces)


def generar_matriz_semantica(
    terminos: List[str],
    voces: List[str],
    modelo: ModeloEmbeddings,
    progress_bar=None,
) -> pd.DataFrame:
    textos_a = [normalizar_semantico_str(t) for t in terminos]
    textos_b = [normalizar_semantico_str(v) for v in voces]

    vecs_a = modelo.encode(textos_a)
    if progress_bar:
        progress_bar.progress(0.5)
    vecs_b = modelo.encode(textos_b)

    cosine = np.dot(vecs_a, vecs_b.T)
    sim = (cosine + 1.0) / 2.0
    sim = np.clip(sim, 0.0, 1.0)

    if progress_bar:
        progress_bar.progress(1.0)

    return pd.DataFrame(sim, index=terminos, columns=voces)


# Diccionario de algoritmos disponibles
ALGORITMOS_DISPONIBLES = {
    "Levenshtein_OCR": {"tipo": "pairwise", "func": levenshtein_ratio_ocr},
    "Levenshtein_Ratio": {"tipo": "pairwise", "func": levenshtein_ratio},
    "Jaro_Winkler": {"tipo": "pairwise", "func": jaro_winkler_similarity},
    "NGram_2": {"tipo": "pairwise", "func": ngram_similarity_2},
    "NGram_3": {"tipo": "pairwise", "func": ngram_similarity_3},
    "Fonetica_DM": {"tipo": "pairwise", "func": phonetic_similarity},
    "Semantica_Embeddings": {"tipo": "semantic", "func": None},
}

# Umbrales por defecto
UMBRALES_DEFAULT = {
    "Levenshtein_OCR": 0.75,
    "Levenshtein_Ratio": 0.75,
    "Jaro_Winkler": 0.85,
    "NGram_2": 0.66,
    "NGram_3": 0.60,
    "Fonetica_DM": 0.85,
    "Semantica_Embeddings": 0.975,
}

ZONAS_GRIS_DEFAULT = {
    "Levenshtein_OCR": (0.71, 0.749),
    "Levenshtein_Ratio": (0.71, 0.749),
    "Jaro_Winkler": (0.80, 0.849),
    "NGram_2": (0.63, 0.659),
    "NGram_3": (0.55, 0.599),
    "Fonetica_DM": (0.80, 0.849),
    "Semantica": (0.965, 0.974),
}

# =============================================================================
# MAPEO DE CAMPOS Y LISTAS
# =============================================================================

CAMPOS_DISPONIBLES = {
    "master_role": {
        "nombre": "Rol del capitan/patron",
        "lista": "master_role.txt",
        "csv": "terminos_master_role.csv"
    },
    "ship_flag": {
        "nombre": "Bandera del barco",
        "lista": "lista_banderas.txt",
        "csv": "terminos_ship_flag.csv"
    },
    "ship_type": {
        "nombre": "Tipo de embarcacion",
        "lista": "lista_barcos.txt",
        "csv": "terminos_ship_type.csv"
    },
    "travel_departure_port": {
        "nombre": "Puerto de salida",
        "lista": "lista_puertos.txt",
        "csv": "terminos_travel_departure_port.csv"
    },
    "travel_arrival_port": {
        "nombre": "Puerto de llegada",
        "lista": "lista_puertos.txt",
        "csv": "terminos_travel_arrival_port.csv"
    },
}


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_terminos_csv(archivo: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Carga terminos unicos desde CSV preprocesado.
    Retorna lista de terminos y diccionario de frecuencias.
    """
    df = pd.read_csv(archivo, encoding="utf-8-sig")
    terminos = df["termino_normalizado"].tolist()
    frecuencias = dict(zip(df["termino_normalizado"], df["frecuencia"]))
    total_valores = df["frecuencia"].sum()
    return terminos, frecuencias, total_valores


# =============================================================================
# FUNCIONES DE PROCESAMIENTO
# =============================================================================


def clasificar_terminos(
    todos_unicos: List[str],
    contador_frecuencias: Dict[str, int],
    matrices: Dict[str, pd.DataFrame],
    config: Dict,
    voz_a_entidad: Dict[str, str],
) -> pd.DataFrame:
    """Clasifica terminos segun configuracion."""

    algoritmos = config["algoritmos"]
    umbrales = config["umbrales"]
    zonas_gris = config["zonas_gris"]
    requiere_lev_ocr = "Levenshtein_OCR" in algoritmos

    datos = []

    for termino in todos_unicos:
        frecuencia = contador_frecuencias.get(termino, 0)
        fila = {"termino": termino, "frecuencia": frecuencia}

        votos_aprobacion = 0
        en_zona_gris = False
        # entidad -> [(algoritmo, voz)]
        votos_por_entidad = {}
        voz_lev_ocr = None  # la voz que voto Levenshtein_OCR

        for nombre_algo in algoritmos:
            if nombre_algo not in matrices:
                continue

            df = matrices[nombre_algo]
            if termino not in df.index:
                continue

            row = df.loc[termino]
            max_sim = row.max()
            mejor_voz = row.idxmax()

            umbral = umbrales.get(nombre_algo, 0.7)
            zona_piso, zona_techo = zonas_gris.get(nombre_algo, (0.6, 0.69))

            fila[f"sim_{nombre_algo}"] = round(max_sim, 4)
            fila[f"voz_{nombre_algo}"] = mejor_voz

            if max_sim >= umbral:
                votos_aprobacion += 1
                # Obtener la entidad de esta voz
                entidad = voz_a_entidad.get(mejor_voz, mejor_voz)  # si no hay entidad, usar la voz misma
                if entidad not in votos_por_entidad:
                    votos_por_entidad[entidad] = []
                votos_por_entidad[entidad].append((nombre_algo, mejor_voz))

                if nombre_algo == "Levenshtein_OCR":
                    voz_lev_ocr = mejor_voz
            elif zona_piso <= max_sim <= zona_techo:
                en_zona_gris = True

        # Determinar entidad consensuada (la que tiene mas votos)
        entidad_consenso = ""
        votos_entidad = 0
        algoritmos_consenso = []
        lev_ocr_en_consenso = False
        voz_consenso = ""

        if votos_por_entidad:
            entidad_consenso = max(votos_por_entidad, key=lambda e: len(votos_por_entidad[e]))
            votos_entidad = len(votos_por_entidad[entidad_consenso])
            algoritmos_consenso = [algo for algo, voz in votos_por_entidad[entidad_consenso]]

            # Verificar si Levenshtein_OCR esta entre los que votan por la entidad consensuada
            lev_ocr_en_consenso = "Levenshtein_OCR" in algoritmos_consenso

            # La voz consenso es la que voto Levenshtein_OCR (si esta en consenso)
            if lev_ocr_en_consenso:
                voz_consenso = voz_lev_ocr
            else:
                # Si Lev-OCR no esta, tomar la primera voz de la entidad consensuada
                voz_consenso = votos_por_entidad[entidad_consenso][0][1]

        fila["votos_aprobacion"] = votos_aprobacion
        fila["entidad_consenso"] = entidad_consenso
        fila["voz_consenso"] = voz_consenso
        fila["votos_entidad_consenso"] = votos_entidad
        fila["Levenshtein_OCR_en_consenso"] = lev_ocr_en_consenso

        # Clasificacion: 2+ votos por entidad y, si aplica, Levenshtein_OCR en consenso
        if votos_entidad >= 2 and (not requiere_lev_ocr or lev_ocr_en_consenso):
            fila["clasificacion"] = "CONSENSUADO"
        elif votos_aprobacion >= 2:
            fila["clasificacion"] = "CONSENSUADO_DEBIL"
        elif votos_aprobacion == 1:
            fila["clasificacion"] = "SOLO_1_VOTO"
        elif en_zona_gris:
            fila["clasificacion"] = "ZONA_GRIS"
        else:
            fila["clasificacion"] = "RECHAZADO"

        datos.append(fila)

    df = pd.DataFrame(datos)
    return df.sort_values("frecuencia", ascending=False)


# =============================================================================
# INFORMES Y EXPORTACION
# =============================================================================

def generar_informe_txt(
    df_clasificacion: pd.DataFrame,
    nombre_campo: str,
    config: Dict,
    stats: Dict,
) -> str:
    """Genera informe de texto con datos generales del proceso."""
    from datetime import datetime

    lineas = []
    lineas.append("=" * 70)
    lineas.append("INFORME DE DESAMBIGUACION - PROYECTO PORTADA")
    lineas.append("=" * 70)
    lineas.append(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append("")

    # Datos generales
    lineas.append("-" * 70)
    lineas.append("DATOS GENERALES")
    lineas.append("-" * 70)
    lineas.append(f"Campo analizado: {nombre_campo}")
    lineas.append(f"Archivo de voces: {stats.get('archivo_voces', 'N/A')}")
    lineas.append(f"Total valores (ocurrencias): {stats.get('total_valores', 0):,}")
    lineas.append(f"Terminos unicos: {stats.get('terminos_unicos', 0):,}")
    lineas.append(f"Voces en lista (raw): {stats.get('voces_raw', 0):,}")
    lineas.append(f"Voces usadas (filtradas): {stats.get('voces_filtradas', 0):,}")
    lineas.append(f"Min apariciones para voz: {stats.get('min_apariciones_voz', 0)}")
    lineas.append("")

    # Configuracion de algoritmos
    lineas.append("-" * 70)
    lineas.append("CONFIGURACION DE ALGORITMOS")
    lineas.append("-" * 70)
    lineas.append(f"Algoritmos seleccionados: {', '.join(config['algoritmos'])}")

    regla = "2+ votos por misma entidad"
    if "Levenshtein_OCR" in config.get("algoritmos", []):
        regla += " + Levenshtein_OCR en consenso"
    lineas.append(f"Regla de consenso: {regla}")
    lineas.append("")
    lineas.append("Umbrales de aprobacion:")
    for algo, umbral in config["umbrales"].items():
        lineas.append(f"  - {algo}: {umbral}")
    lineas.append("")
    lineas.append("Zonas grises (piso, techo):")
    for algo, zona in config["zonas_gris"].items():
        lineas.append(f"  - {algo}: {zona}")
    lineas.append("")

    if "semantic_config" in config:
        sem = config["semantic_config"]
        lineas.append("Configuracion semantica:")
        lineas.append(f"  - backend: {sem.get('backend', 'N/A')}")
        lineas.append(f"  - modelo: {sem.get('modelo', 'N/A')}")
        if sem.get("ruta_local"):
            lineas.append(f"  - ruta_local: {sem.get('ruta_local')}")
        lineas.append("")

    # Resultados por clasificacion
    lineas.append("-" * 70)
    lineas.append("RESULTADOS POR CLASIFICACION")
    lineas.append("-" * 70)

    total_valores = stats.get("total_valores", 1)
    total_terminos = len(df_clasificacion)

    lineas.append(f"{'Clasificacion':<20} {'Terminos':>10} {'Ocurrencias':>12} {'Porcentaje':>10}")
    lineas.append("-" * 54)

    for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
        df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
        n_terminos = len(df_cat)
        ocurrencias = df_cat["frecuencia"].sum()
        pct = ocurrencias / total_valores * 100 if total_valores > 0 else 0
        lineas.append(f"{clasificacion:<20} {n_terminos:>10,} {ocurrencias:>12,} {pct:>9.2f}%")

    lineas.append("-" * 54)
    total_ocurrencias = df_clasificacion["frecuencia"].sum()
    lineas.append(f"{'TOTAL':<20} {total_terminos:>10,} {total_ocurrencias:>12,} {100.0:>9.2f}%")
    lineas.append("")

    # Resumen de cobertura
    lineas.append("-" * 70)
    lineas.append("RESUMEN DE COBERTURA")
    lineas.append("-" * 70)

    freq_cons = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO"]["frecuencia"].sum()
    freq_debil = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"]["frecuencia"].sum()
    pct_cons = freq_cons / total_valores * 100 if total_valores > 0 else 0
    pct_total = (freq_cons + freq_debil) / total_valores * 100 if total_valores > 0 else 0

    lineas.append(f"Consensuado (estricto): {freq_cons:,} de {total_valores:,} ({pct_cons:.2f}%)")
    lineas.append(f"Consensuado + Debil:    {freq_cons + freq_debil:,} de {total_valores:,} ({pct_total:.2f}%)")
    lineas.append("")

    # Archivos generados
    lineas.append("-" * 70)
    lineas.append("ARCHIVOS GENERADOS")
    lineas.append("-" * 70)
    lineas.append(f"- clasificacion_completa_{nombre_campo}.csv")
    for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
        df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
        if len(df_cat) > 0:
            lineas.append(f"- {clasificacion.lower()}_{nombre_campo}.csv")
    lineas.append(f"- informe_{nombre_campo}.txt")
    lineas.append("")

    lineas.append("=" * 70)
    lineas.append("FIN DEL INFORME")
    lineas.append("=" * 70)

    return "\n".join(lineas)


def crear_zip_csvs(
    df_clasificacion: pd.DataFrame,
    nombre_campo: str,
    config: Dict = None,
    stats: Dict = None,
) -> bytes:
    """Crea un ZIP con todos los CSVs y el informe TXT."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # CSV completo
        csv_completo = df_clasificacion.to_csv(index=False, encoding="utf-8-sig")
        zf.writestr(f"clasificacion_completa_{nombre_campo}.csv", csv_completo)

        # CSVs por clasificacion
        for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
            df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
            if len(df_cat) > 0:
                csv_cat = df_cat.to_csv(index=False, encoding="utf-8-sig")
                nombre_archivo = clasificacion.lower() + f"_{nombre_campo}.csv"
                zf.writestr(nombre_archivo, csv_cat)

        # Informe TXT
        if config and stats:
            informe = generar_informe_txt(df_clasificacion, nombre_campo, config, stats)
            zf.writestr(f"informe_{nombre_campo}.txt", informe.encode("utf-8"))

    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

def main():
    st.title("Desambiguador de terminos historicos (v2)")

    st.markdown(
        "*Integra similitud fonetica (Double Metaphone) y semantica (embeddings multilingues).*"
    )

    st.markdown("---")

    # Verificar que existen los archivos necesarios
    directorio_listas = BASE_DIR / "listas"
    directorio_datos = BASE_DIR / "datos"

    if not directorio_listas.exists():
        st.error(f"No se encontro el directorio de listas: {directorio_listas}")
        return

    if not directorio_datos.exists():
        st.error(f"No se encontro el directorio de datos: {directorio_datos}")
        return

    # =================
    # SIDEBAR: Configuracion
    # =================
    st.sidebar.header("Configuracion")

    # 1. Seleccion de campo
    st.sidebar.subheader("1. Campo a analizar")
    campo_seleccionado = st.sidebar.selectbox(
        "Seleccione el campo:",
        options=list(CAMPOS_DISPONIBLES.keys()),
        format_func=lambda x: f"{x} - {CAMPOS_DISPONIBLES[x]['nombre']}"
    )

    campo_config = CAMPOS_DISPONIBLES[campo_seleccionado]
    archivo_lista = directorio_listas / campo_config["lista"]
    archivo_csv = directorio_datos / campo_config["csv"]

    if not archivo_lista.exists():
        st.sidebar.error(f"No existe: {campo_config['lista']}")
        return

    if not archivo_csv.exists():
        st.sidebar.error(f"No existe: {campo_config['csv']}")
        return

    # 2. Seleccion de algoritmos
    st.sidebar.subheader("2. Algoritmos")
    algoritmos_seleccionados = st.sidebar.multiselect(
        "Seleccione algoritmos:",
        options=list(ALGORITMOS_DISPONIBLES.keys()),
        default=["Levenshtein_OCR", "Jaro_Winkler", "NGram_2"]
    )

    if len(algoritmos_seleccionados) < 2:
        st.sidebar.warning("Seleccione al menos 2 algoritmos")
        return

    # 3. Umbrales
    st.sidebar.subheader("3. Umbrales de aprobacion")
    umbrales = {}
    for algo in algoritmos_seleccionados:
        default = UMBRALES_DEFAULT.get(algo, 0.7)
        umbrales[algo] = st.sidebar.slider(
            f"{algo}:",
            min_value=0.5,
            max_value=1.0,
            value=default,
            step=0.01,
            key=f"umbral_{algo}"
        )

    # 4. Zonas grises
    st.sidebar.subheader("4. Zonas grises")
    zonas_gris = {}
    for algo in algoritmos_seleccionados:
        default_piso, default_techo = ZONAS_GRIS_DEFAULT.get(algo, (0.6, 0.69))
        col1, col2 = st.sidebar.columns(2)
        with col1:
            piso = st.number_input(f"{algo} piso:", value=default_piso, step=0.01, key=f"zg_piso_{algo}")
        with col2:
            techo = st.number_input("techo:", value=default_techo, step=0.01, key=f"zg_techo_{algo}")
        zonas_gris[algo] = (piso, techo)

    # 5. Filtros adicionales
    st.sidebar.subheader("5. Filtros")
    min_apariciones_voz = st.sidebar.number_input(
        "Min apariciones para incluir voz (0=todas):",
        min_value=0,
        value=0
    )

    # 6. Configuracion semantica (solo si se usa)
    semantic_config = None
    if "Semantica_Embeddings" in algoritmos_seleccionados:
        st.sidebar.subheader("6. Semantica")
        backend = st.sidebar.selectbox(
            "Backend embeddings:",
            options=["auto", "text2vec", "sentence_transformers"],
            index=0,
        )
        if backend == "text2vec":
            modelo_default = "shibing624/text2vec-base-multilingual"
        elif backend == "sentence_transformers":
            modelo_default = "paraphrase-multilingual-mpnet-base-v2"
        else:
            modelo_default = (
                "shibing624/text2vec-base-multilingual"
                if _TEXT2VEC_DISPONIBLE
                else "paraphrase-multilingual-mpnet-base-v2"
            )
        modelo_texto = st.sidebar.text_input(
            "Modelo (HF):",
            value=modelo_default,
        )
        ruta_local = st.sidebar.text_input(
            "Ruta local (opcional):",
            value="",
        )
        device = "cpu"

        semantic_config = {
            "backend": backend,
            "modelo": modelo_texto,
            "ruta_local": ruta_local.strip() or None,
            "device": device,
        }

    # =================
    # BOTON DE EJECUCION
    # =================
    st.sidebar.markdown("---")
    ejecutar = st.sidebar.button("Ejecutar Analisis", type="primary", use_container_width=True)

    # =================
    # AREA PRINCIPAL
    # =================

    if ejecutar:
        config = {
            "algoritmos": algoritmos_seleccionados,
            "umbrales": umbrales,
            "zonas_gris": zonas_gris,
        }
        if semantic_config:
            config["semantic_config"] = semantic_config

        # Mostrar configuracion
        with st.expander("Configuracion actual", expanded=False):
            st.json(config)

        # Cargar datos desde CSV
        with st.spinner("Cargando terminos preprocesados..."):
            todos_unicos, contador, total_valores = cargar_terminos_csv(archivo_csv)
            st.success(f"Cargados {len(todos_unicos):,} terminos unicos ({total_valores:,} ocurrencias)")

        # Cargar voces
        with st.spinner("Cargando lista de voces..."):
            voces_raw, voz_a_entidad = cargar_voces_lista(archivo_lista)
            voces_norm_map = {}
            for voz in voces_raw:
                voz_norm = normalize_name(voz)
                if voz_norm:
                    if voz_norm not in voces_norm_map:
                        voces_norm_map[voz_norm] = []
                    voces_norm_map[voz_norm].append(voz)
            voces_lista = list(voces_norm_map.keys())
            n_entidades = len(set(voz_a_entidad.values()))
            st.success(f"Cargadas {len(voces_lista)} voces normalizadas ({n_entidades} entidades)")

        # Filtrar voces por apariciones (0 = sin filtro, usar todas)
        apariciones_por_voz = {voz: contador.get(voz, 0) for voz in voces_lista}
        if min_apariciones_voz > 0:
            voces_filtradas = [v for v in voces_lista if apariciones_por_voz[v] >= min_apariciones_voz]
            st.info(f"Usando {len(voces_filtradas)} voces con >= {min_apariciones_voz} apariciones")
        else:
            voces_filtradas = voces_lista.copy()
            st.info(f"Usando todas las {len(voces_filtradas)} voces de la lista (sin filtro)")

        if len(voces_filtradas) == 0:
            st.error("No hay voces disponibles. Verifique el archivo de lista.")
            return

        # Preparar modelo semantico si aplica
        modelo_semantico = None
        if "Semantica_Embeddings" in algoritmos_seleccionados:
            st.subheader("Cargando modelo semantico...")
            try:
                modelo_semantico = cargar_modelo_embeddings(
                    backend=semantic_config["backend"],
                    modelo_texto=semantic_config["modelo"],
                    ruta_local=semantic_config["ruta_local"],
                    device=semantic_config["device"],
                )
                st.success(
                    f"Modelo listo: {modelo_semantico.backend} / {modelo_semantico.modelo}"
                )
            except Exception as exc:
                st.error(f"No se pudo cargar el modelo semantico: {exc}")
                return

        # Generar matrices
        st.subheader("Generando matrices de similitud...")
        matrices = {}

        for algo in algoritmos_seleccionados:
            st.write(f"  Procesando {algo}...")
            progress = st.progress(0)
            tipo = ALGORITMOS_DISPONIBLES[algo]["tipo"]
            if tipo == "semantic":
                matrices[algo] = generar_matriz_semantica(
                    todos_unicos,
                    voces_filtradas,
                    modelo_semantico,
                    progress,
                )
            else:
                matrices[algo] = generar_matriz_similitud_pairwise(
                    todos_unicos,
                    voces_filtradas,
                    ALGORITMOS_DISPONIBLES[algo]["func"],
                    progress,
                )

        st.success("Matrices generadas")

        # Clasificar
        with st.spinner("Clasificando terminos..."):
            df_clasificacion = clasificar_terminos(
                todos_unicos, contador, matrices, config, voz_a_entidad
            )

        # =================
        # RESULTADOS
        # =================
        st.header("Resultados")

        # Metricas principales
        col1, col2, col3, col4 = st.columns(4)

        freq_cons = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO"]["frecuencia"].sum()
        freq_debil = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"]["frecuencia"].sum()
        pct_cons = freq_cons / total_valores * 100 if total_valores > 0 else 0
        pct_total = (freq_cons + freq_debil) / total_valores * 100 if total_valores > 0 else 0

        with col1:
            st.metric("Total ocurrencias", f"{total_valores:,}")
        with col2:
            st.metric("Consensuado (estricto)", f"{pct_cons:.1f}%", f"{freq_cons:,} ocurrencias")
        with col3:
            st.metric("Consensuado + Debil", f"{pct_total:.1f}%", f"{freq_cons + freq_debil:,} ocurrencias")
        with col4:
            st.metric("Terminos unicos", f"{len(todos_unicos):,}")

        # Tabla de distribucion
        st.subheader("Distribucion por clasificacion")

        resumen_data = []
        for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
            df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
            n = len(df_cat)
            freq = df_cat["frecuencia"].sum()
            pct = freq / total_valores * 100 if total_valores > 0 else 0
            resumen_data.append({
                "Clasificacion": clasificacion,
                "Terminos": n,
                "Ocurrencias": freq,
                "Porcentaje": f"{pct:.2f}%"
            })

        st.dataframe(pd.DataFrame(resumen_data), use_container_width=True)

        # Preview de datos
        st.subheader("Vista previa de datos")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"
        ])

        with tab1:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO"].head(50)
            st.dataframe(df_show, use_container_width=True)
        with tab2:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"].head(50)
            st.dataframe(df_show, use_container_width=True)
        with tab3:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "SOLO_1_VOTO"].head(50)
            st.dataframe(df_show, use_container_width=True)
        with tab4:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "ZONA_GRIS"].head(50)
            st.dataframe(df_show, use_container_width=True)
        with tab5:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "RECHAZADO"].head(50)
            st.dataframe(df_show, use_container_width=True)

        # =================
        # DESCARGAS
        # =================
        st.subheader("Descargar resultados")

        # Preparar estadisticas para el informe
        stats = {
            "archivo_voces": campo_config["lista"],
            "total_valores": total_valores,
            "terminos_unicos": len(todos_unicos),
            "voces_raw": len(voces_lista),
            "voces_filtradas": len(voces_filtradas),
            "min_apariciones_voz": min_apariciones_voz,
        }

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV completo
            csv_completo = df_clasificacion.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Descargar CSV completo",
                data=csv_completo,
                file_name=f"clasificacion_completa_{campo_seleccionado}.csv",
                mime="text/csv"
            )

        with col2:
            # ZIP con todos los CSVs
            zip_data = crear_zip_csvs(df_clasificacion, campo_seleccionado, config, stats)
            st.download_button(
                label="Descargar todos los CSVs (ZIP)",
                data=zip_data,
                file_name=f"resultados_{campo_seleccionado}.zip",
                mime="application/zip"
            )

        with col3:
            # Informe TXT individual
            informe_txt = generar_informe_txt(df_clasificacion, campo_seleccionado, config, stats)
            st.download_button(
                label="Descargar informe TXT",
                data=informe_txt.encode("utf-8"),
                file_name=f"informe_{campo_seleccionado}.txt",
                mime="text/plain"
            )

        # Guardar en session state para persistencia
        st.session_state["df_clasificacion"] = df_clasificacion
        st.session_state["campo"] = campo_seleccionado
        st.session_state["config"] = config
        st.session_state["stats"] = stats

    # Mostrar resultados previos si existen
    elif "df_clasificacion" in st.session_state:
        st.info("Mostrando resultados del ultimo analisis. Presione 'Ejecutar Analisis' para recalcular.")


if __name__ == "__main__":
    main()

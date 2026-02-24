"""
Aplicacion Streamlit para desambiguacion de terminos historicos.
Version 3: integra similitud fonetica, semantica, de grafos.
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
import gzip
import shutil
import urllib.request
import importlib.util
import subprocess
import sys
from functools import lru_cache

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

try:
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel
    _TRANSFORMERS_DISPONIBLE = True
except ImportError:
    _TRANSFORMERS_DISPONIBLE = False

try:
    import torch as _torch
    _TORCH_DISPONIBLE = True
except ImportError:
    _TORCH_DISPONIBLE = False

try:
    import fasttext as _fasttext
    _FASTTEXT_DISPONIBLE = True
except ImportError:
    _FASTTEXT_DISPONIBLE = False

# FastText (Common Crawl) - modelo espanol por defecto
FASTTEXT_ES_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz"

# =============================================================================
# CONFIGURACION DE LA APP
# =============================================================================

st.set_page_config(
    page_title="Desambiguador de terminos v3",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _faltan_dependencias() -> List[str]:
    mods = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "abydos": "abydos",
        "jellyfish": "jellyfish",
        "metaphone": "metaphone",
        "rapidfuzz": "rapidfuzz",
        "sklearn": "scikit-learn",
        "sentence_transformers": "sentence-transformers",
        "text2vec": "text2vec",
        "torch": "torch",
        "fasttext": "fasttext",
        "transformers": "transformers",
    }
    faltan = []
    for modulo, paquete in mods.items():
        if importlib.util.find_spec(modulo) is None:
            faltan.append(paquete)
    return faltan


def asegurar_dependencias() -> None:
    if st.session_state.get("_deps_ok"):
        return
    faltan = _faltan_dependencias()
    if not faltan:
        st.session_state["_deps_ok"] = True
        return
    req_path = BASE_DIR / "requirements.txt"
    st.warning(
        "Faltan dependencias. Instalando desde requirements.txt..."
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
        )
        st.session_state["_deps_ok"] = True
        st.success("Dependencias instaladas. Reiniciando la app...")
        st.rerun()
    except Exception as exc:
        st.error(f"No se pudieron instalar dependencias: {exc}")
        st.stop()

# Directorio base
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@lru_cache(maxsize=50000)
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

    def similitud_cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Similitud coseno 0-1 entre embeddings ya calculados."""
        raw = float(np.dot(v1, v2))
        return round((raw + 1.0) / 2.0, 4)

    @property
    def dimension(self) -> int:
        if self._dim == 0 and self._cache:
            self._dim = next(iter(self._cache.values())).shape[0]
        return self._dim


class ModeloFastText:
    def __init__(self, ruta_local: str):
        if not _FASTTEXT_DISPONIBLE:
            raise ImportError("pip install fasttext")
        if not ruta_local:
            raise ValueError("Ruta local requerida para FastText")
        self.ruta_local = ruta_local
        self._modelo = _fasttext.load_model(ruta_local)
        self._dim = int(self._modelo.get_dimension())
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, textos: List[str]) -> np.ndarray:
        nuevos = [t for t in textos if t not in self._cache]
        for t in nuevos:
            v = np.array(self._modelo.get_sentence_vector(t), dtype=np.float32)
            norm = np.linalg.norm(v)
            v = v / max(norm, 1e-10)
            self._cache[t] = v
        return np.array([self._cache[t] for t in textos])

    @property
    def dimension(self) -> int:
        return self._dim


class ModeloByT5:
    def __init__(self, modelo_texto: str, device: str = "cpu"):
        if not _TRANSFORMERS_DISPONIBLE:
            raise ImportError("pip install transformers")
        if not modelo_texto:
            raise ValueError("Modelo requerido para ByT5")
        self.modelo_texto = modelo_texto
        self.device = device
        self._tokenizer = _AutoTokenizer.from_pretrained(modelo_texto)
        self._modelo = _AutoModel.from_pretrained(modelo_texto).to(device)
        self._modelo.eval()
        self._dim = 0
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, textos: List[str], batch_size: int = 32) -> np.ndarray:
        nuevos = [t for t in textos if t not in self._cache]
        if nuevos:
            for i in range(0, len(nuevos), batch_size):
                batch = nuevos[i:i + batch_size]
                inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with _torch.no_grad():
                    outputs = self._modelo.encoder(**inputs)
                vecs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.maximum(norms, 1e-10)
                if self._dim == 0:
                    self._dim = vecs.shape[1]
                for t, v in zip(batch, vecs):
                    self._cache[t] = v
        return np.array([self._cache[t] for t in textos])

    @property
    def dimension(self) -> int:
        return self._dim


def descargar_fasttext_es_si_falta(ruta_destino: Path, progress_bar=None) -> None:
    if ruta_destino.exists():
        return
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    ruta_gz = ruta_destino.with_suffix(ruta_destino.suffix + ".gz")

    if not ruta_gz.exists():
        with urllib.request.urlopen(FASTTEXT_ES_URL) as resp:
            total = resp.length or 0
            descargado = 0
            with open(ruta_gz, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    descargado += len(chunk)
                    if progress_bar and total > 0:
                        progress_bar.progress(min(descargado / total, 1.0))

    with gzip.open(ruta_gz, "rb") as f_in:
        with open(ruta_destino, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


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


@st.cache_resource(show_spinner=False)
def cargar_modelo_fasttext(ruta_local: str) -> ModeloFastText:
    ruta = Path(ruta_local)
    descargar_fasttext_es_si_falta(ruta)
    return ModeloFastText(ruta_local=str(ruta))


@st.cache_resource(show_spinner=False)
def cargar_modelo_byt5(modelo_texto: str, device: str) -> ModeloByT5:
    return ModeloByT5(modelo_texto=modelo_texto, device=device)


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


def construir_indice_voces(voces: List[str]) -> Dict[Tuple[str, int], List[str]]:
    indice = {}
    for voz in voces:
        norm = normalize_name(voz)
        if not norm:
            continue
        pref = norm[:2] if len(norm) >= 2 else norm
        key = (pref, len(norm))
        if key not in indice:
            indice[key] = []
        indice[key].append(voz)
    return indice


def candidatos_para_termino(
    termino: str,
    indice_voces: Dict[Tuple[str, int], List[str]],
    voces: List[str],
    max_candidatos: int = 600,
    ventana_len: int = 2,
) -> List[str]:
    norm = normalize_name(termino)
    if not norm:
        return voces
    pref = norm[:2] if len(norm) >= 2 else norm
    base_len = len(norm)
    candidatos = []
    for ln in range(max(1, base_len - ventana_len), base_len + ventana_len + 1):
        candidatos.extend(indice_voces.get((pref, ln), []))
    if not candidatos:
        return voces
    if len(candidatos) > max_candidatos:
        return candidatos[:max_candidatos]
    return candidatos


def calcular_mejores_pairwise(
    terminos: List[str],
    voces: List[str],
    funcion_similitud,
    progress_bar=None,
    usar_bloqueo: bool = True,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    mejores_sim = {}
    mejores_voz = {}
    indice_voces = construir_indice_voces(voces) if usar_bloqueo else None
    total = len(terminos)
    for i, termino in enumerate(terminos):
        if indice_voces is not None:
            candidatos = candidatos_para_termino(termino, indice_voces, voces)
        else:
            candidatos = voces
        best_sim = -1.0
        best_voz = ""
        for voz in candidatos:
            sim = funcion_similitud(termino, voz)
            if sim > best_sim:
                best_sim = sim
                best_voz = voz
        mejores_sim[termino] = best_sim if best_sim >= 0 else 0.0
        mejores_voz[termino] = best_voz
        if progress_bar:
            progress_bar.progress((i + 1) / total)
    return mejores_sim, mejores_voz


def mejores_desde_matriz(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, str]]:
    mejores_sim = df.max(axis=1).to_dict()
    mejores_voz = df.idxmax(axis=1).to_dict()
    return mejores_sim, mejores_voz


def semantic_similarity(
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


def fasttext_similarity(
    terminos: List[str],
    voces: List[str],
    modelo: ModeloFastText,
    progress_bar=None,
) -> pd.DataFrame:
    return semantic_similarity(terminos, voces, modelo, progress_bar)


# Diccionario de algoritmos disponibles
ALGORITMOS_DISPONIBLES = {
    "Lev_OCR": {"tipo": "pairwise", "func": levenshtein_ratio_ocr},
    "Lev_Ratio": {"tipo": "pairwise", "func": levenshtein_ratio},
    "Jaro_Winkler": {"tipo": "pairwise", "func": jaro_winkler_similarity},
    "NGram_2": {"tipo": "pairwise", "func": ngram_similarity_2},
    "NGram_3": {"tipo": "pairwise", "func": ngram_similarity_3},
    "Fonetica_DM": {"tipo": "pairwise", "func": phonetic_similarity},
    "Semantica": {"tipo": "semantic", "func": semantic_similarity},
    "FastText": {"tipo": "fasttext", "func": fasttext_similarity},
    "ByT5": {"tipo": "byt5", "func": semantic_similarity},
}

# Umbrales por defecto
UMBRALES_DEFAULT = {
    "Lev_OCR": 0.75,
    "Lev_Ratio": 0.75,
    "Jaro_Winkler": 0.85,
    "NGram_2": 0.66,
    "NGram_3": 0.60,
    "Fonetica_DM": 0.85,
    "Semantica": 0.975,
    "FastText": 0.85,
    "ByT5": 0.90,
}

ZONAS_GRIS_DEFAULT = {
    "Lev_OCR": (0.71, 0.749),
    "Lev_Ratio": (0.71, 0.749),
    "Jaro_Winkler": (0.80, 0.849),
    "NGram_2": (0.63, 0.659),
    "NGram_3": (0.55, 0.599),
    "Fonetica_DM": (0.80, 0.849),
    "Semantica": (0.965, 0.974),
    "FastText": (0.80, 0.849),
    "ByT5": (0.86, 0.89),
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
    resultados_algo: Dict[str, Dict[str, Dict[str, object]]],
    config: Dict,
    voz_a_entidad: Dict[str, str],
    voces_disponibles: List[str],
) -> pd.DataFrame:
    """Clasifica terminos segun configuracion."""

    algoritmos = config["algoritmos"]
    umbrales = config["umbrales"]
    zonas_gris = config["zonas_gris"]
    requiere_lev_ocr = "Lev_OCR" in algoritmos

    datos = []

    for termino in todos_unicos:
        frecuencia = contador_frecuencias.get(termino, 0)
        fila = {
            "termino": termino,
            "frecuencia": frecuencia,
            "exact_match": False,
            "voz_exacta": "",
        }

        if termino in voces_disponibles:
            entidad = voz_a_entidad.get(termino, termino)
            fila["exact_match"] = True
            fila["voz_exacta"] = termino
            fila["entidad"] = entidad
            fila["voz"] = termino
            fila["votos_aprobacion"] = 0
            fila["votos_entidad_consenso"] = 0
            fila["LevOCR"] = "Lev_OCR" in algoritmos
            fila["clasificacion"] = "CONSENSUADO"
            datos.append(fila)
            continue

        votos_aprobacion = 0
        en_zona_gris = False
        # entidad -> [(algoritmo, voz)]
        votos_por_entidad = {}
        voz_lev_ocr = None  # la voz que voto Levenshtein_OCR

        for nombre_algo in algoritmos:
            if nombre_algo not in resultados_algo:
                continue

            res = resultados_algo[nombre_algo]
            max_sim = res["sim"].get(termino, 0.0)
            mejor_voz = res["voz"].get(termino, "")
            if not mejor_voz and max_sim == 0.0:
                continue

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

                if nombre_algo == "Lev_OCR":
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
            lev_ocr_en_consenso = "Lev_OCR" in algoritmos_consenso

            # La voz consenso es la que voto Levenshtein_OCR (si esta en consenso)
            if lev_ocr_en_consenso:
                voz_consenso = voz_lev_ocr
            else:
                # Si Lev-OCR no esta, tomar la primera voz de la entidad consensuada
                voz_consenso = votos_por_entidad[entidad_consenso][0][1]

        fila["votos_aprobacion"] = votos_aprobacion
        fila["entidad"] = entidad_consenso
        fila["voz"] = voz_consenso
        fila["votos_entidad_consenso"] = votos_entidad
        fila["LevOCR"] = lev_ocr_en_consenso

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

    regla = "Match exacto => CONSENSUADO. Si no, 2+ votos por misma entidad"
    if "Lev_OCR" in config.get("algoritmos", []):
        regla += " + Lev_OCR en consenso"
    lineas.append(f"Regla de consenso: {regla}")
    lineas.append("")
    lineas.append("Umbrales de aprobacion:")
    for algo, umbral in config["umbrales"].items():
        lineas.append(f"  - {algo}: {umbral}")
    lineas.append("")
    lineas.append("Zonas grises (margen bajo umbral):")
    for algo, zona in config["zonas_gris"].items():
        piso, techo = zona
        umbral = config["umbrales"].get(algo, 0.0)
        margen = max(0.0, umbral - piso)
        lineas.append(f"  - {algo}: margen={margen:.2f} (piso={piso:.2f}, umbral={techo:.2f})")
    lineas.append("")

    if "semantic_config" in config:
        sem = config["semantic_config"]
        lineas.append("Configuracion semantica:")
        lineas.append(f"  - backend: {sem.get('backend', 'N/A')}")
        lineas.append(f"  - modelo: {sem.get('modelo', 'N/A')}")
        if sem.get("ruta_local"):
            lineas.append(f"  - ruta_local: {sem.get('ruta_local')}")
        lineas.append("")

    if "fasttext_config" in config:
        ft = config["fasttext_config"]
        lineas.append("Configuracion FastText:")
        lineas.append(f"  - ruta_local: {ft.get('ruta_local', 'N/A')}")
        lineas.append("")

    if "byt5_config" in config:
        bt = config["byt5_config"]
        lineas.append("Configuracion ByT5:")
        lineas.append(f"  - modelo: {bt.get('modelo', 'N/A')}")
        lineas.append("")

    # Resultados por clasificacion
    lineas.append("-" * 70)
    lineas.append("RESULTADOS POR CLASIFICACION")
    lineas.append("-" * 70)

    total_valores = stats.get("total_valores", 1)
    total_terminos = len(df_clasificacion)

    lineas.append(f"{'Clasificacion':<20} {'Terminos':>10} {'Ocurrencias':>12} {'Porcentaje':>10}")
    lineas.append("-" * 54)

    df_exacto = df_clasificacion[df_clasificacion["exact_match"]]
    n_exacto = len(df_exacto)
    ocurrencias_exacto = df_exacto["frecuencia"].sum()
    pct_exacto = ocurrencias_exacto / total_valores * 100 if total_valores > 0 else 0
    lineas.append(f"{'EXACTO':<20} {n_exacto:>10,} {ocurrencias_exacto:>12,} {pct_exacto:>9.2f}%")

    for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
        df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
        if clasificacion == "CONSENSUADO":
            df_cat = df_cat[~df_cat["exact_match"]]
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

    freq_cons = df_clasificacion[
        (df_clasificacion["clasificacion"] == "CONSENSUADO")
        & (~df_clasificacion["exact_match"])
    ]["frecuencia"].sum()
    freq_debil = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"]["frecuencia"].sum()
    freq_exacto = df_clasificacion[df_clasificacion["exact_match"]]["frecuencia"].sum()
    pct_cons = freq_cons / total_valores * 100 if total_valores > 0 else 0
    pct_total = (freq_cons + freq_debil) / total_valores * 100 if total_valores > 0 else 0

    lineas.append(f"Exacto:               {freq_exacto:,} de {total_valores:,} ({(freq_exacto / total_valores * 100) if total_valores > 0 else 0:.2f}%)")
    lineas.append(f"Consensuado (estricto): {freq_cons:,} de {total_valores:,} ({pct_cons:.2f}%)")
    lineas.append(f"Consensuado + Debil:    {freq_cons + freq_debil:,} de {total_valores:,} ({pct_total:.2f}%)")
    lineas.append("")

    # Archivos generados
    lineas.append("-" * 70)
    lineas.append("ARCHIVOS GENERADOS")
    lineas.append("-" * 70)
    lineas.append(f"- clasificacion_completa_{nombre_campo}.csv")
    if len(df_exacto) > 0:
        lineas.append(f"- exacto_{nombre_campo}.csv")
    for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
        df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
        if clasificacion == "CONSENSUADO":
            df_cat = df_cat[~df_cat["exact_match"]]
        if len(df_cat) > 0:
            lineas.append(f"- {clasificacion.lower()}_{nombre_campo}.csv")
    lineas.append(f"- informe_{nombre_campo}.txt")
    lineas.append("")

    lineas.append("=" * 70)
    lineas.append("FIN DEL INFORME")
    lineas.append("=" * 70)

    return "\n".join(lineas)


def preparar_tabla_salida(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    # Unificar voz + score por algoritmo en una sola columna (nombre simplificado)
    mapeo_algos = {
        "Lev_OCR": "LevOCR",
        "Jaro_Winkler": "Jaro",
        "NGram_2": "NGram2",
        "NGram_3": "NGram3",
        "NGram_4": "NGram4",
        "DoubleMetaphone": "DM",
        "Soundex": "Soundex",
        "Metaphone": "Metaphone",
        "Phonetic": "Phonetic",
        "Cosine": "Cosine",
        "Semantica": "Semantica",
        "FastText": "FastText",
        "ByT5": "ByT5",
    }
    for algo, nombre_corto in mapeo_algos.items():
        col_sim = f"sim_{algo}"
        col_voz = f"voz_{algo}"
        if col_sim in df_out.columns and col_voz in df_out.columns:
            def _fmt(v):
                try:
                    return f"{float(v):.3f}"
                except Exception:
                    return ""
            df_out[nombre_corto] = (
                df_out[col_voz].astype(str).fillna("") + ":" + df_out[col_sim].apply(_fmt)
            ).str.rstrip(":")
            df_out = df_out.drop(columns=[col_sim, col_voz], errors="ignore")
        else:
            if col_sim in df_out.columns:
                df_out = df_out.drop(columns=[col_sim], errors="ignore")
            if col_voz in df_out.columns:
                df_out = df_out.drop(columns=[col_voz], errors="ignore")

    if "votos_entidad_consenso" in df_out.columns:
        df_out["votos"] = df_out["votos_entidad_consenso"]
    elif "votos_aprobacion" in df_out.columns:
        df_out["votos"] = df_out["votos_aprobacion"]
    df_out = df_out.drop(
        columns=[
            "exact_match",
            "voz_exacta",
            "votos_aprobacion",
            "votos_entidad_consenso",
            "clasificacion",
            "Lev_OCR_en_consenso",
        ],
        errors="ignore",
    )
    return df_out


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
        csv_completo = preparar_tabla_salida(df_clasificacion).to_csv(
            index=False, encoding="utf-8-sig"
        )
        zf.writestr(f"clasificacion_completa_{nombre_campo}.csv", csv_completo)

        # CSV exacto
        df_exacto = df_clasificacion[df_clasificacion["exact_match"]]
        if len(df_exacto) > 0:
            csv_exacto = preparar_tabla_salida(df_exacto).to_csv(index=False, encoding="utf-8-sig")
            zf.writestr(f"exacto_{nombre_campo}.csv", csv_exacto)

        # CSVs por clasificacion
        for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
            df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
            if clasificacion == "CONSENSUADO":
                df_cat = df_cat[~df_cat["exact_match"]]
            if len(df_cat) > 0:
                csv_cat = preparar_tabla_salida(df_cat).to_csv(index=False, encoding="utf-8-sig")
                nombre_archivo = clasificacion.lower() + f"_{nombre_campo}.csv"
                zf.writestr(nombre_archivo, csv_cat)

        # Informe TXT
        if config and stats:
            informe = generar_informe_txt(df_clasificacion, nombre_campo, config, stats)
            zf.writestr(f"informe_{nombre_campo}.txt", informe.encode("utf-8"))

    buffer.seek(0)
    return buffer.getvalue()


class OverallProgress:
    def __init__(self, total_units: float, progress_bar) -> None:
        self.total_units = max(total_units, 1.0)
        self.done = 0.0
        self.progress_bar = progress_bar

    def advance(self, units: float) -> None:
        self.done += units
        self.progress_bar.progress(min(self.done / self.total_units, 1.0))


class AlgoProgressProxy:
    def __init__(self, tracker: OverallProgress, algo_units: float) -> None:
        self.tracker = tracker
        self.algo_units = algo_units
        self.last = 0.0

    def progress(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        delta = value - self.last
        if delta <= 0:
            return
        self.last = value
        self.tracker.advance(delta * self.algo_units)


# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

def main():
    st.title("Desambiguador de terminos historicos (v3)")
    asegurar_dependencias()

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
        default=["Lev_OCR", "Jaro_Winkler", "NGram_2"]
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
        default_piso, _default_techo = ZONAS_GRIS_DEFAULT.get(algo, (0.6, 0.69))
        default_umbral = UMBRALES_DEFAULT.get(algo, 0.7)
        default_margen = max(0.0, default_umbral - default_piso)
        margen = st.sidebar.number_input(
            f"{algo} margen bajo umbral:",
            min_value=0.0,
            max_value=1.0,
            value=default_margen,
            step=0.01,
            key=f"zg_margen_{algo}",
        )
        piso = max(0.0, umbrales[algo] - margen)
        techo = umbrales[algo]
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
    if "Semantica" in algoritmos_seleccionados:
        st.sidebar.subheader("6. Semantica")
        modelos_presets = [
            ("sentence-transformers/LaBSE", "sentence_transformers"),
            ("shibing624/text2vec-base-multilingual", "text2vec"),
        ]

        preset_default = "sentence-transformers/LaBSE"
        opciones_modelos = [m for m, _ in modelos_presets]
        if preset_default not in opciones_modelos:
            preset_default = opciones_modelos[0]
        opciones_preset = opciones_modelos + ["Personalizado"]
        preset_modelo = st.sidebar.selectbox(
            "Modelo:",
            options=opciones_preset,
            index=opciones_preset.index(preset_default),
        )
        if preset_modelo == "Personalizado":
            modelo_texto = st.sidebar.text_input(
                "Modelo (HF):",
                value=preset_default,
            )
            texto_norm = modelo_texto.lower()
            if "text2vec" in texto_norm:
                backend = "text2vec"
            elif "sentence-transformers" in texto_norm or "paraphrase-" in texto_norm or "distiluse-" in texto_norm:
                backend = "sentence_transformers"
            else:
                backend = "sentence_transformers"
        else:
            modelo_texto = preset_modelo
            backend = next(b for m, b in modelos_presets if m == preset_modelo)
        ruta_local = st.sidebar.text_input(
            "Ruta local (opcional):",
            value="",
        )
        device = "cpu"
        if _TORCH_DISPONIBLE and _torch.cuda.is_available():
            device = "cuda"
        st.sidebar.caption(f"Device detectado: {device}")

        semantic_config = {
            "backend": backend,
            "modelo": modelo_texto,
            "ruta_local": ruta_local.strip() or None,
            "device": device,
        }

    # 7. Configuracion FastText (solo si se usa)
    fasttext_config = None
    if "FastText" in algoritmos_seleccionados:
        st.sidebar.subheader("7. FastText")
        st.sidebar.caption("Descarga automatica si no existe el modelo local.")
        ruta_default_ft = str(BASE_DIR / "modelos" / "fasttext" / "cc.es.300.bin")
        ruta_fasttext = st.sidebar.text_input(
            "Ruta modelo FastText:",
            value=ruta_default_ft,
        )
        fasttext_config = {
            "ruta_local": ruta_fasttext.strip() or None,
        }

    # 8. Configuracion ByT5 (solo si se usa)
    byt5_config = None
    if "ByT5" in algoritmos_seleccionados:
        st.sidebar.subheader("8. ByT5")
        device_default = "cpu"
        if _TORCH_DISPONIBLE and _torch.cuda.is_available():
            device_default = "cuda"
        st.sidebar.caption(f"Device detectado: {device_default}")
        modelo_byt5 = st.sidebar.text_input(
            "Modelo ByT5 (HF):",
            value="google/byt5-small",
        )
        byt5_config = {
            "modelo": modelo_byt5.strip() or "google/byt5-small",
            "device": device_default,
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
        if fasttext_config:
            config["fasttext_config"] = fasttext_config
        if byt5_config:
            config["byt5_config"] = byt5_config

        # Mostrar configuracion
        with st.expander("Configuracion actual", expanded=False):
            st.json(config)

        # Cargar datos desde CSV
        with st.spinner("Cargando terminos preprocesados..."):
            todos_unicos, contador, total_valores = cargar_terminos_csv(archivo_csv)
            msg_terminos = (
                f"Cargados {len(todos_unicos):,} terminos unicos "
                f"({total_valores:,} ocurrencias)"
            )

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
            msg_voces = f"Cargadas {len(voces_lista)} voces normalizadas ({n_entidades} entidades)"

        # Filtrar voces por apariciones (0 = sin filtro, usar todas)
        apariciones_por_voz = {voz: contador.get(voz, 0) for voz in voces_lista}
        if min_apariciones_voz > 0:
            voces_filtradas = [v for v in voces_lista if apariciones_por_voz[v] >= min_apariciones_voz]
            msg_uso = f"Usando {len(voces_filtradas)} voces con >= {min_apariciones_voz} apariciones"
        else:
            voces_filtradas = voces_lista.copy()
            msg_uso = f"Usando {len(voces_filtradas)} voces de la lista (sin filtro)"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(msg_terminos)
        with col2:
            st.success(msg_voces)
        with col3:
            st.info(msg_uso)

        if len(voces_filtradas) == 0:
            st.error("No hay voces disponibles. Verifique el archivo de lista.")
            return

        # Preparar modelo semantico si aplica
        modelo_semantico = None
        if "Semantica" in algoritmos_seleccionados:
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

        # Preparar modelo FastText si aplica
        modelo_fasttext = None
        if "FastText" in algoritmos_seleccionados:
            st.subheader("Cargando modelo FastText...")
            if not fasttext_config or not fasttext_config.get("ruta_local"):
                st.error("Ruta de modelo FastText requerida.")
                return
            if not _FASTTEXT_DISPONIBLE:
                st.error("fasttext no esta instalado. Instala la dependencia para usar este enfoque.")
                return
            try:
                ruta_ft = Path(fasttext_config["ruta_local"])
                if not ruta_ft.exists():
                    st.info("Descargando modelo FastText (primera vez)...")
                    progress_ft = st.progress(0)
                    descargar_fasttext_es_si_falta(ruta_ft, progress_ft)
                    progress_ft.empty()
                modelo_fasttext = cargar_modelo_fasttext(
                    ruta_local=fasttext_config["ruta_local"]
                )
                st.success(f"Modelo FastText listo: {fasttext_config['ruta_local']}")
            except Exception as exc:
                st.error(f"No se pudo cargar el modelo FastText: {exc}")
                return

        # Preparar modelo ByT5 si aplica
        modelo_byt5 = None
        if "ByT5" in algoritmos_seleccionados:
            st.subheader("Cargando modelo ByT5...")
            if not _TRANSFORMERS_DISPONIBLE:
                st.error("transformers no esta instalado. Instala la dependencia para usar ByT5.")
                return
            if not _TORCH_DISPONIBLE:
                st.error("torch no esta instalado. Instala la dependencia para usar ByT5.")
                return
            try:
                modelo_byt5 = cargar_modelo_byt5(
                    modelo_texto=byt5_config["modelo"],
                    device=byt5_config["device"],
                )
                st.success(f"Modelo ByT5 listo: {byt5_config['modelo']}")
            except Exception as exc:
                st.error(f"No se pudo cargar el modelo ByT5: {exc}")
                return

        # Generar mejores coincidencias (barra de progreso general)
        resultados_algo = {}

        total_units = 0
        for algo in algoritmos_seleccionados:
            tipo = ALGORITMOS_DISPONIBLES[algo]["tipo"]
            if tipo == "pairwise":
                total_units += len(todos_unicos)
            else:
                total_units += 2

        overall_bar = st.progress(0)
        tracker = OverallProgress(total_units, overall_bar)

        for algo in algoritmos_seleccionados:
            tipo = ALGORITMOS_DISPONIBLES[algo]["tipo"]
            if tipo == "pairwise":
                progress = AlgoProgressProxy(tracker, len(todos_unicos))
            else:
                progress = AlgoProgressProxy(tracker, 2)
            func = ALGORITMOS_DISPONIBLES[algo]["func"]
            if tipo == "semantic":
                df_sim = func(
                    todos_unicos,
                    voces_filtradas,
                    modelo_semantico,
                    progress,
                )
                sim, voz = mejores_desde_matriz(df_sim)
                resultados_algo[algo] = {"sim": sim, "voz": voz}
            elif tipo == "fasttext":
                df_sim = func(
                    todos_unicos,
                    voces_filtradas,
                    modelo_fasttext,
                    progress,
                )
                sim, voz = mejores_desde_matriz(df_sim)
                resultados_algo[algo] = {"sim": sim, "voz": voz}
            elif tipo == "byt5":
                df_sim = func(
                    todos_unicos,
                    voces_filtradas,
                    modelo_byt5,
                    progress,
                )
                sim, voz = mejores_desde_matriz(df_sim)
                resultados_algo[algo] = {"sim": sim, "voz": voz}
            else:
                usar_bloqueo = len(voces_filtradas) >= 800
                sim, voz = calcular_mejores_pairwise(
                    todos_unicos,
                    voces_filtradas,
                    func,
                    progress,
                    usar_bloqueo=usar_bloqueo,
                )
                resultados_algo[algo] = {"sim": sim, "voz": voz}

        overall_bar.empty()

        # Clasificar
        with st.spinner("Clasificando terminos..."):
            df_clasificacion = clasificar_terminos(
                todos_unicos, contador, resultados_algo, config, voz_a_entidad, voces_filtradas
            )

        # =================
        # RESULTADOS
        # =================
        st.header("Resultados")

        # Metricas principales
        col1, col2, col3, col4, col5 = st.columns(5)

        if "exact_match" not in df_clasificacion.columns:
            df_clasificacion["exact_match"] = False

        freq_exacto = df_clasificacion[df_clasificacion["exact_match"]]["frecuencia"].sum()
        freq_cons_no_exacto = df_clasificacion[
            (df_clasificacion["clasificacion"] == "CONSENSUADO")
            & (~df_clasificacion["exact_match"])
        ]["frecuencia"].sum()
        freq_debil = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"]["frecuencia"].sum()

        pct_exacto = freq_exacto / total_valores * 100 if total_valores > 0 else 0
        pct_cons_acum = (freq_exacto + freq_cons_no_exacto) / total_valores * 100 if total_valores > 0 else 0
        pct_total_acum = (freq_exacto + freq_cons_no_exacto + freq_debil) / total_valores * 100 if total_valores > 0 else 0

        with col1:
            st.metric("Total ocurrencias", f"{total_valores:,}")
        with col2:
            st.metric("Exacto", f"{pct_exacto:.1f}%", f"{freq_exacto:,} ocurrencias")
        with col3:
            st.metric(
                "Consensuado (estricto)",
                f"{pct_cons_acum:.1f}%",
                f"{freq_exacto + freq_cons_no_exacto:,} ocurrencias",
            )
        with col4:
            st.metric(
                "Consensuado + Debil",
                f"{pct_total_acum:.1f}%",
                f"{freq_exacto + freq_cons_no_exacto + freq_debil:,} ocurrencias",
            )
        with col5:
            st.metric("Terminos unicos", f"{len(todos_unicos):,}")

        # Tabla de distribucion
        st.subheader("Distribucion por clasificacion")

        if "exact_match" not in df_clasificacion.columns:
            df_clasificacion["exact_match"] = False

        resumen_data = []
        df_exacto = df_clasificacion[df_clasificacion["exact_match"]]
        resumen_data.append({
            "Clasificacion": "EXACTO",
            "Terminos": len(df_exacto),
            "Ocurrencias": int(df_exacto["frecuencia"].sum()),
            "Porcentaje": (
                f"{(df_exacto['frecuencia'].sum() / total_valores * 100) if total_valores > 0 else 0:.2f}%"
            ),
        })
        for clasificacion in ["CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"]:
            df_cat = df_clasificacion[df_clasificacion["clasificacion"] == clasificacion]
            if clasificacion == "CONSENSUADO":
                df_cat = df_cat[~df_cat["exact_match"]]
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

        if "exact_match" not in df_clasificacion.columns:
            df_clasificacion["exact_match"] = False

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "EXACTO", "CONSENSUADO", "CONSENSUADO_DEBIL", "SOLO_1_VOTO", "ZONA_GRIS", "RECHAZADO"
        ])

        with tab1:
            df_show = df_clasificacion[df_clasificacion["exact_match"]]
            df_show = df_show[["termino", "frecuencia", "entidad"]].head(50)
            df_show = df_show.rename(columns={"entidad": "entidad"})
            st.dataframe(preparar_tabla_salida(df_show), use_container_width=True)
        with tab2:
            df_show = df_clasificacion[
                (df_clasificacion["clasificacion"] == "CONSENSUADO")
                & (~df_clasificacion["exact_match"])
            ].head(50)
            st.dataframe(preparar_tabla_salida(df_show), use_container_width=True)
        with tab3:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "CONSENSUADO_DEBIL"].head(50)
            df_show = preparar_tabla_salida(df_show).drop(columns=["votos"], errors="ignore")
            st.dataframe(df_show, use_container_width=True)
        with tab4:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "SOLO_1_VOTO"].head(50)
            df_show = preparar_tabla_salida(df_show).drop(columns=["votos"], errors="ignore")
            st.dataframe(df_show, use_container_width=True)
        with tab5:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "ZONA_GRIS"].head(50)
            df_show = preparar_tabla_salida(df_show).drop(
                columns=["entidad", "entidad_consenso", "voz_consenso", "votos"],
                errors="ignore",
            )
            st.dataframe(df_show, use_container_width=True)
        with tab6:
            df_show = df_clasificacion[df_clasificacion["clasificacion"] == "RECHAZADO"].head(50)
            df_show = preparar_tabla_salida(df_show).drop(
                columns=["entidad", "entidad_consenso", "voz_consenso", "votos"],
                errors="ignore",
            )
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
            csv_completo = preparar_tabla_salida(df_clasificacion).to_csv(
                index=False, encoding="utf-8-sig"
            )
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

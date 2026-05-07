
"""
AxiLattice Pro v3.0
Voice I/O + Cube Architecture + Insight Observability
"Works on Any Data" — Production-Ready Proof of Concept
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import hashlib
import base64
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# VOICE I/O — REAL IMPLEMENTATION
# ============================================================

class VoiceManager:
    """Real voice pipeline with browser APIs + local models."""

    def __init__(self):
        self.stt_available = self._check_stt()
        self.tts_available = self._check_tts()

    def _check_stt(self) -> bool:
        try:
            import speech_recognition as sr
            return True
        except ImportError:
            return False

    def _check_tts(self) -> bool:
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        if not self.stt_available:
            return "[Install SpeechRecognition: pip install SpeechRecognition]"
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                    return recognizer.recognize_google(audio)
        except Exception as e:
            return f"[STT Error: {str(e)}]"

    def synthesize_speech(self, text: str) -> Optional[bytes]:
        if not self.tts_available:
            return None
        try:
            from gtts import gTTS
            tts = gTTS(text=text[:500], lang='en', slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp.read()
        except:
            return None

    def get_audio_html(self, audio_bytes: bytes) -> str:
        b64 = base64.b64encode(audio_bytes).decode()
        return f'<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'


# ============================================================
# CUBE ARCHITECTURE — REAL IMPLEMENTATION
# ============================================================

@dataclass
class CubeDimension:
    name: str
    column: str
    levels: List[str]
    cardinality: int

@dataclass
class CubeMeasure:
    name: str
    column: str
    aggregations: List[str]
    format: str = ".2f"

@dataclass
class Cuboid:
    dimensions: Tuple[str, ...]
    df: pd.DataFrame
    row_count: int
    created_at: datetime

class DataCube:
    """OLAP-style cube with pre-aggregation at multiple granularities."""

    def __init__(self, df: pd.DataFrame, profiler):
        self.df = df
        self.profiler = profiler
        self.dimensions: List[CubeDimension] = []
        self.measures: List[CubeMeasure] = []
        self.cuboids: Dict[str, Cuboid] = {}
        self._build_schema()
        self._precompute_cuboids()

    def _build_schema(self):
        for col in self.profiler.categorical_columns:
            unique_vals = self.df[col].nunique()
            self.dimensions.append(CubeDimension(
                name=col, column=col, levels=[col], cardinality=unique_vals
            ))

        if self.profiler.temporal_column:
            dt_series = self.profiler.construct_temporal()
            if dt_series is not None:
                self.dimensions.append(CubeDimension(
                    name="Time", column=self.profiler.temporal_column,
                    levels=["Year", "Quarter", "Month", self.profiler.temporal_column],
                    cardinality=dt_series.nunique()
                ))

        for col in self.profiler.metric_columns:
            self.measures.append(CubeMeasure(
                name=col, column=col,
                aggregations=["sum", "avg", "count", "min", "max", "std"]
            ))

    def _precompute_cuboids(self):
        dim_names = [d.name for d in self.dimensions]

        for dim in dim_names:
            self._create_cuboid((dim,))

        for i, d1 in enumerate(dim_names):
            for d2 in dim_names[i+1:]:
                card1 = next(d.cardinality for d in self.dimensions if d.name == d1)
                card2 = next(d.cardinality for d in self.dimensions if d.name == d2)
                if card1 * card2 < 100000:
                    self._create_cuboid((d1, d2))

        self._create_cuboid(())

    def _create_cuboid(self, dimensions: Tuple[str, ...]):
        cuboid_key = "_".join(dimensions) if dimensions else "total"
        dim_cols = [next(d.column for d in self.dimensions if d.name == dim_name) for dim_name in dimensions]

        agg_dict = {}
        for measure in self.measures:
            for agg in measure.aggregations:
                agg_dict[f"{measure.name}_{agg}"] = (measure.column, agg)

        if dim_cols:
            grouped = self.df.groupby(dim_cols, observed=True)
            cuboid_df = grouped.agg(**agg_dict).reset_index()
        else:
            cuboid_df = pd.DataFrame({
                f"{m.name}_{agg}": [self.df[m.column].agg(agg)]
                for m in self.measures for agg in m.aggregations
            })

        self.cuboids[cuboid_key] = Cuboid(
            dimensions=dimensions, df=cuboid_df,
            row_count=len(cuboid_df), created_at=datetime.now()
        )

    def query(self, dimensions: List[str], measures: List[str], 
              aggregations: List[str], filters: Optional[Dict] = None) -> pd.DataFrame:
        cuboid_key = "_".join(sorted(dimensions)) if dimensions else "total"

        if cuboid_key in self.cuboids:
            result = self.cuboids[cuboid_key].df.copy()
            if filters:
                for col, val in filters.items():
                    if col in result.columns:
                        result = result[result[col] == val]
            measure_cols = [f"{m}_{a}" for m, a in zip(measures, aggregations)]
            available_cols = [c for c in measure_cols if c in result.columns]
            if available_cols:
                return result[dimensions + available_cols]

        return self._query_raw(dimensions, measures, aggregations, filters)

    def _query_raw(self, dimensions, measures, aggregations, filters):
        df = self.df.copy()
        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    df = df[df[col] == val]
        if dimensions:
            agg_dict = {m: (m, a) for m, a in zip(measures, aggregations)}
            return df.groupby(dimensions, observed=True).agg(**agg_dict).reset_index()
        else:
            return pd.DataFrame({f"{m}_{a}": [df[m].agg(a)] for m, a in zip(measures, aggregations)})

    def drill_down(self, current_dims: List[str], new_dim: str) -> pd.DataFrame:
        return self.query(current_dims + [new_dim], [self.measures[0].name], ["sum"])

    def roll_up(self, current_dims: List[str], remove_dim: str) -> pd.DataFrame:
        new_dims = [d for d in current_dims if d != remove_dim]
        return self.query(new_dims, [self.measures[0].name], ["sum"])

    def get_schema_info(self) -> Dict:
        return {
            "dimensions": len(self.dimensions),
            "measures": len(self.measures),
            "cuboids_precomputed": len(self.cuboids),
            "total_cuboid_rows": sum(c.row_count for c in self.cuboids.values()),
            "coverage": len(self.cuboids) / max(1, len(self.dimensions) * (len(self.dimensions) - 1) // 2 + len(self.dimensions) + 1)
        }


# ============================================================
# DATA PROFILER, OPERATION REGISTRY, QUERY RESOLVER
# (Same as v2 but enhanced for cube integration)
# ============================================================

class ColumnType(Enum):
    TEMPORAL = "temporal"
    IDENTIFIER = "identifier"
    METRIC = "metric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"

class DataProfiler:
    def __init__(self, df):
        self.df = df
        self.profiles = {}
        self.temporal_column = None
        self.metric_columns = []
        self.categorical_columns = []
        self.identifier_columns = []
        self._profile_all()

    def _profile_all(self):
        for col in self.df.columns:
            profile = self._profile_column(col)
            self.profiles[col] = profile
            if profile.col_type == ColumnType.TEMPORAL:
                self.temporal_column = col
            elif profile.col_type == ColumnType.METRIC:
                self.metric_columns.append(col)
            elif profile.col_type == ColumnType.CATEGORICAL:
                self.categorical_columns.append(col)
            elif profile.col_type == ColumnType.IDENTIFIER:
                self.identifier_columns.append(col)

    def _profile_column(self, col):
        series = self.df[col]
        dtype = str(series.dtype)
        null_pct = series.isnull().mean() * 100
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / len(series) if len(series) > 0 else 0
        sample = series.dropna().head(5).tolist()

        col_type, confidence, stats, warnings = self._infer_type(series, dtype, unique_ratio, unique_count)

        return type('Profile', (), {
            'name': col, 'dtype': dtype, 'col_type': col_type, 'confidence': confidence,
            'null_pct': null_pct, 'unique_count': unique_count, 'unique_ratio': unique_ratio,
            'sample_values': sample, 'stats': stats, 'warnings': warnings
        })()

    def _infer_type(self, series, dtype, unique_ratio, unique_count):
        warnings = []
        stats = {}

        if self._is_temporal(series):
            return ColumnType.TEMPORAL, 0.95, {"format": "detected"}, []

        if re.search(r'year', series.name, re.I) and series.dtype in ['int64', 'float64']:
            if series.min() > 1900 and series.max() < 2100:
                return ColumnType.TEMPORAL, 0.8, {"component": "year"}, ["Check for Month column"]

        if re.search(r'month', series.name, re.I):
            if series.dtype == 'object':
                month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                               'july', 'august', 'september', 'october', 'november', 'december']
                sample_lower = [str(v).lower() for v in series.dropna().head(20)]
                matches = sum(1 for s in sample_lower if s in month_names)
                if matches / max(1, len(sample_lower)) > 0.8:
                    return ColumnType.TEMPORAL, 0.85, {"component": "month_name"}, ["Check for Year column"]
            elif series.dtype in ['int64', 'float64']:
                if series.min() >= 1 and series.max() <= 12:
                    return ColumnType.TEMPORAL, 0.85, {"component": "month_number"}, ["Check for Year column"]

        if unique_count == 2:
            return ColumnType.BOOLEAN, 0.95, {}, []

        if unique_ratio > 0.9 and series.dtype in ['int64', 'object']:
            return ColumnType.IDENTIFIER, 0.85, {"pattern": "high_cardinality"}, []

        if series.dtype in ['int64', 'float64']:
            if unique_count <= 20 and unique_ratio < 0.05:
                return ColumnType.CATEGORICAL, 0.8, {"numeric_codes": True}, []

            stats['mean'] = series.mean()
            stats['std'] = series.std()
            if stats['std'] == 0 or pd.isna(stats['std']):
                return ColumnType.IDENTIFIER, 0.6, stats, ["Zero variance"]
            return ColumnType.METRIC, 0.9, stats, []

        if series.dtype == 'object':
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return ColumnType.TEXT, 0.9, {"avg_length": avg_len}, []
            if unique_count <= 50 or unique_ratio < 0.2:
                return ColumnType.CATEGORICAL, 0.85, {"categories": unique_count}, []
            return ColumnType.TEXT, 0.7, {"avg_length": avg_len}, []

        return ColumnType.UNKNOWN, 0.5, {}, []

    def _is_temporal(self, series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype == 'object':
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False
            formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%b %Y']
            for fmt in formats:
                try:
                    parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        return True
                except:
                    continue
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    return True
            except:
                pass
        return False

    def construct_temporal(self):
        if self.temporal_column and self.profiles[self.temporal_column].stats.get('component') == 'year':
            month_col = None
            for col, prof in self.profiles.items():
                if prof.col_type == ColumnType.TEMPORAL and prof.stats.get('component') in ['month_name', 'month_number']:
                    month_col = col
                    break
            if month_col:
                try:
                    year = self.df[self.temporal_column].astype(int)
                    month = self.df[month_col]
                    if self.profiles[month_col].stats.get('component') == 'month_name':
                        dt = pd.to_datetime(year.astype(str) + '-' + month, format='%Y-%B', errors='coerce')
                    else:
                        dt = pd.to_datetime(year.astype(str) + '-' + month.astype(str), format='%Y-%m', errors='coerce')
                    return dt
                except:
                    pass
        if self.temporal_column:
            return pd.to_datetime(self.df[self.temporal_column], errors='coerce')
        return None

class OperationRegistry:
    def __init__(self, profiler, cube):
        self.profiler = profiler
        self.cube = cube
        self.df = profiler.df.copy()
        self.dt_col = profiler.construct_temporal()
        if self.dt_col is not None:
            self.df['_axi_dt'] = self.dt_col

    def _get_metric(self, name):
        name_lower = name.lower().replace(' ', '').replace('_', '')
        for col in self.profiler.metric_columns:
            col_clean = col.lower().replace(' ', '').replace('_', '')
            if name_lower == col_clean or name_lower in col_clean or col_clean in name_lower:
                return col
        return None

    def _get_category(self, name):
        name_lower = name.lower().replace(' ', '').replace('_', '')
        for col in self.profiler.categorical_columns:
            col_clean = col.lower().replace(' ', '').replace('_', '')
            if name_lower == col_clean:
                return col
        return None

    def trend_analysis(self, metric):
        col = self._get_metric(metric)
        if not col:
            return type('Result', (), {'valid': False, 'warnings': [f"Metric '{metric}' not found"], 'result': None, 'assumptions': {}, 'viz_type': 'none', 'data': None})()
        if self.dt_col is None:
            return type('Result', (), {'valid': False, 'warnings': ["No temporal column"], 'result': None, 'assumptions': {}, 'viz_type': 'none', 'data': None})()

        df_clean = self.df[[col, '_axi_dt']].dropna().sort_values('_axi_dt')
        if len(df_clean) < 3:
            return type('Result', (), {'valid': False, 'warnings': ["Need 3+ points"], 'result': None, 'assumptions': {}, 'viz_type': 'none', 'data': None})()

        time_numeric = np.arange(len(df_clean))
        tau, p_value = kendalltau(time_numeric, df_clean[col])

        slopes = []
        for i in range(len(df_clean)):
            for j in range(i+1, len(df_clean)):
                if time_numeric[j] != time_numeric[i]:
                    slopes.append((df_clean[col].iloc[j] - df_clean[col].iloc[i]) / (time_numeric[j] - time_numeric[i]))
        theil_slope = np.median(slopes) if slopes else 0

        direction = "increasing" if tau > 0.2 else "decreasing" if tau < -0.2 else "stable"

        result = {
            "direction": direction, "mann_kendall_tau": round(tau, 4),
            "p_value": round(p_value, 4), "significant": p_value < 0.05,
            "theil_sen_slope": round(theil_slope, 4), "n_points": len(df_clean),
            "start_value": round(df_clean[col].iloc[0], 2),
            "end_value": round(df_clean[col].iloc[-1], 2),
            "total_change_pct": round((df_clean[col].iloc[-1] / df_clean[col].iloc[0] - 1) * 100, 2) if df_clean[col].iloc[0] != 0 else None
        }

        return type('Result', (), {'valid': True, 'warnings': [], 'result': result, 'assumptions': {}, 'viz_type': 'line', 'data': df_clean})()

    def group_aggregate(self, metric, by, agg="mean"):
        m_col = self._get_metric(metric)
        b_col = self._get_category(by)
        if not m_col or not b_col:
            return type('Result', (), {'valid': False, 'warnings': ["Column not found"], 'result': None, 'assumptions': {}, 'viz_type': 'none', 'data': None})()

        # Try cube first
        if self.cube:
            try:
                result_df = self.cube.query([b_col], [m_col], [agg])
                if result_df is not None and len(result_df) > 0:
                    result = {"aggregation": agg, "group_by": b_col, "metric": m_col, "n_groups": len(result_df), "values": result_df.to_dict('records'), "source": "cube"}
                    return type('Result', (), {'valid': True, 'warnings': [], 'result': result, 'assumptions': {}, 'viz_type': 'bar', 'data': result_df})()
            except:
                pass

        # Fallback
        result_df = self.df.groupby(b_col)[m_col].agg(agg).reset_index()
        result_df.columns = [b_col, f"{agg}_{m_col}"]
        result = {"aggregation": agg, "group_by": b_col, "metric": m_col, "n_groups": len(result_df), "values": result_df.to_dict('records'), "source": "raw"}
        return type('Result', (), {'valid': True, 'warnings': [], 'result': result, 'assumptions': {}, 'viz_type': 'bar', 'data': result_df})()

class QueryResolver:
    def __init__(self, profiler):
        self.profiler = profiler
        self.context = []

    def resolve(self, query):
        query_lower = query.lower()
        intent = self._detect_intent(query_lower)
        entities = self._extract_entities(query_lower)

        if self.context and not entities.get('metric'):
            for ctx in reversed(self.context):
                if 'metric' in ctx.get('entities', {}):
                    entities['metric'] = ctx['entities']['metric']
                    break

        self.context.append({"query": query, "intent": intent, "entities": entities})
        self.context = self.context[-5:]

        return {"intent": intent, "entities": entities}

    def _detect_intent(self, query):
        if any(w in query for w in ['trend', 'moving', 'direction', 'going', 'trajectory', 'over time']):
            return {"type": "trend_analysis", "confidence": 0.9}
        if any(w in query for w in ['total', 'sum', 'average', 'mean', 'count', 'per', 'by', 'breakdown']):
            return {"type": "group_aggregate", "confidence": 0.85}
        return {"type": "auto", "confidence": 0.5}

    def _extract_entities(self, query):
        entities = {}
        for col in self.profiler.metric_columns:
            patterns = [col.lower(), col.lower().replace('_', ' '), col.lower().replace('_', '')]
            for p in patterns:
                if p in query:
                    entities['metric'] = col
                    break
        for col in self.profiler.categorical_columns:
            if col.lower() in query or col.lower().replace('_', ' ') in query:
                entities['category'] = col
                break
        if 'sum' in query or 'total' in query:
            entities['aggregation'] = 'sum'
        elif 'average' in query or 'mean' in query:
            entities['aggregation'] = 'mean'
        else:
            entities['aggregation'] = 'mean'
        return entities

class InsightObservability:
    def __init__(self):
        self.insights = []
        self.feedback = {}

    def log_insight(self, query, intent, operation, result, duration_ms):
        insight_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        self.insights.append({
            "id": insight_id, "timestamp": datetime.now().isoformat(),
            "query": query, "intent": intent, "operation": operation,
            "valid": result.valid if result else False, "duration_ms": duration_ms,
            "feedback": None
        })
        return insight_id

    def get_catalog(self):
        return {
            "total": len(self.insights),
            "valid_rate": sum(1 for i in self.insights if i["valid"]) / max(1, len(self.insights)),
            "avg_duration": np.mean([i["duration_ms"] for i in self.insights]) if self.insights else 0
        }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AxiLattice Pro v3.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
.chat-user { background: #388bfd20; border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 3px solid #388bfd; }
.chat-assistant { background: #23863620; border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 3px solid #238636; }
.stApp { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'profiler' not in st.session_state:
    st.session_state.profiler = None
if 'cube' not in st.session_state:
    st.session_state.cube = None
if 'registry' not in st.session_state:
    st.session_state.registry = None
if 'resolver' not in st.session_state:
    st.session_state.resolver = None
if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'voice' not in st.session_state:
    st.session_state.voice = VoiceManager()
if 'observability' not in st.session_state:
    st.session_state.observability = InsightObservability()

# Sidebar
with st.sidebar:
    st.markdown('<p class="main-header">🧠 AxiLattice v3.0</p>', unsafe_allow_html=True)
    st.caption("Voice I/O | Cube Architecture | Observability")

    uploaded = st.file_uploader("📁 Upload Data", type=['csv', 'xlsx', 'parquet'])

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif uploaded.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded)
            else:
                uploaded.seek(0)
                df = pd.read_excel(uploaded, engine='openpyxl')

            st.session_state.df = df
            st.session_state.profiler = DataProfiler(df)
            st.session_state.cube = DataCube(df, st.session_state.profiler)
            st.session_state.registry = OperationRegistry(st.session_state.profiler, st.session_state.cube)
            st.session_state.resolver = QueryResolver(st.session_state.profiler)

            st.success(f"✅ {len(df):,} rows × {len(df.columns)} cols")

            with st.expander("📋 Schema + Cube"):
                prof = st.session_state.profiler
                for col, p in prof.profiles.items():
                    icon = {"temporal": "📅", "metric": "📊", "categorical": "🏷️", "identifier": "🔑", "boolean": "☑️", "text": "📝"}.get(p.col_type.value, "❓")
                    st.write(f"{icon} **{col}** ({p.col_type.value}, {p.confidence:.0%})")

                cube_info = st.session_state.cube.get_schema_info()
                st.json(cube_info)

        except Exception as e:
            st.error(f"❌ Failed: {str(e)}")

    if st.session_state.profiler:
        mode = st.radio("Mode", ["🎙️ Ask (Voice/Text)", "📊 Auto-Report", "🔍 Explore", "📈 Observability"])
    else:
        st.info("Upload data to begin")

# Main
if not st.session_state.profiler:
    st.markdown("## 👋 AxiLattice Pro v3.0\n\n**Voice I/O** — Speak questions, hear answers\n**Cube Architecture** — O(1) queries via pre-aggregation\n**Insight Observability** — Track, validate, improve insights")
    st.stop()

prof = st.session_state.profiler
cube = st.session_state.cube
reg = st.session_state.registry
resolver = st.session_state.resolver
voice = st.session_state.voice
obs = st.session_state.observability

if "Ask" in mode:
    st.markdown('<p class="main-header">🎙️ Conversational Analytics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Voice Control")

        audio_file = st.file_uploader("🎤 Upload voice", type=['wav', 'mp3', 'm4a'])
        if audio_file:
            with st.spinner("Transcribing..."):
                audio_bytes = audio_file.read()
                transcribed = voice.transcribe_audio(audio_bytes)
                st.session_state.pending_query = transcribed
                st.success(f"Heard: '{transcribed}'")

        if st.session_state.chat and st.session_state.chat[-1]['role'] == 'assistant':
            last_text = st.session_state.chat[-1].get('tts_text', '')
            if last_text and st.button("🔊 Play Answer"):
                audio_bytes = voice.synthesize_speech(last_text)
                if audio_bytes:
                    st.markdown(voice.get_audio_html(audio_bytes), unsafe_allow_html=True)
                else:
                    st.warning("Install gTTS: pip install gtts")

        st.divider()
        st.subheader("Quick Questions")
        suggestions = [
            f"Trend in {prof.metric_columns[0]}" if prof.metric_columns else "",
            f"Average by {prof.categorical_columns[0]}" if prof.categorical_columns else "",
        ]
        for s in suggestions:
            if s and st.button(s, key=f"btn_{s[:20]}"):
                st.session_state.pending_query = s

    with col2:
        for msg in st.session_state.chat[-10:]:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-user'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-assistant'><b>AxiLattice:</b></div>", unsafe_allow_html=True)
                st.markdown(msg['content'])
                if 'viz' in msg and msg['viz']:
                    st.plotly_chart(msg['viz'], use_container_width=True)

        query = st.chat_input("Ask anything...")
        if not query and 'pending_query' in st.session_state:
            query = st.session_state.pending_query
            del st.session_state.pending_query

        if query:
            import time
            start_time = time.time()

            st.session_state.chat.append({"role": "user", "content": query})

            with st.spinner("Analyzing..."):
                resolved = resolver.resolve(query)
                intent = resolved['intent']['type']
                entities = resolved['entities']

                result = None
                text = ""
                viz = None

                if intent == "trend_analysis":
                    metric = entities.get('metric', prof.metric_columns[0] if prof.metric_columns else None)
                    if metric:
                        result = reg.trend_analysis(metric)
                        if result.valid:
                            r = result.result
                            text = f"**Trend: {metric}**\n\n📈 Direction: {r['direction'].upper()}\n📊 τ = {r['mann_kendall_tau']} (p={r['p_value']})\n📉 Slope: {r['theil_sen_slope']} per period\n💰 Total Change: {r['total_change_pct']}%"
                        else:
                            text = f"❌ {result.warnings[0] if result.warnings else 'Error'}"
                    else:
                        text = "❌ No metric found"

                elif intent == "group_aggregate":
                    metric = entities.get('metric', prof.metric_columns[0] if prof.metric_columns else None)
                    cat = entities.get('category', prof.categorical_columns[0] if prof.categorical_columns else None)
                    agg = entities.get('aggregation', 'mean')
                    if metric and cat:
                        result = reg.group_aggregate(metric, cat, agg)
                        if result.valid:
                            r = result.result
                            text = f"**{agg.title()} of {metric} by {cat}**\n\nGroups: {r['n_groups']} (source: {r.get('source', 'raw')})\n\n"
                            for v in r['values'][:5]:
                                text += f"\n• {v[r['group_by']]}: **{v[f'{agg}_{metric}']:.2f}**"
                        else:
                            text = f"❌ {result.warnings[0] if result.warnings else 'Error'}"
                    else:
                        text = f"Need metric and category"

                else:
                    text = f"🤔 Try: 'Trend in {prof.metric_columns[0] if prof.metric_columns else 'sales'}' or 'Average by {prof.categorical_columns[0] if prof.categorical_columns else 'region'}'"

                duration = int((time.time() - start_time) * 1000)
                insight_id = obs.log_insight(query, resolved['intent'], intent, result, duration)

                tts_text = re.sub(r'[\*\#]', '', text).replace('\n', ' ').strip()[:500]

                if result and result.viz_type == 'line':
                    viz = px.line(result.data, x='_axi_dt', y=result.data.columns[0])
                elif result and result.viz_type == 'bar':
                    viz = px.bar(result.data, x=result.data.columns[0], y=result.data.columns[1])

                st.session_state.chat.append({
                    "role": "assistant", "content": text, "tts_text": tts_text,
                    "viz": viz, "insight_id": insight_id
                })

                st.rerun()

elif "Auto-Report" in mode:
    st.markdown('<p class="main-header">📊 Auto-Report</p>', unsafe_allow_html=True)
    selected = st.multiselect("Metrics", prof.metric_columns, default=prof.metric_columns[:min(3, len(prof.metric_columns))])
    if selected and st.button("🚀 Generate"):
        for metric in selected:
            with st.expander(f"📈 {metric}", expanded=True):
                trend = reg.trend_analysis(metric)
                if trend.valid:
                    st.metric("Trend", trend.result['direction'].upper(), f"τ={trend.result['mann_kendall_tau']}")
                else:
                    st.error(trend.warnings[0] if trend.warnings else "Error")

elif "Explore" in mode:
    st.markdown('<p class="main-header">🔍 Explore</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Data", "📊 Dist", "🧊 Cube"])
    with tab1:
        st.dataframe(st.session_state.df)
    with tab2:
        if prof.metric_columns:
            col = st.selectbox("Metric", prof.metric_columns)
            fig = px.histogram(st.session_state.df, x=col)
            st.plotly_chart(fig)
    with tab3:
        st.json(cube.get_schema_info())
        for key, cuboid in cube.cuboids.items():
            with st.expander(f"Cuboid: {key}"):
                st.write(f"Rows: {cuboid.row_count}, Dimensions: {cuboid.dimensions}")
                st.dataframe(cuboid.df.head())

elif "Observability" in mode:
    st.markdown('<p class="main-header">📈 Insight Observability</p>', unsafe_allow_html=True)
    catalog = obs.get_catalog()
    st.json(catalog)

    if obs.insights:
        st.subheader("Recent Insights")
        for ins in obs.insights[-10:]:
            with st.expander(f"{ins['query'][:50]}... ({ins['duration_ms']}ms)"):
                st.json(ins)

st.divider()
st.caption("AxiLattice Pro v3.0 | Voice I/O | Cube Architecture | Insight Observability")

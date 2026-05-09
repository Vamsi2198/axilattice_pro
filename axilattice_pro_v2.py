"""
AxiLattice Pro — Bulletproof Production Version v2
Fixed: Cube init (avg->mean, cnt->size), None guards, Voice flow, Observability, Slice dims
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json, re, hashlib, base64, io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import kendalltau, spearmanr, chi2_contingency, f_oneway
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NL = "\n"

# ============================================================
# VOICE I/O — LIVE RECORDING + RESPONSE
# ============================================================

class VoiceManager:
    def __init__(self):
        self.stt = self._check('speech_recognition')
        self.tts = self._check('gtts')

    def _check(self, pkg):
        try:
            __import__(pkg)
            return True
        except Exception:
            return False

    def transcribe(self, audio_bytes: bytes) -> str:
        if not self.stt:
            return "[Install SpeechRecognition: pip install SpeechRecognition]"
        import speech_recognition as sr
        try:
            r = sr.Recognizer()
            with io.BytesIO(audio_bytes) as f:
                with sr.AudioFile(f) as src:
                    audio = r.record(src)
                    return r.recognize_google(audio)
        except Exception as e:
            return f"[STT Error: {str(e)}]"

    def speak(self, text: str) -> Optional[bytes]:
        if not self.tts:
            return None
        from gtts import gTTS
        try:
            fp = io.BytesIO()
            gTTS(text=text[:500], lang='en', slow=False).write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception:
            return None

    def html(self, audio_bytes: bytes) -> str:
        b64 = base64.b64encode(audio_bytes).decode()
        return f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>'


# ============================================================
# CUBE ARCHITECTURE — FIXED AGG ALIASES
# ============================================================

@dataclass
class CubeDim:
    name: str
    col: str
    card: int

@dataclass
class CubeMeas:
    name: str
    col: str
    aggs: List[str]

@dataclass
class Cuboid:
    dims: Tuple[str, ...]
    df: pd.DataFrame
    rows: int

# Pandas-compatible agg names
AGG_MAP = {
    "sum": "sum",
    "avg": "mean",      # FIXED: was "avg"
    "mean": "mean",
    "cnt": "size",      # FIXED: was "cnt"
    "count": "size",
    "min": "min",
    "max": "max",
    "std": "std"
}

class DataCube:
    def __init__(self, df, profiler, max_dims=3):
        self.df = df
        self.prof = profiler
        self.max_dims = max_dims
        self.dims: List[CubeDim] = []
        self.meas: List[CubeMeas] = []
        self.cuboids: Dict[str, Cuboid] = {}
        self._build()
        self._precompute()

    def _build(self):
        for c in self.prof.cat_cols:
            self.dims.append(CubeDim(c, c, self.df[c].nunique()))
        if self.prof.temp_col:
            self.dims.append(CubeDim("Time", self.prof.temp_col, self.df[self.prof.temp_col].nunique()))
        for c in self.prof.num_cols:
            self.meas.append(CubeMeas(c, c, ["sum","avg","cnt","min","max","std"]))

    def _precompute(self):
        from itertools import combinations
        names = [d.name for d in self.dims]
        n = len(names)
        for r in range(1, min(self.max_dims+1, n+1)):
            for combo in combinations(names, r):
                cards = [next(x.card for x in self.dims if x.name == nm) for nm in combo]
                if np.prod(cards) < 2e5:
                    self._make(combo)
        self._make(())

    def _make(self, dims: Tuple):
        key = "_".join(sorted(dims)) if dims else "total"
        if key in self.cuboids:
            return
        cols = [next(d.col for d in self.dims if d.name == n) for n in dims]
        ad = {}
        for m in self.meas:
            for a in m.aggs:
                pandas_agg = AGG_MAP.get(a, a)
                ad[f"{m.name}_{a}"] = pd.NamedAgg(column=m.col, aggfunc=pandas_agg)
        if cols:
            df = self.df.groupby(cols, observed=True).agg(**ad).reset_index()
        else:
            total_data = {}
            for m in self.meas:
                for a in m.aggs:
                    pandas_agg = AGG_MAP.get(a, a)
                    total_data[f"{m.name}_{a}"] = [self.df[m.col].agg(pandas_agg)]
            df = pd.DataFrame(total_data)
        self.cuboids[key] = Cuboid(dims, df, len(df))

    def query(self, dims, meas, aggs, filt=None):
        key = "_".join(sorted(dims)) if dims else "total"
        if key in self.cuboids:
            r = self.cuboids[key].df.copy()
            if filt:
                for c, v in filt.items():
                    if c in r.columns:
                        r = r[r[c] == v]
            mc = [f"{m}_{a}" for m, a in zip(meas, aggs)]
            ac = [c for c in mc if c in r.columns]
            if ac:
                return r[dims + ac]
        return self._raw(dims, meas, aggs, filt)

    def _raw(self, dims, meas, aggs, filt):
        d = self.df.copy()
        if filt:
            for c, v in filt.items():
                if c in d.columns:
                    d = d[d[c] == v]
        if dims:
            ad = {}
            for m, a in zip(meas, aggs):
                pandas_agg = AGG_MAP.get(a, a)
                ad[f"{m}_{a}"] = pd.NamedAgg(column=m, aggfunc=pandas_agg)
            return d.groupby(dims, observed=True).agg(**ad).reset_index()
        total_data = {}
        for m, a in zip(meas, aggs):
            pandas_agg = AGG_MAP.get(a, a)
            total_data[f"{m}_{a}"] = [d[m].agg(pandas_agg)]
        return pd.DataFrame(total_data)

    def info(self):
        n = len(self.dims)
        return {
            "dims": n, "meas": len(self.meas),
            "cuboids": len(self.cuboids),
            "rows": sum(c.rows for c in self.cuboids.values()),
            "max_precompute_dims": self.max_dims
        }


# ============================================================
# DATA PROFILER — HANDLES UNNAMED COLUMNS
# ============================================================

class ColType(Enum):
    TEMP = "temporal"
    ID = "identifier"
    NUM = "metric"
    CAT = "categorical"
    BOOL = "boolean"
    TEXT = "text"
    UNK = "unknown"

class DataProfiler:
    def __init__(self, df):
        self.df = df
        self.profs = {}
        self.temp_col = None
        self.num_cols = []
        self.cat_cols = []
        self.id_cols = []
        self._run()

    def _run(self):
        for c in self.df.columns:
            p = self._profile(c)
            self.profs[c] = p
            if p.type == ColType.TEMP:
                self.temp_col = c
            elif p.type == ColType.NUM:
                self.num_cols.append(c)
            elif p.type == ColType.CAT:
                self.cat_cols.append(c)
            elif p.type == ColType.ID:
                self.id_cols.append(c)

    def _profile(self, col):
        s = self.df[col]
        dt = str(s.dtype)
        null = s.isnull().mean()*100
        uniq = s.nunique(dropna=True)
        ratio = uniq/len(s) if len(s) else 0

        t, conf, st, warn = self._infer(s, dt, ratio, uniq)
        return type('P', (), {
            'name': col, 'dtype': dt, 'type': t, 'conf': conf,
            'null': null, 'uniq': uniq, 'ratio': ratio,
            'stats': st, 'warn': warn
        })()

    def _infer(self, s, dt, ratio, uniq):
        if self._temp(s):
            return ColType.TEMP, 0.95, {}, []
        if re.search(r'year', str(s.name), re.I) and dt in ['int64','float64']:
            if 1900 < s.min() < 2100:
                return ColType.TEMP, 0.8, {"part":"year"}, ["Check Month"]
        if re.search(r'month', str(s.name), re.I):
            if dt == 'object':
                mn = ['january','february','march','april','may','june','july','august','september','october','november','december']
                sl = [str(v).lower() for v in s.dropna().head(20)]
                if sum(1 for x in sl if x in mn)/max(1,len(sl)) > 0.8:
                    return ColType.TEMP, 0.85, {"part":"month_name"}, ["Check Year"]
            elif dt in ['int64','float64'] and 1 <= s.min() <= 12 <= s.max() <= 12:
                return ColType.TEMP, 0.85, {"part":"month_num"}, ["Check Year"]
        if uniq == 2:
            return ColType.BOOL, 0.95, {}, []
        if ratio > 0.9 and dt in ['int64','object']:
            return ColType.ID, 0.85, {"high_card":True}, []
        if dt in ['int64','float64']:
            if uniq <= 20 and ratio < 0.05:
                return ColType.CAT, 0.8, {"codes":True}, []
            mu, sd = s.mean(), s.std()
            if sd == 0 or pd.isna(sd):
                return ColType.ID, 0.6, {"mean":mu}, ["Zero var"]
            return ColType.NUM, 0.9, {"mean":mu,"std":sd}, []
        if dt == 'object':
            al = s.dropna().astype(str).str.len().mean()
            if al > 50:
                return ColType.TEXT, 0.9, {"avg_len":al}, []
            if uniq <= 50 or ratio < 0.2:
                return ColType.CAT, 0.85, {"cats":uniq}, []
            return ColType.TEXT, 0.7, {"avg_len":al}, []
        return ColType.UNK, 0.5, {}, []

    def _temp(self, s):
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if s.dtype == 'object':
            sm = s.dropna().head(100)
            if len(sm) == 0:
                return False
            for fmt in ['%Y-%m-%d','%Y/%m/%d','%d-%m-%Y','%d/%m/%Y','%Y-%m-%d %H:%M:%S','%m/%d/%Y','%b %Y']:
                try:
                    if pd.to_datetime(sm, format=fmt, errors='coerce').notna().mean() > 0.8:
                        return True
                except Exception:
                    pass
            try:
                if pd.to_datetime(sm, errors='coerce').notna().mean() > 0.8:
                    return True
            except Exception:
                pass
        return False

    def get_dt(self):
        if self.temp_col and self.profs[self.temp_col].stats.get('part') == 'year':
            for c, p in self.profs.items():
                if p.type == ColType.TEMP and p.stats.get('part') in ['month_name','month_num']:
                    try:
                        y = self.df[self.temp_col].astype(int)
                        m = self.df[c]
                        if p.stats.get('part') == 'month_name':
                            return pd.to_datetime(y.astype(str)+'-'+m, format='%Y-%B', errors='coerce')
                        return pd.to_datetime(y.astype(str)+'-'+m.astype(str), format='%Y-%m', errors='coerce')
                    except Exception:
                        pass
        if self.temp_col:
            return pd.to_datetime(self.df[self.temp_col], errors='coerce')
        return None


# ============================================================
# INSIGHT ENGINE
# ============================================================

class InsightResult:
    def __init__(self, valid=True, text="", viz=None, data=None, warnings=None, meta=None):
        self.valid = valid
        self.text = text
        self.viz = viz
        self.data = data
        self.warnings = warnings or []
        self.meta = meta or {}

class InsightEngine:
    def __init__(self, profiler, cube):
        self.prof = profiler
        self.cube = cube
        self.df = profiler.df.copy()
        self.dt = profiler.get_dt()
        if self.dt is not None:
            self.df['_dt'] = self.dt

    def _find(self, name, pool):
        if not name:
            return None
        nl = name.lower().replace(' ','').replace('_','')
        for c in pool:
            cl = c.lower().replace(' ','').replace('_','')
            if nl == cl or nl in cl or cl in nl:
                return c
        return None

    def _find_num(self, name):
        return self._find(name, self.prof.num_cols)

    def _find_cat(self, name):
        return self._find(name, self.prof.cat_cols)

    def _find_all_cats(self, query_text):
        found = []
        ql = query_text.lower()
        for c in self.prof.cat_cols:
            patterns = [c.lower(), c.lower().replace('_',' '), c.lower().replace('_','')]
            for p in patterns:
                if p in ql and c not in found:
                    found.append(c)
                    break
        return found

    def trend(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna().sort_values('_dt')
        if len(d) < 3:
            return InsightResult(False, "Need 3+ points")
        tn = np.arange(len(d))
        tau, p = kendalltau(tn, d[col])
        slopes = [(d[col].iloc[j]-d[col].iloc[i])/(j-i) for i in range(len(d)) for j in range(i+1,len(d))]
        slope = np.median(slopes) if slopes else 0
        dir = "up" if tau > 0.2 else "down" if tau < -0.2 else "flat"
        r = {
            "dir": dir, "tau": round(tau,4), "p": round(p,4),
            "sig": p < 0.05, "slope": round(slope,4), "n": len(d),
            "start": round(d[col].iloc[0],2), "end": round(d[col].iloc[-1],2),
            "chg": round((d[col].iloc[-1]/d[col].iloc[0]-1)*100,2) if d[col].iloc[0] != 0 else None
        }
        fig = px.line(d, x='_dt', y=col, title=f"{col} Trend")
        txt = f"**Trend: {col}**" + NL + NL + f"📈 {dir.upper()} | τ={r['tau']} (p={r['p']})" + NL + f"📉 Slope: {r['slope']}/period" + NL + f"💰 Change: {r['chg']}%"
        return InsightResult(True, txt, fig, d, meta={"type":"trend","result":r})

    def aggregate(self, metric, by_dims, agg="sum"):
        mc = self._find_num(metric)
        if not mc:
            return InsightResult(False, f"Metric '{metric}' not found")
        if isinstance(by_dims, str):
            by_dims = [by_dims]
        bc = []
        for b in by_dims:
            found = self._find_cat(b)
            if found:
                bc.append(found)
        if not bc:
            return InsightResult(False, f"No valid dimensions found in {by_dims}")
        try:
            rdf = self.cube.query(bc, [mc], [agg])
            src = "cube"
        except Exception as e:
            pandas_agg = AGG_MAP.get(agg, agg)
            rdf = self.df.groupby(bc)[mc].agg(pandas_agg).reset_index()
            src = "raw"
        vals = rdf.to_dict('records')
        dim_str = " × ".join(bc)
        txt = f"**{agg.title()} {mc} by {dim_str}**" + NL + NL + f"Groups: {len(rdf)} ({src})"
        for v in vals[:10]:
            dim_vals = " | ".join([str(v[b]) for b in bc])
            measure_col = f"{mc}_{agg}" if f"{mc}_{agg}" in rdf.columns else mc
            txt += NL + f"• {dim_vals}: **{v[measure_col]:.2f}**"
        if len(vals) > 10:
            txt += NL + NL + f"... and {len(vals)-10} more groups"
        if len(bc) == 1:
            fig = px.bar(rdf, x=bc[0], y=rdf.columns[-1], title=f"{agg.title()} {mc} by {bc[0]}")
        elif len(bc) == 2:
            fig = px.density_heatmap(rdf, x=bc[0], y=bc[1], z=rdf.columns[-1],
                                     title=f"{agg.title()} {mc} by {bc[0]} × {bc[1]}",
                                     color_continuous_scale="Blues")
        else:
            fig = None
        return InsightResult(True, txt, fig, rdf,
            meta={"type":"aggregate","result":{"agg":agg,"by":bc,"metric":mc,"n":len(rdf),"src":src}})

    def slice_data(self, metric, dimensions, filters=None, agg="sum"):
        mc = self._find_num(metric)
        if not mc:
            return InsightResult(False, f"Metric '{metric}' not found")
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        resolved_dims = []
        for d in dimensions:
            found = self._find_cat(d)
            if found:
                resolved_dims.append(found)
        if not resolved_dims:
            return InsightResult(False, f"No valid dimensions in {dimensions}")
        df_work = self.df.copy()
        if filters:
            for col, val in filters.items():
                found_col = self._find_cat(col) or self._find_num(col)
                if found_col and found_col in df_work.columns:
                    df_work = df_work[df_work[found_col] == val]
        try:
            pandas_agg = AGG_MAP.get(agg, agg)
            rdf = df_work.groupby(resolved_dims)[mc].agg(pandas_agg).reset_index()
        except Exception as e:
            return InsightResult(False, f"Aggregation failed: {e}")
        dim_str = " × ".join(resolved_dims)
        txt = f"**{agg.title()} {mc} by {dim_str}**" + NL + NL
        if filters:
            filt_str = " | ".join([f"{k}={v}" for k, v in filters.items()])
            txt += f"🔍 Filtered: {filt_str}" + NL + NL
        txt += f"Groups: {len(rdf)}"
        for _, v in rdf.head(15).iterrows():
            dim_vals = " | ".join([str(v[d]) for d in resolved_dims])
            txt += NL + f"• {dim_vals}: **{v[mc]:.2f}**"
        if len(rdf) > 15:
            txt += NL + NL + f"... and {len(rdf)-15} more"
        if len(resolved_dims) == 1:
            fig = px.bar(rdf, x=resolved_dims[0], y=mc, title=f"{agg.title()} {mc} by {resolved_dims[0]}")
        elif len(resolved_dims) == 2:
            fig = px.density_heatmap(rdf, x=resolved_dims[0], y=resolved_dims[1], z=mc,
                                     title=f"{agg.title()} {mc} by {dim_str}", color_continuous_scale="Blues")
        elif len(resolved_dims) == 3:
            fig = px.bar(rdf, x=resolved_dims[0], y=mc, color=resolved_dims[1], facet_col=resolved_dims[2],
                        title=f"{agg.title()} {mc} by {dim_str}")
        else:
            fig = None
        return InsightResult(True, txt, fig, rdf,
            meta={"type":"slice","result":{"agg":agg,"by":resolved_dims,"metric":mc,"n":len(rdf),"filters":filters}})

    def distribution(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        s = self.df[col].dropna()
        if len(s) < 3:
            return InsightResult(False, "Need 3+ values")
        q1, q2, q3 = s.quantile([0.25,0.5,0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        out = s[(s < lo) | (s > hi)]
        skew = s.skew()
        kurt = s.kurtosis()
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram","Box Plot"))
        fig.add_trace(go.Histogram(x=s, nbinsx=30, name="Hist"), row=1, col=1)
        fig.add_trace(go.Box(x=s, name="Box"), row=1, col=2)
        fig.update_layout(title=f"{col} Distribution")
        txt = f"**Distribution: {col}**" + NL + NL + f"📊 Median: {q2:.2f} | IQR: {iqr:.2f}" + NL + f"📈 Skew: {skew:.2f} | Kurt: {kurt:.2f}" + NL + f"⚠️ Outliers: {len(out)} ({len(out)/len(s)*100:.1f}%)"
        return InsightResult(True, txt, fig, s, meta={"type":"distribution","result":{"median":q2,"iqr":iqr,"skew":skew,"outliers":len(out)}})

    def correlation(self, m1, m2):
        c1 = self._find_num(m1)
        c2 = self._find_num(m2)
        if not c1 or not c2:
            return InsightResult(False, "Metrics not found")
        d = self.df[[c1,c2]].dropna()
        if len(d) < 3:
            return InsightResult(False, "Need 3+ pairs")
        r, p = spearmanr(d[c1], d[c2])
        fig = px.scatter(d, x=c1, y=c2, trendline="ols", title=f"{c1} vs {c2}")
        txt = f"**Correlation: {c1} vs {c2}**" + NL + NL + f"🔗 ρ = {r:.4f} (p={p:.4f})" + NL + f"📊 Strength: {'strong' if abs(r)>0.7 else 'moderate' if abs(r)>0.4 else 'weak'} {'positive' if r>0 else 'negative'}"
        return InsightResult(True, txt, fig, d, meta={"type":"correlation","result":{"rho":r,"p":p}})

    def corr_matrix(self):
        if len(self.prof.num_cols) < 2:
            return InsightResult(False, "Need 2+ metrics")
        d = self.df[self.prof.num_cols].corr()
        fig = px.imshow(d, text_auto=".2f", aspect="auto", title="Correlation Matrix")
        mask = np.triu(np.ones_like(d, dtype=bool), k=1)
        corr_vals = d.where(mask).stack().sort_values(key=abs, ascending=False)
        top = corr_vals.head(3)
        txt = "**Correlation Matrix**" + NL + NL + "Top pairs:"
        for (a,b), v in top.items():
            txt += NL + f"• {a} ↔ {b}: {v:.3f}"
        return InsightResult(True, txt, fig, d, meta={"type":"corr_matrix","top":top.to_dict()})

    def anomaly(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        d = self.df[[col]].dropna()
        if len(d) < 10:
            return InsightResult(False, "Need 10+ rows")
        iso = IsolationForest(contamination=0.05, random_state=42)
        d['anomaly'] = iso.fit_predict(d[[col]])
        n_anom = (d['anomaly'] == -1).sum()
        fig = px.scatter(d.reset_index(), x='index', y=col, color='anomaly',
                        color_discrete_map={-1:"red",1:"blue"}, title=f"Anomalies in {col}")
        txt = f"**Anomaly Detection: {col}**" + NL + NL + f"🔴 Anomalies: {n_anom} ({n_anom/len(d)*100:.1f}%)" + NL + f"✅ Normal: {len(d)-n_anom}"
        return InsightResult(True, txt, fig, d, meta={"type":"anomaly","result":{"anomalies":n_anom}})

    def segment(self, metrics, n=3):
        cols = [self._find_num(m) for m in metrics]
        cols = [c for c in cols if c]
        if len(cols) < 2:
            return InsightResult(False, "Need 2+ valid metrics")
        d = self.df[cols].dropna()
        if len(d) < n*5:
            return InsightResult(False, "Need more data")
        sc = StandardScaler()
        X = sc.fit_transform(d)
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        d['cluster'] = km.fit_predict(X)
        centers = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=cols)
        centers['cluster'] = range(n)
        fig = px.scatter_matrix(d, dimensions=cols, color='cluster', title=f"K-Means (k={n})")
        txt = f"**Segmentation (k={n})**" + NL + NL
        for _, row in centers.iterrows():
            txt += f"Cluster {int(row['cluster'])}: " + " | ".join([f"{c}={row[c]:.2f}" for c in cols]) + NL
        return InsightResult(True, txt, fig, d, meta={"type":"segment","centers":centers.to_dict()})

    def change_point(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna().sort_values('_dt')
        if len(d) < 10:
            return InsightResult(False, "Need 10+ points")
        vals = d[col].values
        best_i, best_score = None, -1
        for i in range(5, len(vals)-5):
            before, after = vals[:i], vals[i:]
            if len(before) > 1 and len(after) > 1:
                _, p = f_oneway(before, after)
                score = -np.log10(p) if p > 0 else 100
                if score > best_score:
                    best_score, best_i = score, i
        if best_i is None:
            return InsightResult(False, "No change point found")
        cp_date = d['_dt'].iloc[best_i]
        before_mean = vals[:best_i].mean()
        after_mean = vals[best_i:].mean()
        fig = px.line(d, x='_dt', y=col, title=f"{col} Change Point")
        fig.add_vline(x=cp_date, line_dash="dash", line_color="red")
        txt = f"**Change Point: {col}**" + NL + NL + f"🎯 Date: {cp_date}" + NL + f"📉 Before: {before_mean:.2f} | After: {after_mean:.2f}" + NL + f"📊 Shift: {((after_mean/before_mean-1)*100):.1f}%"
        return InsightResult(True, txt, fig, d, meta={"type":"change_point","result":{"date":str(cp_date),"before":before_mean,"after":after_mean}})

    def pareto(self, metric, by):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        d = self.df.groupby(bc)[mc].sum().sort_values(ascending=False).reset_index()
        d['cum_pct'] = d[mc].cumsum() / d[mc].sum() * 100
        n80 = (d['cum_pct'] <= 80).sum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=d[bc], y=d[mc], name=mc), secondary_y=False)
        fig.add_trace(go.Scatter(x=d[bc], y=d['cum_pct'], name="Cumulative %", mode='lines+markers'), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.update_layout(title=f"Pareto: {mc} by {bc}")
        txt = f"**Pareto: {mc} by {bc}**" + NL + NL + f"📊 Top {n80} of {len(d)} groups = 80% of total" + NL + f"🏆 Leader: {d.iloc[0][bc]} ({d.iloc[0][mc]:.2f})"
        return InsightResult(True, txt, fig, d, meta={"type":"pareto","result":{"n80":n80,"total":len(d)}})

    def forecast(self, metric, periods=6):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna().sort_values('_dt')
        if len(d) < 6:
            return InsightResult(False, "Need 6+ points")
        alpha = 0.3
        fcast = [d[col].iloc[0]]
        for i in range(1, len(d)):
            fcast.append(alpha * d[col].iloc[i-1] + (1-alpha) * fcast[-1])
        last_date = d['_dt'].iloc[-1]
        freq = (d['_dt'].iloc[-1] - d['_dt'].iloc[-2]).days if len(d) > 1 else 30
        future_dates = [last_date + timedelta(days=freq*(i+1)) for i in range(periods)]
        last_level = fcast[-1]
        future_vals = [last_level] * periods
        hist = d.copy()
        hist['type'] = 'Historical'
        fut = pd.DataFrame({'_dt': future_dates, col: future_vals, 'type': 'Forecast'})
        combined = pd.concat([hist, fut])
        fig = px.line(combined, x='_dt', y=col, color='type', title=f"{col} Forecast")
        txt = f"**Forecast: {col}**" + NL + NL + f"📈 Next {periods} periods: ~{last_level:.2f} each" + NL + "⚠️ Simple model — use with caution"
        return InsightResult(True, txt, fig, combined, meta={"type":"forecast","result":{"next":last_level,"periods":periods}})

    def top_n(self, metric, by, n=5, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        pandas_agg = AGG_MAP.get(agg, agg)
        d = self.df.groupby(bc)[mc].agg(pandas_agg).sort_values(ascending=False).head(n).reset_index()
        fig = px.bar(d, x=bc, y=mc, title=f"Top {n} {by} by {agg} {mc}")
        txt = f"**Top {n} {by} by {agg} {mc}**" + NL + NL + NL.join([f"{i+1}. {row[bc]}: **{row[mc]:.2f}**" for i, row in d.iterrows()])
        return InsightResult(True, txt, fig, d, meta={"type":"top_n","result":{"n":n,"agg":agg}})

    def composition(self, metric, by, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        pandas_agg = AGG_MAP.get(agg, agg)
        d = self.df.groupby(bc)[mc].agg(pandas_agg).reset_index()
        fig = px.pie(d, names=bc, values=mc, title=f"{mc} Composition by {bc}")
        total = d[mc].sum()
        txt = f"**Composition: {mc} by {bc}**" + NL + NL + f"Total: {total:.2f}" + NL + NL.join([f"• {row[bc]}: {row[mc]/total*100:.1f}%" for _, row in d.iterrows()])
        return InsightResult(True, txt, fig, d, meta={"type":"composition","result":{"total":total}})

    def growth(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna().sort_values('_dt')
        if len(d) < 2:
            return InsightResult(False, "Need 2+ points")
        d['growth'] = d[col].pct_change() * 100
        d['rolling_growth'] = d['growth'].rolling(3).mean()
        avg_growth = d['growth'].mean()
        fig = px.line(d, x='_dt', y=['growth','rolling_growth'], title=f"{col} Growth Rate %")
        txt = f"**Growth: {col}**" + NL + NL + f"📈 Avg growth: {avg_growth:.2f}%" + NL + f"📊 Latest: {d['growth'].iloc[-1]:.2f}%"
        return InsightResult(True, txt, fig, d, meta={"type":"growth","result":{"avg":avg_growth}})

    def seasonality(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna()
        d['month'] = d['_dt'].dt.month
        d['year'] = d['_dt'].dt.year
        monthly = d.groupby('month')[col].mean().sort_index()
        fig = px.line(monthly.reset_index(), x='month', y=col, title=f"{col} Seasonality")
        peak = monthly.idxmax()
        low = monthly.idxmin()
        txt = f"**Seasonality: {col}**" + NL + NL + f"🔺 Peak: Month {peak} ({monthly[peak]:.2f})" + NL + f"🔻 Low: Month {low} ({monthly[low]:.2f})"
        return InsightResult(True, txt, fig, monthly, meta={"type":"seasonality","result":{"peak":peak,"low":low}})

    def variance(self, metric, by):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        groups = [g[mc].dropna().values for _, g in self.df.groupby(bc)]
        if len(groups) < 2:
            return InsightResult(False, "Need 2+ groups")
        f_stat, p = f_oneway(*groups)
        d = self.df.groupby(bc)[mc].agg(['mean','std','count']).reset_index()
        fig = px.bar(d, x=bc, y='mean', error_y='std', title=f"{mc} Variance by {bc}")
        txt = f"**Variance: {mc} by {bc}**" + NL + NL + f"📊 F={f_stat:.2f} (p={p:.4f})" + NL + ('✅ Significant diff' if p < 0.05 else '❌ No significant diff')
        return InsightResult(True, txt, fig, d, meta={"type":"variance","result":{"f":f_stat,"p":p}})

    def crosstab(self, cat1, cat2):
        c1 = self._find_cat(cat1)
        c2 = self._find_cat(cat2)
        if not c1 or not c2:
            return InsightResult(False, "Categories not found")
        ct = pd.crosstab(self.df[c1], self.df[c2])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return InsightResult(False, "Need 2+ categories each")
        chi2, p, dof, expected = chi2_contingency(ct)
        fig = px.imshow(ct, text_auto=True, aspect="auto", title=f"{c1} × {c2}")
        txt = f"**Cross-tab: {c1} × {c2}**" + NL + NL + f"📊 χ²={chi2:.2f} (p={p:.4f}, df={dof})" + NL + ('✅ Associated' if p < 0.05 else '❌ Independent')
        return InsightResult(True, txt, fig, ct, meta={"type":"crosstab","result":{"chi2":chi2,"p":p}})

    def ranking(self, metric, by, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        pandas_agg = AGG_MAP.get(agg, agg)
        d = self.df.groupby(bc)[mc].agg(pandas_agg).sort_values(ascending=True).reset_index()
        d['rank'] = range(1, len(d)+1)
        fig = px.bar(d, x=mc, y=bc, orientation='h', title=f"{by} Ranked by {agg} {mc}")
        txt = f"**Ranking: {by} by {agg} {mc}**" + NL + NL + f"🥇 #1: {d.iloc[-1][bc]} ({d.iloc[-1][mc]:.2f})" + NL + f"🥉 Last: {d.iloc[0][bc]} ({d.iloc[0][mc]:.2f})"
        return InsightResult(True, txt, fig, d, meta={"type":"ranking"})

    def outlier_table(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        s = self.df[col].dropna()
        q1, q3 = s.quantile([0.25,0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        out = self.df[(self.df[col] < lo) | (self.df[col] > hi)][self.df.columns[:5].tolist() + [col]]
        txt = f"**Outliers: {col}**" + NL + NL + f"Range: [{lo:.2f}, {hi:.2f}]" + NL + f"🔴 {len(out)} outliers found"
        return InsightResult(True, txt, None, out, meta={"type":"outlier_table","result":{"n":len(out),"lo":lo,"hi":hi}})

    def profile_summary(self):
        figs = []
        txt = "**Dataset Profile**" + NL + NL + f"📊 {len(self.df):,} rows × {len(self.df.columns)} cols" + NL + f"📈 Metrics: {len(self.prof.num_cols)} | 🏷️ Categories: {len(self.prof.cat_cols)}"
        if self.prof.num_cols:
            fig = px.box(self.df, y=self.prof.num_cols[0], title=f"{self.prof.num_cols[0]} Overview")
            figs.append(fig)
        return InsightResult(True, txt, figs[0] if figs else None, None, meta={"type":"profile"})

    def auto_insights(self):
        insights = []
        if self.prof.num_cols and self.dt is not None:
            insights.append(self.trend(self.prof.num_cols[0]))
        if len(self.prof.num_cols) >= 2:
            insights.append(self.correlation(self.prof.num_cols[0], self.prof.num_cols[1]))
        if self.prof.num_cols and self.prof.cat_cols:
            insights.append(self.aggregate(self.prof.num_cols[0], [self.prof.cat_cols[0]]))
            insights.append(self.pareto(self.prof.num_cols[0], self.prof.cat_cols[0]))
        if len(self.prof.num_cols) >= 2:
            insights.append(self.segment(self.prof.num_cols[:3], 3))
        if self.prof.num_cols:
            insights.append(self.distribution(self.prof.num_cols[0]))
            insights.append(self.anomaly(self.prof.num_cols[0]))
        return insights


# ============================================================
# QUERY RESOLVER
# ============================================================

class QueryResolver:
    def __init__(self, profiler):
        self.prof = profiler
        self.ctx = []

    def resolve(self, q):
        ql = q.lower()
        intent = self._intent(ql)
        ent = self._entities(ql)
        if self.ctx and not ent.get('metric'):
            for c in reversed(self.ctx):
                if 'metric' in c.get('ent', {}):
                    ent['metric'] = c['ent']['metric']
                    break
        self.ctx.append({"q": q, "intent": intent, "ent": ent})
        self.ctx = self.ctx[-5:]
        return {"intent": intent, "ent": ent}

    def _intent(self, q):
        patterns = {
            "trend": ['trend','direction','going','trajectory','over time'],
            "aggregate": ['total','sum','avg','mean','count','per','by','breakdown'],
            "distribution": ['distribution','histogram','spread','skew'],
            "correlation": ['correlation','correlate','vs','versus','relationship'],
            "anomaly": ['anomaly','outlier','weird','unusual'],
            "segment": ['segment','cluster','group','cohort'],
            "change": ['change point','shift','breakpoint','regime'],
            "pareto": ['pareto','80/20','vital few'],
            "forecast": ['forecast','predict','future','project'],
            "top": ['top','best','worst','rank'],
            "composition": ['composition','share','proportion','pie'],
            "growth": ['growth','growth rate','% change','pct change'],
            "seasonality": ['seasonal','seasonality','monthly pattern'],
            "variance": ['variance','anova','difference between groups'],
            "crosstab": ['crosstab','chi square','association','contingency'],
            "ranking": ['rank','ranking','ordered','sorted'],
            "outliers": ['outlier table','outlier list','extreme values'],
            "profile": ['profile','overview','summary','describe'],
            "auto": ['auto','all insights','full report','everything'],
            "slice": ['sales for','by region','by product','for a','in month','drill down']
        }
        for name, words in patterns.items():
            if any(w in q for w in words):
                return {"type": name, "conf": 0.9}
        return {"type": "auto", "conf": 0.5}

    def _entities(self, q):
        e = {}
        ql = q.lower()
        for c in self.prof.num_cols:
            for p in [c.lower(), c.lower().replace('_',' ')]:
                if p in ql:
                    e['metric'] = c
                    break
        e['categories'] = []
        for c in self.prof.cat_cols:
            patterns = [c.lower(), c.lower().replace('_',' '), c.lower().replace('_','')]
            for p in patterns:
                if p in ql and c not in e['categories']:
                    e['categories'].append(c)
                    break
        if e['categories']:
            e['category'] = e['categories'][0]
        for c in self.prof.num_cols:
            if c != e.get('metric') and (c.lower() in ql or c.lower().replace('_',' ') in ql):
                e['metric2'] = c
                break
        if 'sum' in ql or 'total' in ql:
            e['agg'] = 'sum'
        elif 'avg' in ql or 'average' in ql or 'mean' in ql:
            e['agg'] = 'mean'
        elif 'count' in ql:
            e['agg'] = 'count'
        else:
            e['agg'] = 'sum'
        e['filters'] = {}
        for c in self.prof.cat_cols:
            pattern = rf"(?:for|in|where)\s+{re.escape(c.lower())}\s+(\w+)"
            match = re.search(pattern, ql)
            if match:
                e['filters'][c] = match.group(1)
        nums = re.findall(r'\btop\s+(\d+)\b', ql)
        if nums:
            e['n'] = int(nums[0])
        return e


# ============================================================
# OBSERVABILITY — FIXED
# ============================================================

class Observability:
    def __init__(self):
        self.logs = []

    def log(self, q, intent, op, result, ms):
        iid = hashlib.md5(f"{q}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        entry = {
            "id": iid, "ts": datetime.now().isoformat(),
            "q": q, "intent": intent, "op": op,
            "valid": result.valid if result else False, "ms": ms
        }
        self.logs.append(entry)
        return iid

    def catalog(self):
        n = len(self.logs)
        return {
            "total": n,
            "valid_rate": sum(1 for x in self.logs if x["valid"])/max(1,n),
            "avg_ms": np.mean([x["ms"] for x in self.logs]) if self.logs else 0
        }


# ============================================================
# STREAMLIT UI — BULLETPROOF
# ============================================================

st.set_page_config(page_title="AxiLattice Pro", layout="wide")

st.markdown("""
<style>
.hdr { font-size:1.8rem; font-weight:700; color:#f0f6fc; }
.u { background:#388bfd20; border-radius:10px; padding:10px; margin:6px 0; border-left:3px solid #388bfd; }
.a { background:#23863620; border-radius:10px; padding:10px; margin:6px 0; border-left:3px solid #238636; }
.err { background:#da363320; border-radius:10px; padding:10px; margin:6px 0; border-left:3px solid #da3633; color:#ff7b72; }
</style>
""", unsafe_allow_html=True)

# Initialize ALL session state keys properly
for key, default in [
    ('prof', None), ('cube', None), ('engine', None), ('resolver', None),
    ('chat', []), ('df', None), ('voice', VoiceManager()), ('obs', Observability()),
    ('last_tts', ""), ('pending', None), ('upload_error', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.markdown('<p class="hdr">🧠 AxiLattice Pro</p>', unsafe_allow_html=True)
    st.caption("Live Voice | Multi-Dimensional | Bulletproof v2")

    up = st.file_uploader("📁 Upload CSV/Excel/Parquet", type=['csv','xlsx','parquet'])
    if up:
        try:
            if up.name.endswith('.csv'):
                df = None
                for enc in ['utf-8','utf-8-sig','latin-1','cp1252']:
                    try:
                        up.seek(0)
                        df = pd.read_csv(up, encoding=enc, index_col=False)
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    raise ValueError("Could not decode CSV with any encoding")
            elif up.name.endswith('.parquet'):
                df = pd.read_parquet(up)
            else:
                df = pd.read_excel(up, engine='openpyxl')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Validate we got actual data
            if df is None or df.empty:
                raise ValueError("Uploaded file contains no data")

            st.session_state.df = df
            st.session_state.prof = DataProfiler(df)
            st.session_state.cube = DataCube(df, st.session_state.prof, max_dims=3)
            st.session_state.engine = InsightEngine(st.session_state.prof, st.session_state.cube)
            st.session_state.resolver = QueryResolver(st.session_state.prof)
            st.session_state.upload_error = None
            st.success(f"✅ {len(df):,} rows × {len(df.columns)} cols")

            with st.expander("📋 Schema"):
                for c, p in st.session_state.prof.profs.items():
                    icon = {"temporal":"📅","metric":"📊","categorical":"🏷️","identifier":"🔑","boolean":"☑️","text":"📝"}.get(p.type.value,"❓")
                    st.write(f"{icon} **{c}** ({p.type.value})")
                if st.session_state.cube:
                    st.json(st.session_state.cube.info())
        except Exception as e:
            st.session_state.upload_error = str(e)
            st.error(f"❌ Failed to load: {str(e)}")

    if st.session_state.upload_error:
        st.warning("Fix the upload error to enable all modes.")

    if st.session_state.prof:
        mode = st.radio("Mode", ["🎙️ Ask (Voice/Text)", "📊 Auto-Report", "🔍 Explore", "📈 Observability", "🧊 Slice"])
    else:
        st.info("Upload data to begin")
        mode = None


# Main content area
if not st.session_state.prof:
    st.markdown("## 👋 AxiLattice Pro" + NL + NL + "**Live Voice + Multi-Dimensional Analytics**" + NL + NL + "**Ask naturally:**" + NL + "- *'Profit of laptops in Kolkata for January 2021'*" + NL + "- *'Sales by region by product by month'*" + NL + "- *'Revenue for region North by product'*" + NL + NL + "**Features:**" + NL + "- 🎙️ **Press & Speak** — Live voice recording (no file upload needed)" + NL + "- 🔊 **Voice Answers** — Hear the actual number back" + NL + "- 🧊 **Multi-dimensional** — Slice by any dimensions" + NL + "- 📊 **21 Insight Types** — Trend, Aggregate, Distribution, Correlation, Anomaly, Segmentation, Change Point, Pareto, Forecast, Top-N, Composition, Growth, Seasonality, Variance, Cross-tab, Ranking, Outlier Table, Profile, Auto-Insights, Slice")
    st.stop()

prof = st.session_state.prof
cube = st.session_state.cube
eng = st.session_state.engine
res = st.session_state.resolver
voice = st.session_state.voice
obs = st.session_state.obs

# ============================================================
# ASK MODE — FIXED VOICE FLOW
# ============================================================
if mode and "Ask" in mode:
    st.markdown('<p class="hdr">🎙️ Conversational Analytics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("🎤 Live Voice")
        audio_value = st.audio_input("Press to speak", key="live_audio")
        if audio_value:
            with st.spinner("Transcribing..."):
                audio_bytes = audio_value.read()
                transcribed = voice.transcribe(audio_bytes)
                if not transcribed.startswith("["):
                    st.session_state.pending = transcribed
                    st.success(f"Heard: '{transcribed}'")
                    st.rerun()
                else:
                    st.warning(transcribed)

        if st.session_state.last_tts:
            st.subheader("🔊 Last Answer")
            if st.button("Play Again", key="play_again"):
                audio_bytes = voice.speak(st.session_state.last_tts)
                if audio_bytes:
                    st.markdown(voice.html(audio_bytes), unsafe_allow_html=True)
                else:
                    st.warning("Install gTTS: pip install gtts")

        st.divider()
        st.subheader("Quick Questions")
        sug = []
        if prof.num_cols:
            sug.append(f"Trend in {prof.num_cols[0]}")
            if prof.cat_cols:
                if len(prof.cat_cols) >= 2:
                    sug.append(f"{prof.num_cols[0]} by {prof.cat_cols[0]} by {prof.cat_cols[1]}")
                sug.append(f"Average {prof.num_cols[0]} by {prof.cat_cols[0]}")
        sug.extend(["Auto insights", "Profile summary"])
        for s in sug:
            if st.button(s, key=f"q_{hashlib.md5(s.encode()).hexdigest()[:8]}"):
                st.session_state.pending = s
                st.rerun()

    with c2:
        # Display chat history
        for m in st.session_state.chat[-12:]:
            if m['role'] == 'u':
                st.markdown(f"<div class='u'><b>You:</b> {m['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='a'><b>Axi:</b></div>", unsafe_allow_html=True)
                st.markdown(m['text'])
                if m.get('viz'):
                    st.plotly_chart(m['viz'], use_container_width=True)
                if m.get('data') is not None and isinstance(m['data'], pd.DataFrame):
                    st.dataframe(m['data'].head(20), use_container_width=True)
                if m.get('tts') and m.get('tts') != st.session_state.last_tts:
                    audio_bytes = voice.speak(m['tts'])
                    if audio_bytes:
                        st.markdown(voice.html(audio_bytes), unsafe_allow_html=True)
                    st.session_state.last_tts = m['tts']

        q = st.chat_input("Ask anything...")
        if not q and st.session_state.pending:
            q = st.session_state.pending
            st.session_state.pending = None

        if q:
            import time
            t0 = time.time()
            st.session_state.chat.append({"role": "u", "text": q})

            with st.spinner("Analyzing..."):
                try:
                    r = res.resolve(q)
                    intent = r['intent']['type']
                    e = r['ent']
                    result = None

                    if intent == "trend":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.trend(m) if m else InsightResult(False, "No metric found")
                    elif intent in ["aggregate", "slice"]:
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        cats = e.get('categories', [])
                        if not cats and e.get('category'):
                            cats = [e['category']]
                        if not cats and prof.cat_cols:
                            cats = [prof.cat_cols[0]]
                        if m and cats:
                            result = eng.slice_data(m, cats, e.get('filters'), e.get('agg', 'sum'))
                        else:
                            result = InsightResult(False, "Need metric and dimensions")
                    elif intent == "distribution":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.distribution(m) if m else InsightResult(False, "No metric")
                    elif intent == "correlation":
                        m1 = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        m2 = e.get('metric2', prof.num_cols[1] if len(prof.num_cols) > 1 else None)
                        result = eng.correlation(m1, m2) if m1 and m2 else InsightResult(False, "Need 2 metrics")
                    elif intent == "corr_matrix":
                        result = eng.corr_matrix()
                    elif intent == "anomaly":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.anomaly(m) if m else InsightResult(False, "No metric")
                    elif intent == "segment":
                        ms = [e.get('metric', prof.num_cols[0])] + ([e.get('metric2')] if e.get('metric2') else [])
                        ms = [m for m in ms if m]
                        if not ms and prof.num_cols:
                            ms = prof.num_cols[:3]
                        result = eng.segment(ms, 3) if len(ms) >= 2 else InsightResult(False, "Need 2+ metrics")
                    elif intent == "change":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.change_point(m) if m else InsightResult(False, "No metric")
                    elif intent == "pareto":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        result = eng.pareto(m, c) if m and c else InsightResult(False, "Need metric+category")
                    elif intent == "forecast":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.forecast(m, 6) if m else InsightResult(False, "No metric")
                    elif intent == "top":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        n = e.get('n', 5)
                        result = eng.top_n(m, c, n, e.get('agg', 'sum')) if m and c else InsightResult(False, "Need metric+category")
                    elif intent == "composition":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        result = eng.composition(m, c, e.get('agg', 'sum')) if m and c else InsightResult(False, "Need metric+category")
                    elif intent == "growth":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.growth(m) if m else InsightResult(False, "No metric")
                    elif intent == "seasonality":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.seasonality(m) if m else InsightResult(False, "No metric")
                    elif intent == "variance":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        result = eng.variance(m, c) if m and c else InsightResult(False, "Need metric+category")
                    elif intent == "crosstab":
                        c1 = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        c2 = e.get('metric', prof.cat_cols[1] if len(prof.cat_cols) > 1 else None)
                        result = eng.crosstab(c1, c2) if c1 and c2 else InsightResult(False, "Need 2 categories")
                    elif intent == "ranking":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                        result = eng.ranking(m, c, e.get('agg', 'sum')) if m and c else InsightResult(False, "Need metric+category")
                    elif intent == "outliers":
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        result = eng.outlier_table(m) if m else InsightResult(False, "No metric")
                    elif intent == "profile":
                        result = eng.profile_summary()
                    elif intent == "auto":
                        results = eng.auto_insights()
                        for ri in results:
                            st.session_state.chat.append({
                                "role": "a", "text": ri.text, "viz": ri.viz, "data": ri.data,
                                "tts": re.sub(r'[\*#]', '', ri.text).replace('\n', ' ')[:500]
                            })
                        ms = int((time.time() - t0) * 1000)
                        obs.log(q, r['intent'], "auto", None, ms)
                        st.rerun()
                        st.stop()
                    else:
                        m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                        cats = e.get('categories', [])
                        if m and cats:
                            result = eng.slice_data(m, cats, e.get('filters'), e.get('agg', 'sum'))
                        else:
                            result = InsightResult(False, "🤔 Try: 'Profit of laptops in Kolkata' or 'Sales by region by product'")

                    ms = int((time.time() - t0) * 1000)
                    iid = obs.log(q, r['intent'], intent, result, ms)

                    if result and result.valid:
                        tts_base = re.sub(r'[\*#]', '', result.text).replace('\n', ' ').strip()[:500]
                        if result.data is not None and isinstance(result.data, pd.DataFrame) and len(result.data) == 1:
                            val = result.data.iloc[0, -1]
                            tts_base = f"The answer is {val:.2f}"
                        st.session_state.last_tts = tts_base
                    else:
                        tts_base = result.text if result else "Sorry, I couldn't understand that."

                    st.session_state.chat.append({
                        "role": "a", "text": result.text if result else "",
                        "viz": result.viz if result else None,
                        "data": result.data if result else None,
                        "tts": tts_base, "id": iid
                    })
                    st.rerun()
                except Exception as ex:
                    ms = int((time.time() - t0) * 1000)
                    obs.log(q, {"type":"error"}, "exception", InsightResult(False, str(ex)), ms)
                    st.session_state.chat.append({
                        "role": "a",
                        "text": f"❌ Error processing query: {str(ex)}",
                        "viz": None, "data": None, "tts": "Sorry, an error occurred."
                    })
                    st.rerun()


# ============================================================
# AUTO-REPORT MODE — FIXED NONE GUARD
# ============================================================
elif mode and "Auto" in mode:
    st.markdown('<p class="hdr">📊 Auto-Report</p>', unsafe_allow_html=True)
    if not prof.num_cols:
        st.warning("No numeric columns detected. Upload data with metrics.")
    else:
        sel = st.multiselect("Metrics", prof.num_cols, default=prof.num_cols[:min(3, len(prof.num_cols))])
        if sel and st.button("🚀 Generate"):
            for m in sel:
                with st.expander(f"📈 {m}", expanded=True):
                    for fn, title in [(eng.trend, "Trend"), (eng.distribution, "Distribution"),
                                      (eng.anomaly, "Anomaly"), (eng.growth, "Growth")]:
                        try:
                            r = fn(m)
                            if r.valid:
                                st.subheader(title)
                                st.markdown(r.text)
                                if r.viz:
                                    st.plotly_chart(r.viz, use_container_width=True)
                            else:
                                st.caption(f"{title}: {r.warnings[0] if r.warnings else 'N/A'}")
                        except Exception as ex:
                            st.error(f"{title} failed: {ex}")

# ============================================================
# EXPLORE MODE
# ============================================================
elif mode and "Explore" in mode:
    st.markdown('<p class="hdr">🔍 Explore</p>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["📋 Data", "📊 Dist", "🧊 Cube", "🔗 Corr"])
    with t1:
        st.dataframe(st.session_state.df, use_container_width=True)
    with t2:
        if prof.num_cols:
            c = st.selectbox("Metric", prof.num_cols)
            st.plotly_chart(px.histogram(st.session_state.df, x=c), use_container_width=True)
        else:
            st.info("No numeric columns available.")
    with t3:
        if st.session_state.cube:
            st.json(st.session_state.cube.info())
            for k, cub in st.session_state.cube.cuboids.items():
                with st.expander(f"Cuboid: {k}"):
                    st.write(f"Rows: {cub.rows}, Dims: {cub.dims}")
                    st.dataframe(cub.df.head(10), use_container_width=True)
        else:
            st.warning("Cube not initialized. Please re-upload data.")
    with t4:
        if len(prof.num_cols) >= 2:
            st.plotly_chart(px.imshow(st.session_state.df[prof.num_cols].corr(), text_auto=".2f"), use_container_width=True)
        else:
            st.info("Need 2+ numeric columns for correlation matrix.")

# ============================================================
# OBSERVABILITY MODE — FIXED
# ============================================================
elif mode and "Observe" in mode:
    st.markdown('<p class="hdr">📈 Observability</p>', unsafe_allow_html=True)
    cat = obs.catalog()
    st.json(cat)
    if obs.logs:
        st.subheader("Recent Queries")
        for lg in obs.logs[-15:]:
            with st.expander(f"{lg['q'][:50]}... ({lg['ms']}ms)"):
                st.json(lg)
    else:
        st.info("No queries logged yet. Run some analytics first.")

# ============================================================
# SLICE MODE — FIXED DIMENSION PICKUP
# ============================================================
elif mode and "Slice" in mode:
    st.markdown('<p class="hdr">🧊 Multi-Dimensional Slice</p>', unsafe_allow_html=True)
    st.caption("Point-and-click slice & dice")

    if not prof.num_cols:
        st.error("No metrics found. Please upload data with numeric columns.")
    elif not prof.cat_cols:
        st.error("No dimensions found. Please upload data with categorical columns.")
    else:
        m = st.selectbox("Metric", prof.num_cols)
        dims = st.multiselect("Dimensions", prof.cat_cols, default=prof.cat_cols[:min(2, len(prof.cat_cols))])
        agg = st.selectbox("Aggregation", ["sum", "avg", "count", "min", "max"], index=0)

        st.subheader("Filters")
        filters = {}
        if prof.cat_cols:
            cols = st.columns(min(3, len(prof.cat_cols)))
            for i, c in enumerate(prof.cat_cols[:3]):
                with cols[i % 3]:
                    vals = ["All"] + sorted(st.session_state.df[c].dropna().unique().tolist())
                    sel = st.selectbox(f"{c}", vals, index=0)
                    if sel != "All":
                        filters[c] = sel

        if m and dims and st.button("🔍 Slice"):
            try:
                r = eng.slice_data(m, dims, filters if filters else None, agg)
                if r.valid:
                    st.markdown(r.text)
                    if r.viz:
                        st.plotly_chart(r.viz, use_container_width=True)
                    if r.data is not None:
                        st.dataframe(r.data, use_container_width=True)
                        csv = r.data.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, f"slice_{m}.csv", "text/csv")
                    if st.session_state.last_tts:
                        audio_bytes = voice.speak(st.session_state.last_tts)
                        if audio_bytes:
                            st.markdown(voice.html(audio_bytes), unsafe_allow_html=True)
                else:
                    st.error(r.warnings[0] if r.warnings else "Error")
            except Exception as ex:
                st.error(f"Slice failed: {ex}")

st.divider()
st.caption("AxiLattice Pro v2 | Live Voice | Multi-Dimensional | 21 Insight Types | Bulletproof")

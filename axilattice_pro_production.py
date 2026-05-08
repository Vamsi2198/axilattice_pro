"""
AxiLattice Pro — Production-Grade Insight Engine
"Works on Any Data" — Full Insight Suite
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
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
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# VOICE I/O
# ============================================================

class VoiceManager:
    def __init__(self):
        self.stt = self._check('speech_recognition')
        self.tts = self._check('gtts')

    def _check(self, pkg):
        try:
            __import__(pkg)
            return True
        except:
            return False

    def transcribe(self, audio: bytes) -> str:
        if not self.stt:
            return "[pip install SpeechRecognition]"
        import speech_recognition as sr
        try:
            r = sr.Recognizer()
            with io.BytesIO(audio) as f:
                with sr.AudioFile(f) as src:
                    return r.recognize_google(r.record(src))
        except Exception as e:
            return f"[STT: {e}]"

    def speak(self, text: str) -> Optional[bytes]:
        if not self.tts:
            return None
        from gtts import gTTS
        try:
            fp = io.BytesIO()
            gTTS(text=text[:500], lang='en').write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except:
            return None

    def html(self, audio: bytes) -> str:
        b64 = base64.b64encode(audio).decode()
        return f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>'


# ============================================================
# CUBE ARCHITECTURE
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

class DataCube:
    def __init__(self, df, profiler):
        self.df = df
        self.prof = profiler
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
        names = [d.name for d in self.dims]
        for d in names:
            self._make((d,))
        for i, a in enumerate(names):
            for b in names[i+1:]:
                c1 = next(x.card for x in self.dims if x.name == a)
                c2 = next(x.card for x in self.dims if x.name == b)
                if c1 * c2 < 1e5:
                    self._make((a, b))
        self._make(())

    def _make(self, dims: Tuple):
        key = "_".join(dims) if dims else "total"
        cols = [next(d.col for d in self.dims if d.name == n) for n in dims]
        ad = {}
        for m in self.meas:
            for a in m.aggs:
                ad[f"{m.name}_{a}"] = (m.col, a)
        if cols:
            df = self.df.groupby(cols, observed=True).agg(**ad).reset_index()
        else:
            df = pd.DataFrame({f"{m.name}_{a}": [self.df[m.col].agg(a)] for m in self.meas for a in m.aggs})
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
            ad = {m: (m, a) for m, a in zip(meas, aggs)}
            return d.groupby(dims, observed=True).agg(**ad).reset_index()
        return pd.DataFrame({f"{m}_{a}": [d[m].agg(a)] for m, a in zip(meas, aggs)})

    def info(self):
        n = len(self.dims)
        total = max(1, n*(n-1)//2 + n + 1)
        return {
            "dims": n, "meas": len(self.meas),
            "cuboids": len(self.cuboids),
            "rows": sum(c.rows for c in self.cuboids.values()),
            "coverage": len(self.cuboids)/total
        }


# ============================================================
# DATA PROFILER
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
        if re.search(r'year', s.name, re.I) and dt in ['int64','float64']:
            if 1900 < s.min() < 2100:
                return ColType.TEMP, 0.8, {"part":"year"}, ["Check Month"]
        if re.search(r'month', s.name, re.I):
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
                except:
                    pass
            try:
                if pd.to_datetime(sm, errors='coerce').notna().mean() > 0.8:
                    return True
            except:
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
                    except:
                        pass
        if self.temp_col:
            return pd.to_datetime(self.df[self.temp_col], errors='coerce')
        return None


# ============================================================
# INSIGHT ENGINE — ALL TYPES
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

    # 1. TREND
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
        return InsightResult(True,
            f"**Trend: {col}**\n\n📈 {dir.upper()} | τ={r['tau']} (p={r['p']})\n📉 Slope: {r['slope']}/period\n💰 Change: {r['chg']}%",
            fig, d, meta={"type":"trend","result":r})

    # 2. AGGREGATE
    def aggregate(self, metric, by, agg="mean"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        # Cube
        try:
            rdf = self.cube.query([bc], [mc], [agg])
            src = "cube"
        except:
            rdf = self.df.groupby(bc)[mc].agg(agg).reset_index()
            rdf.columns = [bc, f"{agg}_{mc}"]
            src = "raw"
        vals = rdf.to_dict('records')
        fig = px.bar(rdf, x=bc, y=rdf.columns[1], title=f"{agg.title()} {mc} by {bc}")
        return InsightResult(True,
            f"**{agg.title()} {mc} by {bc}**\n\nGroups: {len(rdf)} ({src})\n" + "\n".join([f"• {v[bc]}: **{v[rdf.columns[1]]:.2f}**" for v in vals[:5]]),
            fig, rdf, meta={"type":"aggregate","result":{"agg":agg,"by":bc,"metric":mc,"n":len(rdf),"src":src}})

    # 3. DISTRIBUTION
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
        return InsightResult(True,
            f"**Distribution: {col}**\n\n📊 Median: {q2:.2f} | IQR: {iqr:.2f}\n📈 Skew: {skew:.2f} | Kurt: {kurt:.2f}\n⚠️ Outliers: {len(out)} ({len(out)/len(s)*100:.1f}%)",
            fig, s, meta={"type":"distribution","result":{"median":q2,"iqr":iqr,"skew":skew,"outliers":len(out)}})

    # 4. CORRELATION
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
        return InsightResult(True,
            f"**Correlation: {c1} vs {c2}**\n\n🔗 ρ = {r:.4f} (p={p:.4f})\n📊 Strength: {'strong' if abs(r)>0.7 else 'moderate' if abs(r)>0.4 else 'weak'} {'positive' if r>0 else 'negative'}",
            fig, d, meta={"type":"correlation","result":{"rho":r,"p":p}})

    # 5. CORRELATION MATRIX
    def corr_matrix(self):
        if len(self.prof.num_cols) < 2:
            return InsightResult(False, "Need 2+ metrics")
        d = self.df[self.prof.num_cols].corr()
        fig = px.imshow(d, text_auto=".2f", aspect="auto", title="Correlation Matrix")
        # Find strongest
        mask = np.triu(np.ones_like(d, dtype=bool), k=1)
        corr_vals = d.where(mask).stack().sort_values(key=abs, ascending=False)
        top = corr_vals.head(3)
        txt = "**Correlation Matrix**\n\nTop pairs:\n"
        for (a,b), v in top.items():
            txt += f"• {a} ↔ {b}: {v:.3f}\n"
        return InsightResult(True, txt, fig, d, meta={"type":"corr_matrix","top":top.to_dict()})

    # 6. ANOMALY DETECTION
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
        return InsightResult(True,
            f"**Anomaly Detection: {col}**\n\n🔴 Anomalies: {n_anom} ({n_anom/len(d)*100:.1f}%)\n✅ Normal: {len(d)-n_anom}",
            fig, d, meta={"type":"anomaly","result":{"anomalies":n_anom}})

    # 7. SEGMENTATION (K-Means)
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
        txt = f"**Segmentation (k={n})**\n\n"
        for _, row in centers.iterrows():
            txt += f"Cluster {int(row['cluster'])}: " + " | ".join([f"{c}={row[c]:.2f}" for c in cols]) + "\n"
        return InsightResult(True, txt, fig, d, meta={"type":"segment","centers":centers.to_dict()})

    # 8. CHANGE POINT
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
        return InsightResult(True,
            f"**Change Point: {col}**\n\n🎯 Date: {cp_date}\n📉 Before: {before_mean:.2f} | After: {after_mean:.2f}\n📊 Shift: {((after_mean/before_mean-1)*100):.1f}%",
            fig, d, meta={"type":"change_point","result":{"date":str(cp_date),"before":before_mean,"after":after_mean}})

    # 9. PARETO
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
        return InsightResult(True,
            f"**Pareto: {mc} by {bc}**\n\n📊 Top {n80} of {len(d)} groups = 80% of total\n🏆 Leader: {d.iloc[0][bc]} ({d.iloc[0][mc]:.2f})",
            fig, d, meta={"type":"pareto","result":{"n80":n80,"total":len(d)}})

    # 10. FORECAST (Simple)
    def forecast(self, metric, periods=6):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        if self.dt is None:
            return InsightResult(False, "No temporal column")
        d = self.df[[col,'_dt']].dropna().sort_values('_dt')
        if len(d) < 6:
            return InsightResult(False, "Need 6+ points")
        # Simple exponential smoothing
        alpha = 0.3
        fcast = [d[col].iloc[0]]
        for i in range(1, len(d)):
            fcast.append(alpha * d[col].iloc[i-1] + (1-alpha) * fcast[-1])
        last_date = d['_dt'].iloc[-1]
        freq = (d['_dt'].iloc[-1] - d['_dt'].iloc[-2]).days if len(d) > 1 else 30
        future_dates = [last_date + timedelta(days=freq*(i+1)) for i in range(periods)]
        last_level = fcast[-1]
        future_vals = [last_level] * periods  # flat forecast for simplicity
        hist = d.copy()
        hist['type'] = 'Historical'
        fut = pd.DataFrame({'_dt': future_dates, col: future_vals, 'type': 'Forecast'})
        combined = pd.concat([hist, fut])
        fig = px.line(combined, x='_dt', y=col, color='type', title=f"{col} Forecast")
        return InsightResult(True,
            f"**Forecast: {col}**\n\n📈 Next {periods} periods: ~{last_level:.2f} each\n⚠️ Simple model — use with caution",
            fig, combined, meta={"type":"forecast","result":{"next":last_level,"periods":periods}})

    # 11. TOP-N
    def top_n(self, metric, by, n=5, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        d = self.df.groupby(bc)[mc].agg(agg).sort_values(ascending=False).head(n).reset_index()
        fig = px.bar(d, x=bc, y=mc, title=f"Top {n} {by} by {agg} {mc}")
        return InsightResult(True,
            f"**Top {n} {by} by {agg} {mc}**\n\n" + "\n".join([f"{i+1}. {row[bc]}: **{row[mc]:.2f}**" for i, row in d.iterrows()]),
            fig, d, meta={"type":"top_n","result":{"n":n,"agg":agg}})

    # 12. COMPOSITION
    def composition(self, metric, by, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        d = self.df.groupby(bc)[mc].agg(agg).reset_index()
        fig = px.pie(d, names=bc, values=mc, title=f"{mc} Composition by {bc}")
        total = d[mc].sum()
        return InsightResult(True,
            f"**Composition: {mc} by {bc}**\n\nTotal: {total:.2f}\n" + "\n".join([f"• {row[bc]}: {row[mc]/total*100:.1f}%" for _, row in d.iterrows()]),
            fig, d, meta={"type":"composition","result":{"total":total}})

    # 13. GROWTH RATE
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
        return InsightResult(True,
            f"**Growth: {col}**\n\n📈 Avg growth: {avg_growth:.2f}%\n📊 Latest: {d['growth'].iloc[-1]:.2f}%",
            fig, d, meta={"type":"growth","result":{"avg":avg_growth}})

    # 14. SEASONALITY
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
        return InsightResult(True,
            f"**Seasonality: {col}**\n\n🔺 Peak: Month {peak} ({monthly[peak]:.2f})\n🔻 Low: Month {low} ({monthly[low]:.2f})",
            fig, monthly, meta={"type":"seasonality","result":{"peak":peak,"low":low}})

    # 15. VARIANCE ANALYSIS
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
        return InsightResult(True,
            f"**Variance: {mc} by {bc}**\n\n📊 F={f_stat:.2f} (p={p:.4f})\n{'✅ Significant diff' if p < 0.05 else '❌ No significant diff'}",
            fig, d, meta={"type":"variance","result":{"f":f_stat,"p":p}})

    # 16. CROSS-TAB / CHI-SQUARE
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
        return InsightResult(True,
            f"**Cross-tab: {c1} × {c2}**\n\n📊 χ²={chi2:.2f} (p={p:.4f}, df={dof})\n{'✅ Associated' if p < 0.05 else '❌ Independent'}",
            fig, ct, meta={"type":"crosstab","result":{"chi2":chi2,"p":p}})

    # 17. RANKING
    def ranking(self, metric, by, agg="sum"):
        mc = self._find_num(metric)
        bc = self._find_cat(by)
        if not mc or not bc:
            return InsightResult(False, "Columns not found")
        d = self.df.groupby(bc)[mc].agg(agg).sort_values(ascending=True).reset_index()
        d['rank'] = range(1, len(d)+1)
        fig = px.bar(d, x=mc, y=bc, orientation='h', title=f"{by} Ranked by {agg} {mc}")
        return InsightResult(True,
            f"**Ranking: {by} by {agg} {mc}**\n\n🥇 #1: {d.iloc[-1][bc]} ({d.iloc[-1][mc]:.2f})\n🥉 Last: {d.iloc[0][bc]} ({d.iloc[0][mc]:.2f})",
            fig, d, meta={"type":"ranking"})

    # 18. OUTLIER TABLE
    def outlier_table(self, metric):
        col = self._find_num(metric)
        if not col:
            return InsightResult(False, f"Metric '{metric}' not found")
        s = self.df[col].dropna()
        q1, q3 = s.quantile([0.25,0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        out = self.df[(self.df[col] < lo) | (self.df[col] > hi)][self.df.columns[:5].tolist() + [col]]
        return InsightResult(True,
            f"**Outliers: {col}**\n\nRange: [{lo:.2f}, {hi:.2f}]\n🔴 {len(out)} outliers found",
            None, out, meta={"type":"outlier_table","result":{"n":len(out),"lo":lo,"hi":hi}})

    # 19. PROFILE SUMMARY
    def profile_summary(self):
        figs = []
        txt = "**Dataset Profile**\n\n"
        txt += f"📊 {len(self.df):,} rows × {len(self.df.columns)} cols\n"
        txt += f"📈 Metrics: {len(self.prof.num_cols)} | 🏷️ Categories: {len(self.prof.cat_cols)}\n"
        if self.prof.num_cols:
            fig = px.box(self.df, y=self.prof.num_cols[0], title=f"{self.prof.num_cols[0]} Overview")
            figs.append(fig)
        return InsightResult(True, txt, figs[0] if figs else None, None, meta={"type":"profile"})

    # 20. AUTO-INSIGHTS (runs all applicable)
    def auto_insights(self):
        insights = []
        if self.prof.num_cols and self.dt is not None:
            insights.append(self.trend(self.prof.num_cols[0]))
        if len(self.prof.num_cols) >= 2:
            insights.append(self.correlation(self.prof.num_cols[0], self.prof.num_cols[1]))
        if self.prof.num_cols and self.prof.cat_cols:
            insights.append(self.aggregate(self.prof.num_cols[0], self.prof.cat_cols[0]))
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
            "auto": ['auto','all insights','full report','everything']
        }
        for name, words in patterns.items():
            if any(w in q for w in words):
                return {"type": name, "conf": 0.9}
        return {"type": "auto", "conf": 0.5}

    def _entities(self, q):
        e = {}
        for c in self.prof.num_cols:
            for p in [c.lower(), c.lower().replace('_',' ')]:
                if p in q:
                    e['metric'] = c
                    break
        for c in self.prof.cat_cols:
            if c.lower() in q or c.lower().replace('_',' ') in q:
                e['category'] = c
                break
        for c in self.prof.num_cols:
            if c != e.get('metric') and (c.lower() in q or c.lower().replace('_',' ') in q):
                e['metric2'] = c
                break
        if 'sum' in q or 'total' in q:
            e['agg'] = 'sum'
        elif 'avg' in q or 'average' in q or 'mean' in q:
            e['agg'] = 'mean'
        elif 'count' in q:
            e['agg'] = 'count'
        else:
            e['agg'] = 'mean'
        # Extract numbers for top-n
        import re as re_mod
        nums = re_mod.findall(r'\btop\s+(\d+)\b', q)
        if nums:
            e['n'] = int(nums[0])
        return e


# ============================================================
# OBSERVABILITY
# ============================================================

class Observability:
    def __init__(self):
        self.logs = []

    def log(self, q, intent, op, result, ms):
        iid = hashlib.md5(f"{q}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        self.logs.append({
            "id": iid, "ts": datetime.now().isoformat(),
            "q": q, "intent": intent, "op": op,
            "valid": result.valid if result else False, "ms": ms
        })
        return iid

    def catalog(self):
        n = len(self.logs)
        return {
            "total": n,
            "valid_rate": sum(1 for x in self.logs if x["valid"])/max(1,n),
            "avg_ms": np.mean([x["ms"] for x in self.logs]) if self.logs else 0
        }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AxiLattice Pro", layout="wide")

st.markdown("""
<style>
.hdr { font-size:1.8rem; font-weight:700; color:#f0f6fc; }
.u { background:#388bfd20; border-radius:10px; padding:10px; margin:6px 0; border-left:3px solid #388bfd; }
.a { background:#23863620; border-radius:10px; padding:10px; margin:6px 0; border-left:3px solid #238636; }
</style>
""", unsafe_allow_html=True)

# State
for k in ['prof','cube','engine','resolver','chat','df','voice','obs']:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.voice is None:
    st.session_state.voice = VoiceManager()
if st.session_state.obs is None:
    st.session_state.obs = Observability()

# Sidebar
with st.sidebar:
    st.markdown('<p class="hdr">🧠 AxiLattice Pro</p>', unsafe_allow_html=True)
    st.caption("Production Insight Engine | 20 Insight Types")

    up = st.file_uploader("📁 Upload", type=['csv','xlsx','parquet'])
    if up:
        try:
            if up.name.endswith('.csv'):
                for enc in ['utf-8','utf-8-sig','latin-1','cp1252']:
                    try:
                        up.seek(0)
                        df = pd.read_csv(up, encoding=enc)
                        break
                    except:
                        continue
            elif up.name.endswith('.parquet'):
                df = pd.read_parquet(up)
            else:
                df = pd.read_excel(up, engine='openpyxl')
            st.session_state.df = df
            st.session_state.prof = DataProfiler(df)
            st.session_state.cube = DataCube(df, st.session_state.prof)
            st.session_state.engine = InsightEngine(st.session_state.prof, st.session_state.cube)
            st.session_state.resolver = QueryResolver(st.session_state.prof)
            st.success(f"✅ {len(df):,} × {len(df.columns)}")
            with st.expander("Schema"):
                for c, p in st.session_state.prof.profs.items():
                    icon = {"temporal":"📅","metric":"📊","categorical":"🏷️","identifier":"🔑","boolean":"☑️","text":"📝"}.get(p.type.value,"❓")
                    st.write(f"{icon} **{c}** ({p.type.value})")
                st.json(st.session_state.cube.info())
        except Exception as e:
            st.error(f"❌ {e}")

    if st.session_state.prof:
        mode = st.radio("Mode", ["🎙️ Ask","📊 Auto","🔍 Explore","📈 Observe"])
    else:
        st.info("Upload data")

# Main
if not st.session_state.prof:
    st.markdown("## 👋 AxiLattice Pro\n\n**20 Production Insight Types:**\n- Trend, Aggregate, Distribution, Correlation, Corr-Matrix\n- Anomaly, Segmentation, Change Point, Pareto, Forecast\n- Top-N, Composition, Growth, Seasonality, Variance\n- Cross-tab, Ranking, Outlier Table, Profile, Auto-Insights")
    st.stop()

prof = st.session_state.prof
cube = st.session_state.cube
eng = st.session_state.engine
res = st.session_state.resolver
voice = st.session_state.voice
obs = st.session_state.obs

if "Ask" in mode:
    st.markdown('<p class="hdr">🎙️ Conversational Analytics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1,3])
    with c1:
        st.subheader("Voice")
        af = st.file_uploader("🎤 Voice", type=['wav','mp3','m4a'])
        if af:
            with st.spinner("Transcribing..."):
                txt = voice.transcribe(af.read())
                st.session_state.pending = txt
                st.success(f"Heard: '{txt}'")
        if st.session_state.chat and st.session_state.chat[-1]['role'] == 'a':
            lt = st.session_state.chat[-1].get('tts','')
            if lt and st.button("🔊 Play"):
                ab = voice.speak(lt)
                if ab:
                    st.markdown(voice.html(ab), unsafe_allow_html=True)
                else:
                    st.warning("pip install gtts")
        st.divider()
        st.subheader("Quick")
        sug = []
        if prof.num_cols:
            sug.append(f"Trend in {prof.num_cols[0]}")
            sug.append(f"Distribution of {prof.num_cols[0]}")
            sug.append(f"Anomaly in {prof.num_cols[0]}")
        if prof.num_cols and prof.cat_cols:
            sug.append(f"Average {prof.num_cols[0]} by {prof.cat_cols[0]}")
            sug.append(f"Pareto {prof.num_cols[0]} by {prof.cat_cols[0]}")
        if len(prof.num_cols) >= 2:
            sug.append(f"Correlation {prof.num_cols[0]} vs {prof.num_cols[1]}")
        sug.extend(["Auto insights","Profile summary"])
        for s in sug:
            if st.button(s, key=f"q_{s[:15]}"):
                st.session_state.pending = s

    with c2:
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

        q = st.chat_input("Ask anything...")
        if not q and 'pending' in st.session_state:
            q = st.session_state.pending
            del st.session_state.pending

        if q:
            import time
            t0 = time.time()
            st.session_state.chat.append({"role":"u","text":q})

            with st.spinner("Analyzing..."):
                r = res.resolve(q)
                intent = r['intent']['type']
                e = r['ent']
                result = None

                # Route to engine
                if intent == "trend":
                    m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                    result = eng.trend(m) if m else InsightResult(False, "No metric")
                elif intent == "aggregate":
                    m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                    c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                    result = eng.aggregate(m, c, e.get('agg','mean')) if m and c else InsightResult(False, "Need metric+category")
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
                    result = eng.top_n(m, c, n, e.get('agg','sum')) if m and c else InsightResult(False, "Need metric+category")
                elif intent == "composition":
                    m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                    c = e.get('category', prof.cat_cols[0] if prof.cat_cols else None)
                    result = eng.composition(m, c, e.get('agg','sum')) if m and c else InsightResult(False, "Need metric+category")
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
                    result = eng.ranking(m, c, e.get('agg','sum')) if m and c else InsightResult(False, "Need metric+category")
                elif intent == "outliers":
                    m = e.get('metric', prof.num_cols[0] if prof.num_cols else None)
                    result = eng.outlier_table(m) if m else InsightResult(False, "No metric")
                elif intent == "profile":
                    result = eng.profile_summary()
                elif intent == "auto":
                    results = eng.auto_insights()
                    for ri in results:
                        st.session_state.chat.append({
                            "role":"a","text":ri.text,"viz":ri.viz,"data":ri.data,"tts":re.sub(r'[\*#]','',ri.text).replace('\n',' ')[:500]
                        })
                    ms = int((time.time()-t0)*1000)
                    obs.log(q, r['intent'], "auto", None, ms)
                    st.rerun()
                    st.stop()
                else:
                    result = InsightResult(False, f"🤔 Try: 'Trend in {prof.num_cols[0] if prof.num_cols else 'sales'}' or 'Auto insights'")

                ms = int((time.time()-t0)*1000)
                iid = obs.log(q, r['intent'], intent, result, ms)
                tts = re.sub(r'[\*#]','',result.text).replace('\n',' ').strip()[:500] if result else ""

                st.session_state.chat.append({
                    "role":"a","text":result.text if result else "","viz":result.viz if result else None,
                    "data":result.data if result else None,"tts":tts,"id":iid
                })
                st.rerun()

elif "Auto" in mode:
    st.markdown('<p class="hdr">📊 Auto-Report</p>', unsafe_allow_html=True)
    sel = st.multiselect("Metrics", prof.num_cols, default=prof.num_cols[:min(3,len(prof.num_cols))])
    if sel and st.button("🚀 Generate"):
        for m in sel:
            with st.expander(f"📈 {m}", expanded=True):
                for fn, title in [(eng.trend,"Trend"),(eng.distribution,"Distribution"),(eng.anomaly,"Anomaly"),(eng.growth,"Growth")]:
                    r = fn(m)
                    if r.valid:
                        st.subheader(title)
                        st.markdown(r.text)
                        if r.viz:
                            st.plotly_chart(r.viz, use_container_width=True)
                    else:
                        st.caption(f"{title}: {r.warnings[0] if r.warnings else 'N/A'}")

elif "Explore" in mode:
    st.markdown('<p class="hdr">🔍 Explore</p>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["📋 Data","📊 Dist","🧊 Cube","🔗 Corr"])
    with t1:
        st.dataframe(st.session_state.df, use_container_width=True)
    with t2:
        if prof.num_cols:
            c = st.selectbox("Metric", prof.num_cols)
            st.plotly_chart(px.histogram(st.session_state.df, x=c), use_container_width=True)
    with t3:
        st.json(cube.info())
        for k, cub in cube.cuboids.items():
            with st.expander(f"Cuboid: {k}"):
                st.write(f"Rows: {cub.rows}, Dims: {cub.dims}")
                st.dataframe(cub.df.head(10), use_container_width=True)
    with t4:
        if len(prof.num_cols) >= 2:
            st.plotly_chart(px.imshow(st.session_state.df[prof.num_cols].corr(), text_auto=".2f"), use_container_width=True)

elif "Observe" in mode:
    st.markdown('<p class="hdr">📈 Observability</p>', unsafe_allow_html=True)
    st.json(obs.catalog())
    if obs.logs:
        st.subheader("Recent")
        for lg in obs.logs[-15:]:
            with st.expander(f"{lg['q'][:50]}... ({lg['ms']}ms)"):
                st.json(lg)

st.divider()
st.caption("AxiLattice Pro | 20 Insight Types | Production-Ready")

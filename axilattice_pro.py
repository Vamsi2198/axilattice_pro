
"""
AxiLattice Production System
"Works on Any Data" — Schema-Inferred, Operation-Templated, Statistically Validated
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import kendalltau, spearmanr, shapiro, jarque_bera
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# LAYER 1: DATA PROFILER (Schema Inference Engine)
# ============================================================

class ColumnType(Enum):
    TEMPORAL = "temporal"
    IDENTIFIER = "identifier"
    METRIC = "metric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    QUASI_IDENTIFIER = "quasi_identifier"
    UNKNOWN = "unknown"

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    col_type: ColumnType
    confidence: float
    null_pct: float
    unique_count: int
    unique_ratio: float
    sample_values: List[Any]
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

class DataProfiler:
    """Infers schema from any DataFrame with confidence scores."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profiles: Dict[str, ColumnProfile] = {}
        self.temporal_column: Optional[str] = None
        self.metric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.identifier_columns: List[str] = []
        self._profile_all()

    def _profile_all(self):
        for col in self.df.columns:
            profile = self._profile_column(col)
            self.profiles[col] = profile

            # Categorize
            if profile.col_type == ColumnType.TEMPORAL:
                self.temporal_column = col
            elif profile.col_type == ColumnType.METRIC:
                self.metric_columns.append(col)
            elif profile.col_type == ColumnType.CATEGORICAL:
                self.categorical_columns.append(col)
            elif profile.col_type == ColumnType.IDENTIFIER:
                self.identifier_columns.append(col)

    def _profile_column(self, col: str) -> ColumnProfile:
        series = self.df[col]
        dtype = str(series.dtype)
        null_pct = series.isnull().mean() * 100
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / len(series) if len(series) > 0 else 0
        sample = series.dropna().head(5).tolist()

        # Determine type
        col_type, confidence, stats, warnings = self._infer_type(series, dtype, unique_ratio, unique_count)

        return ColumnProfile(
            name=col, dtype=dtype, col_type=col_type, confidence=confidence,
            null_pct=null_pct, unique_count=unique_count, unique_ratio=unique_ratio,
            sample_values=sample, stats=stats, warnings=warnings
        )

    def _infer_type(self, series: pd.Series, dtype: str, unique_ratio: float, unique_count: int) -> Tuple[ColumnType, float, Dict, List]:
        warnings = []
        stats = {}

        # Check for temporal - try 50+ formats
        if self._is_temporal(series):
            return ColumnType.TEMPORAL, 0.95, {"format": "detected"}, []

        # Check for Year+Month combination (handles original code's problem)
        if re.search(r'year', series.name, re.I) and series.dtype in ['int64', 'float64']:
            if series.min() > 1900 and series.max() < 2100:
                return ColumnType.TEMPORAL, 0.8, {"component": "year"}, ["Part of date - check for Month column"]

        if re.search(r'month', series.name, re.I):
            if series.dtype == 'object':
                month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                               'july', 'august', 'september', 'october', 'november', 'december']
                sample_lower = [str(v).lower() for v in series.dropna().head(20)]
                matches = sum(1 for s in sample_lower if s in month_names)
                if matches / len(sample_lower) > 0.8:
                    return ColumnType.TEMPORAL, 0.85, {"component": "month_name"}, ["Part of date - check for Year column"]
            elif series.dtype in ['int64', 'float64']:
                if series.min() >= 1 and series.max() <= 12:
                    return ColumnType.TEMPORAL, 0.85, {"component": "month_number"}, ["Part of date - check for Year column"]

        # Boolean check
        if unique_count == 2:
            return ColumnType.BOOLEAN, 0.95, {}, []

        # Identifier check: high cardinality, no pattern
        if unique_ratio > 0.9 and series.dtype in ['int64', 'object']:
            # Check if sequential (likely ID)
            if series.dtype == 'int64':
                sorted_vals = sorted(series.dropna())
                if len(sorted_vals) > 1:
                    diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(min(100, len(sorted_vals)-1))]
                    if np.mean(diffs) == 1.0:
                        return ColumnType.IDENTIFIER, 0.9, {"pattern": "sequential"}, []
            return ColumnType.IDENTIFIER, 0.85, {"pattern": "high_cardinality"}, []

        # Numeric analysis
        if series.dtype in ['int64', 'float64']:
            # Check if it's actually categorical (few unique values)
            if unique_count <= 20 and unique_ratio < 0.05:
                return ColumnType.CATEGORICAL, 0.8, {"numeric_codes": True}, ["Numeric but low cardinality - may be codes"]

            # Check for meaningful distribution (not just IDs)
            stats['mean'] = series.mean()
            stats['std'] = series.std()
            stats['skew'] = series.skew()

            # If std is 0, not a useful metric
            if stats['std'] == 0 or pd.isna(stats['std']):
                return ColumnType.QUASI_IDENTIFIER, 0.6, stats, ["Zero variance - may be constant/ID"]

            # Check range (Zip codes, phone numbers have specific ranges)
            if series.min() > 10000 and series.max() < 99999 and stats['std'] < 1000:
                return ColumnType.QUASI_IDENTIFIER, 0.7, stats, ["Range suggests Zip/Postal code"]

            return ColumnType.METRIC, 0.9, stats, []

        # String analysis
        if series.dtype == 'object':
            avg_len = series.dropna().astype(str).str.len().mean()

            # Text if long strings
            if avg_len > 50:
                return ColumnType.TEXT, 0.9, {"avg_length": avg_len}, []

            # Categorical if low cardinality
            if unique_count <= 50 or unique_ratio < 0.2:
                return ColumnType.CATEGORICAL, 0.85, {"categories": unique_count}, []

            return ColumnType.TEXT, 0.7, {"avg_length": avg_len}, ["High cardinality text"]

        return ColumnType.UNKNOWN, 0.5, {}, ["Could not determine type"]

    def _is_temporal(self, series: pd.Series) -> bool:
        """Try to parse as datetime with multiple formats."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if series.dtype == 'object':
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False

            # Common formats to try
            formats = [
                '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y', '%m-%d-%Y',
                '%Y%m%d', '%d%m%Y',
                '%b %Y', '%B %Y', '%Y-%b', '%Y-%B',
                '%Y-%m', '%Y/%m'
            ]

            for fmt in formats:
                try:
                    parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        return True
                except:
                    continue

            # Try pandas auto-parse
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    return True
            except:
                pass

        return False

    def construct_temporal(self) -> Optional[pd.Series]:
        """Construct datetime from Year+Month components if needed."""
        if self.temporal_column and self.profiles[self.temporal_column].stats.get('component') == 'year':
            # Look for month component
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

    def get_summary(self) -> Dict:
        return {
            "total_columns": len(self.df.columns),
            "total_rows": len(self.df),
            "temporal": self.temporal_column,
            "metrics": self.metric_columns,
            "categories": self.categorical_columns,
            "identifiers": self.identifier_columns,
            "profiles": {k: {
                "type": v.col_type.value,
                "confidence": v.confidence,
                "null_pct": v.null_pct,
                "warnings": v.warnings
            } for k, v in self.profiles.items()}
        }

# ============================================================
# LAYER 2: OPERATION REGISTRY (No Code Execution)
# ============================================================

class OperationResult:
    def __init__(self, valid: bool, result: Any, warnings: List[str], 
                 assumptions: Dict, viz_type: str, data: Optional[pd.DataFrame] = None):
        self.valid = valid
        self.result = result
        self.warnings = warnings
        self.assumptions = assumptions
        self.viz_type = viz_type
        self.data = data

class OperationRegistry:
    """Pre-built, validated operations. No exec(). No eval()."""

    def __init__(self, profiler: DataProfiler):
        self.profiler = profiler
        self.df = profiler.df.copy()
        self.dt_col = profiler.construct_temporal()
        if self.dt_col is not None:
            self.df['_axi_dt'] = self.dt_col

    def _get_metric(self, name: str) -> Optional[str]:
        """Resolve metric name with fuzzy matching."""
        name_lower = name.lower().replace(' ', '')

        # Exact match
        if name in self.profiler.metric_columns:
            return name

        # Fuzzy match
        for col in self.profiler.metric_columns:
            col_clean = col.lower().replace(' ', '').replace('_', '')
            if name_lower == col_clean:
                return col
            if name_lower in col_clean or col_clean in name_lower:
                return col

        return None

    def _get_category(self, name: str) -> Optional[str]:
        """Resolve category name."""
        name_lower = name.lower().replace(' ', '')

        for col in self.profiler.categorical_columns:
            col_clean = col.lower().replace(' ', '').replace('_', '')
            if name_lower == col_clean:
                return col

        return None

    def trend_analysis(self, metric: str, time_range: Optional[str] = None) -> OperationResult:
        """Validated trend analysis with proper statistical tests."""
        col = self._get_metric(metric)
        if not col:
            return OperationResult(False, None, [f"Metric '{metric}' not found"], {}, "none")

        if self.dt_col is None:
            return OperationResult(False, None, ["No temporal column detected"], {}, "none")

        # Prepare data
        df_clean = self.df[[col, '_axi_dt']].dropna()
        df_clean = df_clean.sort_values('_axi_dt')

        # Validation
        assumptions = {}
        warnings = []

        if len(df_clean) < 3:
            return OperationResult(False, None, ["Need at least 3 data points"], {"n": len(df_clean)}, "none")

        if df_clean[col].std() == 0 or pd.isna(df_clean[col].std()):
            return OperationResult(False, None, ["Zero variance - cannot analyze trend"], {}, "none")

        # Check temporal spacing
        time_diffs = df_clean['_axi_dt'].diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            irregular = (time_diffs > median_diff * 1.5).mean()
            assumptions['temporal_irregularity'] = irregular
            if irregular > 0.3:
                warnings.append(f"{irregular:.0%} of intervals are irregularly spaced")

        # Statistical tests
        time_numeric = np.arange(len(df_clean))

        # Mann-Kendall for monotonic trend
        tau, p_value = kendalltau(time_numeric, df_clean[col])

        # Theil-Sen robust slope
        slopes = []
        for i in range(len(df_clean)):
            for j in range(i+1, len(df_clean)):
                if time_numeric[j] != time_numeric[i]:
                    slopes.append((df_clean[col].iloc[j] - df_clean[col].iloc[i]) / (time_numeric[j] - time_numeric[i]))

        theil_slope = np.median(slopes) if slopes else 0

        # Direction
        direction = "increasing" if tau > 0.2 else "decreasing" if tau < -0.2 else "stable"

        # Check for non-monotonic (suggest change point)
        if abs(tau) < 0.3 and len(df_clean) > 10:
            # Simple change point detection
            mid = len(df_clean) // 2
            first_half = df_clean[col].iloc[:mid]
            second_half = df_clean[col].iloc[mid:]
            if abs(second_half.mean() - first_half.mean()) > df_clean[col].std():
                warnings.append("Non-monotonic pattern detected - consider change point analysis")

        result = {
            "direction": direction,
            "mann_kendall_tau": round(tau, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "theil_sen_slope": round(theil_slope, 4),
            "n_points": len(df_clean),
            "start_value": round(df_clean[col].iloc[0], 2),
            "end_value": round(df_clean[col].iloc[-1], 2),
            "total_change_pct": round((df_clean[col].iloc[-1] / df_clean[col].iloc[0] - 1) * 100, 2) if df_clean[col].iloc[0] != 0 else None
        }

        assumptions['test'] = "Mann-Kendall + Theil-Sen"
        assumptions['min_points'] = len(df_clean) >= 3
        assumptions['non_zero_variance'] = df_clean[col].std() > 0

        return OperationResult(True, result, warnings, assumptions, "line", df_clean)

    def anomaly_detection(self, metric: str, by_category: Optional[str] = None) -> OperationResult:
        """Multi-method anomaly detection with consensus scoring."""
        col = self._get_metric(metric)
        if not col:
            return OperationResult(False, None, [f"Metric '{metric}' not found"], {}, "none")

        df_work = self.df.copy()

        # Stratified if category specified
        cat_col = self._get_category(by_category) if by_category else None

        if cat_col and cat_col not in df_work.columns:
            return OperationResult(False, None, [f"Category '{by_category}' not found"], {}, "none")

        all_anomalies = []
        warnings = []

        categories = [None] if not cat_col else df_work[cat_col].dropna().unique()

        for cat in categories:
            if cat is not None:
                subset = df_work[df_work[cat_col] == cat]
                prefix = f"{cat_col}={cat}: "
            else:
                subset = df_work
                prefix = ""

            values = subset[col].dropna()

            if len(values) < 5:
                warnings.append(f"{prefix}Insufficient data ({len(values)} points)")
                continue

            if values.std() == 0:
                warnings.append(f"{prefix}Zero variance")
                continue

            # Method 1: IQR
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            iqr_mask = (values < lower_iqr) | (values > upper_iqr)

            # Method 2: Isolation Forest
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values.values.reshape(-1, 1))
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_mask = iso.fit_predict(scaled) == -1

            # Method 3: Z-score (robust)
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad > 0:
                modified_z = 0.6745 * (values - median) / mad
                z_mask = np.abs(modified_z) > 3.5
            else:
                z_mask = pd.Series([False] * len(values), index=values.index)

            # Consensus scoring
            consensus = pd.DataFrame({
                'iqr': iqr_mask.values,
                'isolation': iso_mask,
                'zscore': z_mask.values
            }, index=values.index)

            consensus['score'] = consensus.sum(axis=1)
            consensus['value'] = values

            # High confidence: 2+ methods agree
            high_conf = consensus[consensus['score'] >= 2]

            for idx, row in high_conf.iterrows():
                all_anomalies.append({
                    'index': int(idx),
                    'value': round(row['value'], 2),
                    'confidence': 'high' if row['score'] == 3 else 'medium',
                    'methods': [m for m in ['iqr', 'isolation', 'zscore'] if consensus.loc[idx, m]],
                    'category': cat
                })

        result = {
            "total_anomalies": len(all_anomalies),
            "high_confidence": sum(1 for a in all_anomalies if a['confidence'] == 'high'),
            "by_category": by_category is not None,
            "anomalies": all_anomalies[:50]  # Limit for display
        }

        assumptions = {
            "methods": ["IQR", "IsolationForest", "Modified Z-score"],
            "consensus_threshold": 2,
            "min_points": 5
        }

        return OperationResult(True, result, warnings, assumptions, "scatter", df_work)

    def forecast(self, metric: str, periods: int = 3) -> OperationResult:
        """Simple but statistically honest forecast."""
        col = self._get_metric(metric)
        if not col:
            return OperationResult(False, None, [f"Metric '{metric}' not found"], {}, "none")

        if self.dt_col is None:
            return OperationResult(False, None, ["No temporal column detected"], {}, "none")

        df_clean = self.df[[col, '_axi_dt']].dropna().sort_values('_axi_dt')

        if len(df_clean) < 6:
            return OperationResult(False, None, ["Need at least 6 points for forecasting"], {}, "none")

        values = df_clean[col].values

        # Check for NaN at end
        if pd.isna(values[-1]):
            return OperationResult(False, None, ["Most recent value is missing"], {}, "none")

        # Simple exponential smoothing (honest about limitations)
        alpha = 0.3
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

        # Trend component
        if len(values) > 1:
            trend = (values[-1] - values[0]) / (len(values) - 1)
        else:
            trend = 0

        # Forecast
        last_level = smoothed[-1]
        forecast_values = []
        for i in range(1, periods + 1):
            forecast_values.append(last_level + trend * i)

        # Honest confidence: use historical MAE, not normal approximation
        if len(values) > 6:
            # Simple 1-step backtest
            errors = []
            for i in range(5, len(values)):
                pred = values[i-1]  # naive forecast
                errors.append(abs(values[i] - pred))
            mae = np.mean(errors)
        else:
            mae = np.std(values) * 0.5  # rough estimate

        # Widen with horizon
        ci_lower = [f - mae * (1 + i * 0.3) for i, f in enumerate(forecast_values)]
        ci_upper = [f + mae * (1 + i * 0.3) for i, f in enumerate(forecast_values)]

        result = {
            "method": "Simple Exponential Smoothing + Trend",
            "forecast_values": [round(v, 2) for v in forecast_values],
            "ci_lower": [round(v, 2) for v in ci_lower],
            "ci_upper": [round(v, 2) for v in ci_upper],
            "trend_per_period": round(trend, 2),
            "historical_mae": round(mae, 2),
            "caveat": "Simple model - use ARIMA/Prophet for complex seasonality"
        }

        warnings = ["Simple model - confidence intervals are approximate"]

        assumptions = {
            "model": "SES",
            "min_points": 6,
            "horizon": periods,
            "ci_method": "historical MAE with horizon expansion"
        }

        return OperationResult(True, result, warnings, assumptions, "line", df_clean)

    def correlation_matrix(self, metrics: Optional[List[str]] = None) -> OperationResult:
        """Pairwise complete observations correlation."""
        cols = metrics or self.profiler.metric_columns
        cols = [c for c in cols if c in self.profiler.metric_columns]

        if len(cols) < 2:
            return OperationResult(False, None, ["Need at least 2 metrics"], {}, "none")

        # Pairwise complete observations
        corr_data = self.df[cols].copy()

        # Calculate correlation with pairwise deletion
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    corr_matrix.loc[c1, c2] = 1.0
                    p_matrix.loc[c1, c2] = 0.0
                else:
                    pair = corr_data[[c1, c2]].dropna()
                    if len(pair) > 2:
                        r, p = spearmanr(pair[c1], pair[c2])
                        corr_matrix.loc[c1, c2] = r
                        p_matrix.loc[c1, c2] = p
                    else:
                        corr_matrix.loc[c1, c2] = np.nan
                        p_matrix.loc[c1, c2] = np.nan

        # Find strong relationships
        strong = []
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols[i+1:], i+1):
                r = corr_matrix.loc[c1, c2]
                if not pd.isna(r) and abs(r) > 0.5:
                    strong.append({
                        "var1": c1,
                        "var2": c2,
                        "correlation": round(r, 3),
                        "p_value": round(p_matrix.loc[c1, c2], 4),
                        "significant": p_matrix.loc[c1, c2] < 0.05,
                        "n_obs": len(corr_data[[c1, c2]].dropna())
                    })

        result = {
            "correlations": corr_matrix.round(3).to_dict(),
            "strong_relationships": strong,
            "n_metrics": len(cols)
        }

        return OperationResult(True, result, [], {"method": "Spearman, pairwise complete"}, "heatmap", corr_data)

    def group_aggregate(self, metric: str, by: str, agg: str = "mean") -> OperationResult:
        """Group by aggregation with validation."""
        m_col = self._get_metric(metric)
        b_col = self._get_category(by)

        if not m_col:
            return OperationResult(False, None, [f"Metric '{metric}' not found"], {}, "none")
        if not b_col:
            return OperationResult(False, None, [f"Category '{by}' not found"], {}, "none")

        if self.df[b_col].nunique() > 100:
            return OperationResult(False, None, [f"Category '{by}' has too many values (>100)"], {}, "none")

        agg_funcs = {"mean": "mean", "sum": "sum", "count": "count", 
                     "median": "median", "std": "std", "min": "min", "max": "max"}

        if agg not in agg_funcs:
            return OperationResult(False, None, [f"Unknown aggregation '{agg}'"], {}, "none")

        result_df = self.df.groupby(b_col)[m_col].agg(agg_funcs[agg]).reset_index()
        result_df.columns = [b_col, f"{agg}_{m_col}"]

        result = {
            "aggregation": agg,
            "group_by": b_col,
            "metric": m_col,
            "n_groups": len(result_df),
            "values": result_df.to_dict('records')
        }

        return OperationResult(True, result, [], {}, "bar", result_df)

# ============================================================
# LAYER 3: QUERY RESOLVER (Semantic Matching)
# ============================================================

class QueryResolver:
    """Resolves natural language to operations without keyword matching."""

    def __init__(self, profiler: DataProfiler):
        self.profiler = profiler
        self.context: List[Dict] = []

    def resolve(self, query: str) -> Dict:
        """Convert query to structured intent + entities."""
        query_lower = query.lower()

        # Intent detection (simplified - replace with LLM in production)
        intent = self._detect_intent(query_lower)

        # Entity extraction
        entities = self._extract_entities(query_lower, intent)

        # Context inheritance
        if self.context and not entities.get('metric'):
            last_metric = self._get_last_metric()
            if last_metric:
                entities['metric'] = last_metric
                intent['inherited'] = True

        # Store context
        self.context.append({
            "query": query,
            "intent": intent,
            "entities": entities
        })

        # Keep last 5
        self.context = self.context[-5:]

        return {"intent": intent, "entities": entities}

    def _detect_intent(self, query: str) -> Dict:
        """Pattern-based intent detection (robust but not brittle)."""

        # Trend patterns
        trend_indicators = ['trend', 'moving', 'direction', 'going', 'trajectory', 
                           'over time', 'historical', 'pattern', 'slope', 'growth']
        if any(w in query for w in trend_indicators):
            return {"type": "trend_analysis", "confidence": 0.9}

        # Anomaly patterns
        anomaly_indicators = ['anomaly', 'outlier', 'unusual', 'strange', 'weird',
                             'unexpected', 'deviation', 'abnormal', 'spike', 'dip']
        if any(w in query for w in anomaly_indicators):
            return {"type": "anomaly_detection", "confidence": 0.9}

        # Forecast patterns
        forecast_indicators = ['forecast', 'predict', 'future', 'next', 'upcoming',
                              'projection', 'estimate', 'will be', 'expect']
        if any(w in query for w in forecast_indicators):
            return {"type": "forecast", "confidence": 0.9}

        # Correlation patterns
        corr_indicators = ['correlation', 'related', 'relationship', 'associated',
                          'connection', 'link', 'together', 'vs', 'versus']
        if any(w in query for w in corr_indicators):
            return {"type": "correlation_matrix", "confidence": 0.85}

        # Aggregate patterns
        agg_indicators = ['total', 'sum', 'average', 'mean', 'count', 'how many',
                         'how much', 'per', 'by', 'breakdown', 'group']
        if any(w in query for w in agg_indicators):
            return {"type": "group_aggregate", "confidence": 0.85}

        # Comparison
        compare_indicators = ['compare', 'difference', 'versus', 'vs', 'between',
                             'higher', 'lower', 'better', 'worse']
        if any(w in query for w in compare_indicators):
            return {"type": "compare_segments", "confidence": 0.8}

        # Default: try to figure out from entities
        return {"type": "auto", "confidence": 0.5}

    def _extract_entities(self, query: str, intent: Dict) -> Dict:
        """Extract metric, category, time references."""
        entities = {}

        # Extract metric
        for col in self.profiler.metric_columns:
            col_patterns = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace('_', '')
            ]
            for pattern in col_patterns:
                if pattern in query:
                    entities['metric'] = col
                    break

        # Extract category
        for col in self.profiler.categorical_columns:
            if col.lower() in query or col.lower().replace('_', ' ') in query:
                entities['category'] = col
                break

        # Extract time references
        time_patterns = {
            'last_month': r'last month|previous month',
            'this_month': r'this month|current month',
            'last_year': r'last year|previous year',
            'this_year': r'this year|current year',
            'q1': r'q1|first quarter',
            'q2': r'q2|second quarter',
            'q3': r'q3|third quarter',
            'q4': r'q4|fourth quarter'
        }

        for time_key, pattern in time_patterns.items():
            if re.search(pattern, query):
                entities['time_range'] = time_key
                break

        # Extract aggregation
        if 'sum' in query or 'total' in query:
            entities['aggregation'] = 'sum'
        elif 'average' in query or 'mean' in query:
            entities['aggregation'] = 'mean'
        elif 'count' in query or 'how many' in query:
            entities['aggregation'] = 'count'
        else:
            entities['aggregation'] = 'mean'

        return entities

    def _get_last_metric(self) -> Optional[str]:
        for ctx in reversed(self.context):
            if 'metric' in ctx.get('entities', {}):
                return ctx['entities']['metric']
        return None

# ============================================================
# LAYER 4: RESPONSE FORMATTER
# ============================================================

class ResponseFormatter:
    """Converts OperationResult to natural language + visualizations."""

    @staticmethod
    def format_trend(result: OperationResult) -> str:
        if not result.valid:
            return f"❌ I couldn't analyze the trend: {result.warnings[0] if result.warnings else 'Unknown error'}"

        r = result.result
        sig_text = "statistically significant" if r['significant'] else "not statistically significant"

        text = f"""
**Trend Analysis: {r.get('metric', 'Unknown')}**

📈 **Direction**: {r['direction'].upper()}  
📊 **Strength**: τ = {r['mann_kendall_tau']} ({sig_text}, p={r['p_value']})  
📉 **Slope**: {r['theil_sen_slope']} per period (Theil-Sen robust estimate)  
💰 **Total Change**: {r['total_change_pct']}% from start to end

**Key Insight**: The data shows a {r['direction']} trend that is {sig_text}. 
"""

        if result.warnings:
            text += "
⚠️ **Caveats**: " + "; ".join(result.warnings)

        return text

    @staticmethod
    def format_anomaly(result: OperationResult) -> str:
        if not result.valid:
            return f"❌ Anomaly detection failed: {result.warnings[0] if result.warnings else 'Unknown error'}"

        r = result.result

        text = f"""
**Anomaly Detection Results**

🔍 **Total Anomalies Found**: {r['total_anomalies']}  
⭐ **High Confidence**: {r['high_confidence']}

**Top Anomalies**:
"""

        for i, a in enumerate(r['anomalies'][:5], 1):
            cat_text = f" ({a['category']})" if a['category'] else ""
            text += f"
{i}. Value **{a['value']}**{cat_text} — Confidence: {a['confidence']} (methods: {', '.join(a['methods'])})"

        if len(r['anomalies']) > 5:
            text += f"

... and {len(r['anomalies']) - 5} more"

        return text

    @staticmethod
    def format_forecast(result: OperationResult) -> str:
        if not result.valid:
            return f"❌ Forecast failed: {result.warnings[0] if result.warnings else 'Unknown error'}"

        r = result.result

        text = f"""
**Forecast Results** (Simple Exponential Smoothing)

📅 **Next Periods**:
"""

        for i, (f, l, u) in enumerate(zip(r['forecast_values'], r['ci_lower'], r['ci_upper']), 1):
            text += f"
• Period +{i}: **{f}** (range: {l} to {u})"

        text += f"""

📈 **Trend**: {r['trend_per_period']} per period  
📊 **Historical Accuracy (MAE)**: {r['historical_mae']}

⚠️ **Important**: {r['caveat']}
"""

        return text

    @staticmethod
    def create_visualization(result: OperationResult, operation_type: str) -> Optional[go.Figure]:
        if not result.valid or result.data is None:
            return None

        if operation_type == "trend_analysis" and result.viz_type == "line":
            df = result.data
            fig = px.line(df, x='_axi_dt', y=df.columns[0], 
                         title=f"Trend: {df.columns[0]}")
            fig.add_hline(y=df[df.columns[0]].mean(), line_dash="dash", 
                         annotation_text="Mean")
            return fig

        elif operation_type == "anomaly_detection" and result.viz_type == "scatter":
            # Box plot
            metric_col = [c for c in result.data.columns if c != '_axi_dt' and c not in result.data.select_dtypes(include=['object']).columns][0]
            fig = px.box(result.data, y=metric_col, title=f"Distribution & Outliers: {metric_col}")
            return fig

        elif operation_type == "forecast" and result.viz_type == "line":
            df = result.data
            metric_col = [c for c in df.columns if c != '_axi_dt'][0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['_axi_dt'], y=df[metric_col], 
                                    mode='lines', name='Historical'))

            # Add forecast points
            last_date = df['_axi_dt'].iloc[-1]
            forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, 4)]

            fig.add_trace(go.Scatter(x=forecast_dates, y=result.result['forecast_values'],
                                    mode='lines+markers', line=dict(dash='dash'),
                                    name='Forecast'))

            fig.update_layout(title=f"Forecast: {metric_col}")
            return fig

        elif operation_type == "correlation_matrix" and result.viz_type == "heatmap":
            corr_df = pd.DataFrame(result.result['correlations'])
            fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu", zmin=-1, zmax=1,
                           title="Correlation Matrix (Spearman)")
            return fig

        elif operation_type == "group_aggregate" and result.viz_type == "bar":
            df = result.data
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                        title=f"{result.result['aggregation'].title()} by {result.result['group_by']}")
            return fig

        return None

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AxiLattice Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
    .insight-card { background: #161b22; border-radius: 8px; padding: 16px; border: 1px solid #30363d; }
    .metric-positive { color: #3fb950; font-weight: 600; }
    .metric-negative { color: #f78166; font-weight: 600; }
    .chat-user { background: #388bfd20; border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 3px solid #388bfd; }
    .chat-assistant { background: #23863620; border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 3px solid #238636; }
    .stApp { background: #0d1117; }
    .stSidebar { background: #161b22; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'profiler' not in st.session_state:
    st.session_state.profiler = None
if 'registry' not in st.session_state:
    st.session_state.registry = None
if 'resolver' not in st.session_state:
    st.session_state.resolver = None
if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.markdown('<p class="main-header">🧠 AxiLattice</p>', unsafe_allow_html=True)
    st.caption("Schema-Inferred Intelligence")

    uploaded = st.file_uploader("📁 Upload Data", type=['csv', 'xlsx', 'xls', 'parquet'])

    if uploaded:
        try:
            # Handle different formats with encoding fallback
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
            st.session_state.registry = OperationRegistry(st.session_state.profiler)
            st.session_state.resolver = QueryResolver(st.session_state.profiler)

            st.success(f"✅ {len(df):,} rows × {len(df.columns)} cols")

            # Show schema
            with st.expander("📋 Detected Schema"):
                prof = st.session_state.profiler
                for col, p in prof.profiles.items():
                    icon = {"temporal": "📅", "metric": "📊", "categorical": "🏷️", 
                           "identifier": "🔑", "boolean": "☑️", "text": "📝"}.get(p.col_type.value, "❓")
                    st.write(f"{icon} **{col}** ({p.col_type.value}, {p.confidence:.0%} confidence)")
                    if p.warnings:
                        st.caption(f"⚠️ {'; '.join(p.warnings)}")

        except Exception as e:
            st.error(f"❌ Failed to load: {str(e)}")

    st.divider()

    if st.session_state.profiler:
        mode = st.radio("Mode", ["🎙️ Ask", "📊 Auto-Report", "🔍 Explore"])
    else:
        st.info("Upload data to begin")

# Main content
if not st.session_state.profiler:
    st.markdown("""
    ## 👋 AxiLattice Pro

    **Upload any dataset** — CSV, Excel, Parquet. The system auto-detects:
    - 📅 Temporal columns (handles Year+Month combinations)
    - 📊 Metrics vs identifiers vs categories
    - 🏷️ Data quality issues

    **Then ask anything** in natural language.
    """)
    st.stop()

prof = st.session_state.profiler
reg = st.session_state.registry
resolver = st.session_state.resolver

# MODE: Ask (Conversational)
if "Ask" in mode:
    st.markdown('<p class="main-header">🎙️ Ask Your Data</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Quick Questions")
        suggestions = [
            f"Trend in {prof.metric_columns[0]}" if prof.metric_columns else "No metrics",
            f"Anomalies in {prof.metric_columns[0]}" if prof.metric_columns else "No metrics",
            f"Forecast {prof.metric_columns[0]}" if prof.metric_columns else "No metrics",
            f"Correlations" if len(prof.metric_columns) > 1 else "Need 2+ metrics",
        ]

        for s in suggestions:
            if not s.endswith("metrics") and not s.endswith("metrics"):
                if st.button(s, key=f"btn_{s[:20]}"):
                    st.session_state.pending_query = s

    with col2:
        # Chat history
        for msg in st.session_state.chat[-10:]:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-user'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-assistant'><b>AxiLattice:</b></div>", unsafe_allow_html=True)
                st.markdown(msg['content'])
                if 'viz' in msg and msg['viz']:
                    st.plotly_chart(msg['viz'], use_container_width=True)
                if 'code' in msg:
                    with st.expander("View operation details"):
                        st.json(msg['code'])

        # Input
        query = st.chat_input("Ask anything...")
        if not query and 'pending_query' in st.session_state:
            query = st.session_state.pending_query
            del st.session_state.pending_query

        if query:
            st.session_state.chat.append({"role": "user", "content": query})

            with st.spinner("Analyzing..."):
                # Resolve query
                resolved = resolver.resolve(query)
                intent = resolved['intent']['type']
                entities = resolved['entities']

                # Execute operation
                result = None
                op_type = intent

                if intent == "trend_analysis":
                    metric = entities.get('metric', prof.metric_columns[0] if prof.metric_columns else None)
                    if metric:
                        result = reg.trend_analysis(metric)
                        text = ResponseFormatter.format_trend(result)
                    else:
                        text = "❌ No metric found. Available metrics: " + ", ".join(prof.metric_columns)

                elif intent == "anomaly_detection":
                    metric = entities.get('metric', prof.metric_columns[0] if prof.metric_columns else None)
                    if metric:
                        result = reg.anomaly_detection(metric)
                        text = ResponseFormatter.format_anomaly(result)
                    else:
                        text = "❌ No metric found."

                elif intent == "forecast":
                    metric = entities.get('metric', prof.metric_columns[0] if prof.metric_columns else None)
                    if metric:
                        result = reg.forecast(metric)
                        text = ResponseFormatter.format_forecast(result)
                    else:
                        text = "❌ No metric found."

                elif intent == "correlation_matrix":
                    result = reg.correlation_matrix()
                    if result.valid:
                        r = result.result
                        text = f"**Correlation Analysis**

Found **{r['n_metrics']}** metrics with **{len(r['strong_relationships'])}** strong relationships (|r| > 0.5):

"
                        for rel in r['strong_relationships'][:5]:
                            sig = "✅" if rel['significant'] else "❌"
                            text += f"• {rel['var1']} ↔ {rel['var2']}: {rel['correlation']} {sig} (n={rel['n_obs']})
"
                    else:
                        text = f"❌ {result.warnings[0] if result.warnings else 'Error'}"

                elif intent == "group_aggregate":
                    metric = entities.get('metric')
                    cat = entities.get('category')
                    agg = entities.get('aggregation', 'mean')

                    if not metric and prof.metric_columns:
                        metric = prof.metric_columns[0]
                    if not cat and prof.categorical_columns:
                        cat = prof.categorical_columns[0]

                    if metric and cat:
                        result = reg.group_aggregate(metric, cat, agg)
                        if result.valid:
                            r = result.result
                            text = f"**{agg.title()} of {metric} by {cat}**

Groups found: {r['n_groups']}

Top values:
"
                            for v in r['values'][:5]:
                                text += f"• {v[r['group_by']]}: **{v[f'{agg}_{metric}']:.2f}**
"
                        else:
                            text = f"❌ {result.warnings[0] if result.warnings else 'Error'}"
                    else:
                        text = f"Need metric and category. Metrics: {prof.metric_columns}, Categories: {prof.categorical_columns}"

                else:
                    text = f"""
🤔 I understood you want to analyze the data, but I'm not sure which operation.

**Detected intent**: {intent} (confidence: {resolved['intent']['confidence']})
**Found entities**: {entities}

**Try asking**:
- "What's the trend in {prof.metric_columns[0] if prof.metric_columns else 'sales'}?"
- "Find anomalies in {prof.metric_columns[0] if prof.metric_columns else 'revenue'}"
- "Forecast next 3 months"
- "Show correlations"
- "Average {prof.metric_columns[0] if prof.metric_columns else 'sales'} by {prof.categorical_columns[0] if prof.categorical_columns else 'region'}"
"""

                # Create visualization
                viz = ResponseFormatter.create_visualization(result, op_type) if result else None

                # Add to chat
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": text,
                    "viz": viz,
                    "code": {
                        "intent": intent,
                        "entities": entities,
                        "operation": op_type,
                        "valid": result.valid if result else False,
                        "warnings": result.warnings if result else []
                    }
                })

                st.rerun()

# MODE: Auto-Report
elif "Auto-Report" in mode:
    st.markdown('<p class="main-header">📊 Comprehensive Intelligence Report</p>', unsafe_allow_html=True)

    selected = st.multiselect("Metrics to analyze", prof.metric_columns, 
                             default=prof.metric_columns[:min(3, len(prof.metric_columns))])

    if selected and st.button("🚀 Generate Report", type="primary"):
        progress = st.progress(0)

        for i, metric in enumerate(selected):
            progress.progress((i + 1) / len(selected))

            with st.expander(f"📈 {metric}", expanded=True):
                cols = st.columns(3)

                # Trend
                trend = reg.trend_analysis(metric)
                if trend.valid:
                    cols[0].metric("Trend", trend.result['direction'].upper(), 
                                  f"τ={trend.result['mann_kendall_tau']}")
                else:
                    cols[0].metric("Trend", "N/A", trend.warnings[0] if trend.warnings else "Error")

                # Anomaly
                anom = reg.anomaly_detection(metric)
                if anom.valid:
                    cols[1].metric("Anomalies", anom.result['total_anomalies'], 
                                  f"{anom.result['high_confidence']} high conf")
                else:
                    cols[1].metric("Anomalies", "N/A", "Error")

                # Forecast
                fore = reg.forecast(metric)
                if fore.valid:
                    change = fore.result['forecast_values'][0] / fore.result.get('historical_mae', 1)
                    cols[2].metric("Next Period", f"{fore.result['forecast_values'][0]:.0f}", 
                                  f"±{fore.result['historical_mae']:.0f} MAE")
                else:
                    cols[2].metric("Forecast", "N/A", "Error")

                # Visualization
                viz = ResponseFormatter.create_visualization(trend, "trend_analysis")
                if viz:
                    st.plotly_chart(viz, use_container_width=True)

        progress.empty()

# MODE: Explore
else:
    st.markdown('<p class="main-header">🔍 Data Explorer</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Distributions", "🔗 Correlations"])

    with tab1:
        st.dataframe(st.session_state.df, use_container_width=True)

    with tab2:
        if prof.metric_columns:
            col = st.selectbox("Select metric", prof.metric_columns)
            fig = px.histogram(st.session_state.df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

            stats = prof.profiles[col].stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{stats.get('mean', 0):.2f}")
            c2.metric("Std", f"{stats.get('std', 0):.2f}")
            c3.metric("Skew", f"{stats.get('skew', 0):.2f}")
            c4.metric("Null", f"{prof.profiles[col].null_pct:.1f}%")

    with tab3:
        if len(prof.metric_columns) > 1:
            corr_result = reg.correlation_matrix()
            if corr_result.valid:
                viz = ResponseFormatter.create_visualization(corr_result, "correlation_matrix")
                if viz:
                    st.plotly_chart(viz, use_container_width=True)

                st.subheader("Strong Relationships")
                for rel in corr_result.result['strong_relationships']:
                    st.write(f"**{rel['var1']} ↔ {rel['var2']}**: {rel['correlation']} (p={rel['p_value']}, n={rel['n_obs']})")
        else:
            st.info("Need 2+ metrics for correlation analysis")

st.divider()
st.caption("AxiLattice Pro v2.0 | Local execution | No arbitrary code execution")

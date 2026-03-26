
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, math
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from scipy.interpolate import RegularGridInterpolator
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── Constants ────────────────────────────────────────────────
COLORS = ['#00d4ff','#ff6b35','#7fff00','#ff1493','#ffd700','#da70d6','#87ceeb','#98fb98']
FBG = '#0d1117'; ABG = '#161b22'; TC = '#e6edf3'; MC = '#8b949e'
LAGS = [1, 3, 7, 14, 21]
WINDOWS = [7, 14, 30]
FORECAST_HORIZON = 14
LEAD_TIME = 7
plt.style.use('dark_background')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title='SmartWH AI System',
    page_icon='🏭',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SUBSYSTEM FUNCTIONS
# ══════════════════════════════════════════════════════════════

def clean_data(df):
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    if df['Description'].isna().any():
        mode_desc = df.groupby('StockCode')['Description'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN')
        df['Description'] = df.apply(
            lambda r: mode_desc.get(r['StockCode'], 'UNKNOWN')
            if pd.isna(r['Description']) else r['Description'], axis=1)
    df['HasCustomerID'] = ~df['CustomerID'].isna()
    df['CustomerID']    = df['CustomerID'].fillna('ANON')
    return df

def engineer_features(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
    df['Date']      = df['InvoiceDate'].dt.date
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Month']     = df['InvoiceDate'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    return df

def build_daily_demand(df):
    daily = df.groupby(['Date','StockCode']).agg(
        {'Quantity':'sum','UnitPrice':'mean','TotalRevenue':'sum','Description':'first'}
    ).reset_index()
    daily.columns = ['Date','StockCode','DailyQty','AvgPrice','DailyRevenue','Description']
    daily['Date'] = pd.to_datetime(daily['Date'])
    return daily.sort_values('Date').reset_index(drop=True)

def build_inventory_summary(df):
    inv = df.groupby('StockCode').agg({
        'Quantity':'sum','UnitPrice':'mean','TotalRevenue':'sum',
        'Description':'first',
        'Country': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    inv.columns = ['StockCode','TotalQty','AvgUnitPrice','TotalRevenue','Description','TopCountry']
    return inv

def create_lag_features(ts):
    df = ts.copy()
    for lag in LAGS:
        df[f'lag_{lag}'] = df['DailyQty'].shift(lag)
    for win in WINDOWS:
        df[f'ma_{win}']  = df['DailyQty'].rolling(win, min_periods=1).mean()
        df[f'std_{win}'] = df['DailyQty'].rolling(win, min_periods=1).std().fillna(0)
    df['dow']   = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    return df.bfill()

def build_training_data(daily_ts, top_n=50):
    top_skus = daily_ts.groupby('StockCode')['DailyQty'].sum().nlargest(top_n).index.tolist()
    subset = daily_ts[daily_ts['StockCode'].isin(top_skus)].copy()
    parts  = []
    for sku in top_skus:
        sku_ts = subset[subset['StockCode'] == sku].sort_values('Date').reset_index(drop=True)
        parts.append(create_lag_features(sku_ts))
    return pd.concat(parts, ignore_index=True)

def chronological_split(df, train=0.70, val=0.15):
    n = len(df)
    return df[:int(n*train)], df[int(n*train):int(n*(train+val))], df[int(n*(train+val)):]

def forecast_sku(model, feature_cols, daily_df, sku, horizon=FORECAST_HORIZON):
    s = daily_df[daily_df['StockCode'] == sku].set_index('Date')['DailyQty'].sort_index()
    if len(s) == 0:
        return pd.DataFrame({'date':[], 'forecast':[]})
    history_values = list(s.values)
    last_date = s.index[-1]
    preds, dates = [], []
    for step in range(horizon):
        fut_date = last_date + pd.Timedelta(days=step+1)
        row = {}
        for lag in LAGS:
            idx = len(history_values) - lag
            row[f'lag_{lag}'] = history_values[idx] if idx >= 0 else 0
        for win in WINDOWS:
            recent = history_values[-win:]
            row[f'ma_{win}']  = float(np.mean(recent)) if recent else 0.0
            row[f'std_{win}'] = float(np.std(recent))  if len(recent) > 1 else 0.0
        row['dow']   = fut_date.dayofweek
        row['month'] = fut_date.month
        X_row = np.array([[row[c] for c in feature_cols]])
        pred  = float(np.maximum(0, model.predict(X_row))[0])
        preds.append(pred); dates.append(fut_date)
        history_values.append(pred)
    return pd.DataFrame({'date': dates, 'forecast': preds})

def build_fuzzy_system():
    cov_u = np.arange(0, 5.05, 0.05)
    vol_u = np.arange(0, 3.05, 0.05)
    pri_u = np.arange(0, 101, 1)
    cov = ctrl.Antecedent(cov_u, 'coverage')
    vol = ctrl.Antecedent(vol_u, 'volatility')
    pri = ctrl.Consequent(pri_u, 'priority')
    cov['critical'] = fuzz.trapmf(cov.universe, [0,0,0.5,1.0])
    cov['low']      = fuzz.trimf(cov.universe,  [0.5,1.5,2.5])
    cov['adequate'] = fuzz.trimf(cov.universe,  [2.0,3.0,4.0])
    cov['high']     = fuzz.trapmf(cov.universe, [3.5,4.5,5,5])
    vol['stable']   = fuzz.trapmf(vol.universe, [0,0,0.3,0.6])
    vol['variable'] = fuzz.trimf(vol.universe,  [0.4,1.0,1.8])
    vol['volatile'] = fuzz.trapmf(vol.universe, [1.5,2.5,3,3])
    pri['none']     = fuzz.trapmf(pri.universe, [0,0,10,20])
    pri['low']      = fuzz.trimf(pri.universe,  [15,30,45])
    pri['medium']   = fuzz.trimf(pri.universe,  [35,50,65])
    pri['high']     = fuzz.trimf(pri.universe,  [55,70,85])
    pri['critical'] = fuzz.trapmf(pri.universe, [75,90,100,100])
    rules = [
        ctrl.Rule(cov['critical'],                       pri['critical']),
        ctrl.Rule(cov['low']      & vol['volatile'],     pri['critical']),
        ctrl.Rule(cov['low']      & vol['variable'],     pri['high']),
        ctrl.Rule(cov['low']      & vol['stable'],       pri['medium']),
        ctrl.Rule(cov['adequate'] & vol['volatile'],     pri['high']),
        ctrl.Rule(cov['adequate'] & vol['variable'],     pri['medium']),
        ctrl.Rule(cov['adequate'] & vol['stable'],       pri['low']),
        ctrl.Rule(cov['high']     & vol['volatile'],     pri['low']),
        ctrl.Rule(cov['high']     & vol['variable'],     pri['none']),
        ctrl.Rule(cov['high']     & vol['stable'],       pri['none']),
    ]
    system = ctrl.ControlSystem(rules)
    cov_pts = np.arange(0, 5.1, 0.1)
    vol_pts = np.arange(0, 3.1, 0.1)
    grid    = np.zeros((len(cov_pts), len(vol_pts)))
    sim     = ctrl.ControlSystemSimulation(system)
    for i, c in enumerate(cov_pts):
        for j, v in enumerate(vol_pts):
            try:
                sim.input['coverage']   = float(c)
                sim.input['volatility'] = float(v)
                sim.compute()
                grid[i,j] = sim.output['priority']
            except:
                grid[i,j] = 50.0
    interp = RegularGridInterpolator((cov_pts, vol_pts), grid,
                                     method='linear', bounds_error=False, fill_value=None)
    return interp

# ══════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    df_raw   = pd.read_excel(file_bytes, dtype={'InvoiceNo':str,'StockCode':str})
    df_clean = engineer_features(clean_data(df_raw))
    daily    = build_daily_demand(df_clean)
    inv      = build_inventory_summary(df_clean)
    df_seg   = df_clean[df_clean['HasCustomerID']].copy()
    return df_raw, df_clean, daily, inv, df_seg

@st.cache_resource(show_spinner=False)
def train_rf(_daily):
    df_train = build_training_data(_daily)
    feature_cols = (
        [f'lag_{l}' for l in LAGS] +
        [f'ma_{w}'  for w in WINDOWS] +
        [f'std_{w}' for w in WINDOWS] +
        ['dow', 'month']
    )
    train_set, val_set, test_set = chronological_split(df_train.dropna())
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train_set[feature_cols], train_set['DailyQty'])
    y_pred = model.predict(test_set[feature_cols])
    metrics = {
        'mae':  mean_absolute_error(test_set['DailyQty'], y_pred),
        'rmse': np.sqrt(mean_squared_error(test_set['DailyQty'], y_pred)),
        'r2':   r2_score(test_set['DailyQty'], y_pred),
        'train': len(train_set), 'val': len(val_set), 'test': len(test_set),
    }
    return model, feature_cols, metrics

@st.cache_resource(show_spinner=False)
def compute_rfm(_df_seg):
    ref  = _df_seg['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm  = _df_seg.groupby('CustomerID').agg({
        'InvoiceDate':  lambda x: (ref - x.max()).days,
        'InvoiceNo':    'count',
        'TotalRevenue': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID','Recency','Frequency','Monetary']
    rfm['Monetary']      = rfm['Monetary'].round(2)
    rfm['log_Frequency'] = np.log1p(rfm['Frequency'])
    rfm['log_Monetary']  = np.log1p(rfm['Monetary'])
    scaled = RobustScaler().fit_transform(rfm[['Recency','log_Frequency','log_Monetary']])
    rfm['Segment'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)
    stats = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
    names = {}
    for seg in stats.index:
        r, f, m = stats.loc[seg,'Recency'], stats.loc[seg,'Frequency'], stats.loc[seg,'Monetary']
        if m > rfm['Monetary'].quantile(0.75) and f > rfm['Frequency'].quantile(0.75):
            names[seg] = 'VIP'
        elif r <= rfm['Recency'].quantile(0.33) and f >= rfm['Frequency'].quantile(0.50):
            names[seg] = 'Active'
        elif r >= rfm['Recency'].quantile(0.66):
            names[seg] = 'Churn Risk'
        else:
            names[seg] = 'Potential'
    rfm['SegmentName'] = rfm['Segment'].map(names)
    return rfm

@st.cache_resource(show_spinner=False)
def compute_anomalies(_df_clean):
    a = _df_clean.copy()
    a['QtyZScore_SKU']   = a.groupby('StockCode')['Quantity'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6))
    a['PriceZScore_SKU'] = a.groupby('StockCode')['UnitPrice'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6))
    feats = ['Quantity','UnitPrice','TotalRevenue','QtyZScore_SKU','PriceZScore_SKU']
    X     = a[feats].fillna(0)
    iso   = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(X)
    a['AnomalyScore']  = -iso.decision_function(X)
    a['IsAnomaly_IF']  = (iso.predict(X) == -1).astype(int)
    flags = []
    for col in ['Quantity','UnitPrice','TotalRevenue']:
        Q1, Q3 = a[col].quantile(0.25), a[col].quantile(0.75)
        IQR = Q3 - Q1
        flags.append(((a[col] < Q1-1.5*IQR) | (a[col] > Q3+1.5*IQR)).astype(int))
    a['IsAnomaly_IQR'] = (sum(flags) > 0).astype(int)
    return a[['InvoiceNo','StockCode','Quantity','UnitPrice','TotalRevenue',
               'QtyZScore_SKU','PriceZScore_SKU','AnomalyScore',
               'IsAnomaly_IF','IsAnomaly_IQR','InvoiceDate']]

@st.cache_resource(show_spinner=False)
def compute_alerts(_daily, _inv):
    interp_fn = build_fuzzy_system()
    sku_dem   = _daily.groupby('StockCode')['DailyQty'].agg(Avg='mean', Std='std').fillna(0).reset_index()
    p = _inv.merge(sku_dem, on='StockCode', how='left').fillna(0)
    p['EstCurrentStock'] = p['TotalQty'] * 0.1
    p['Coverage']  = np.where(p['Avg'] > 0, (p['EstCurrentStock'] / (p['Avg'] * LEAD_TIME)).clip(0,5), 5.0)
    p['DemandCV']  = np.where(p['Avg'] > 0.01, (p['Std'] / p['Avg']).clip(0,3), 0.0)
    p['SafetyStock']= (1.645 * p['Std'] * math.sqrt(LEAD_TIME)).round(1)
    p['ROP']       = (p['Avg'] * LEAD_TIME + p['SafetyStock']).round(1)
    ann = p['Avg'] * 365
    hpu = p['AvgUnitPrice'] * 0.25
    p['EOQ'] = np.where(hpu > 0, np.sqrt(2 * ann * 50 / hpu), 0).round(0)
    pts = np.column_stack([p['Coverage'].values.clip(0,5), p['DemandCV'].values.clip(0,3)])
    p['FuzzyPriority'] = interp_fn(pts)
    def lvl(s):
        if s >= 75: return 'CRITICAL'
        if s >= 50: return 'WARNING'
        if s >= 25: return 'MONITOR'
        return 'OK'
    p['AlertLevel'] = p['FuzzyPriority'].apply(lvl)
    return p

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🏭 SmartWH AI System")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📁 Upload Online Retail.xlsx", type=['xlsx'])

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "📈 Forecast Explorer",
    "🚨 Alert Filter",
    "👥 Segment Explorer",
    "🔍 Anomaly Detection",
    "ℹ️ System Info",
])

# ── No file uploaded ─────────────────────────────────────────
if uploaded_file is None:
    st.title("🏭 Smart Warehouse AI System")
    st.info("👈 Upload **Online Retail.xlsx** in the sidebar to begin.")
    st.markdown("""
    ### System Overview
    | Subsystem | Description |
    |-----------|-------------|
    | 📊 Subsystem 1 | Data Ingestion & Validation |
    | 📈 Subsystem 2 | AI Demand Forecasting (Random Forest) |
    | 🔍 Subsystem 3 | Anomaly Detection (Isolation Forest + IQR) |
    | 👥 Subsystem 4 | Customer Segmentation (RFM + K-Means) |
    | 🚨 Subsystem 5 | Fuzzy Logic Reorder System |
    | 📊 Subsystem 6 | Analytics Dashboard |
    """)
    st.stop()

# ── Load and train ────────────────────────────────────────────
with st.spinner("⚙️ Loading data..."):
    df_raw, df_clean, daily, inv, df_seg = load_data(uploaded_file)

with st.spinner("🌲 Training Random Forest..."):
    rf_model, feature_cols, rf_metrics = train_rf(daily)

with st.spinner("👥 Segmenting customers..."):
    rfm = compute_rfm(df_seg)

with st.spinner("🔍 Detecting anomalies..."):
    anom_df = compute_anomalies(df_clean)

with st.spinner("🚨 Computing fuzzy alerts..."):
    alerts = compute_alerts(daily, inv)

st.sidebar.success(f"✅ {len(df_clean):,} records loaded")

# ══════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total SKUs",         f"{inv['StockCode'].nunique():,}")
    c2.metric("Total Customers",    f"{df_seg['CustomerID'].nunique():,}")
    c3.metric("Clean Records",      f"{len(df_clean):,}")
    c4.metric("Critical Alerts",    f"{len(alerts[alerts['AlertLevel']=='CRITICAL']):,}")
    c5.metric("Anomalies (IF)",     f"{int(anom_df['IsAnomaly_IF'].sum()):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Sales Volume")
        fig, ax = plt.subplots(figsize=(8,3), facecolor=FBG)
        ax.set_facecolor(ABG)
        dt = daily.groupby('Date')['DailyQty'].sum()
        ax.plot(dt.index, dt.values, color=COLORS[0], linewidth=0.8)
        ax.fill_between(dt.index, dt.values, alpha=0.2, color=COLORS[0])
        ax.set_xlabel('Date', color=MC); ax.set_ylabel('Units Sold', color=MC)
        ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Customer Segments")
        fig, ax = plt.subplots(figsize=(8,3), facecolor=FBG)
        sc = rfm['SegmentName'].value_counts()
        ax.pie(sc.values, labels=sc.index, autopct='%1.1f%%',
               colors=COLORS[:len(sc)], textprops={'color':TC})
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Alert Distribution")
        fig, ax = plt.subplots(figsize=(8,3), facecolor=FBG)
        ax.set_facecolor(ABG)
        ad = alerts['AlertLevel'].value_counts().reindex(['CRITICAL','WARNING','MONITOR','OK'], fill_value=0)
        ac = {'CRITICAL':'#ff4444','WARNING':'#ffaa00','MONITOR':'#ffd700','OK':'#44ff44'}
        bars = ax.bar(ad.index, ad.values, color=[ac[k] for k in ad.index], alpha=0.85)
        for bar, v in zip(bars, ad.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(v), ha='center', color=TC)
        ax.set_ylabel('SKUs', color=MC); ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Top 10 Products by Revenue")
        fig, ax = plt.subplots(figsize=(8,3), facecolor=FBG)
        ax.set_facecolor(ABG)
        t10 = inv.nlargest(10,'TotalRevenue')
        ax.barh(range(len(t10)), t10['TotalRevenue'],
                color=[COLORS[i%len(COLORS)] for i in range(len(t10))], alpha=0.85)
        ax.set_yticks(range(len(t10)))
        ax.set_yticklabels([s[:25] for s in t10['Description']], color=MC, fontsize=7)
        ax.set_xlabel('Total Revenue (£)', color=MC); ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE: Forecast Explorer
# ══════════════════════════════════════════════════════════════
elif page == "📈 Forecast Explorer":
    st.title("📈 Demand Forecast Explorer")

    sku_list = (daily.groupby('StockCode')
                     .filter(lambda x: len(x) > 5)['StockCode']
                     .unique().tolist())

    col1, col2 = st.columns([3,1])
    with col1:
        selected_sku = st.selectbox("Select SKU", sku_list)
    with col2:
        horizon = st.slider("Horizon (days)", 1, 30, 14)

    if st.button("🔮 Run Forecast", use_container_width=True):
        with st.spinner(f"Forecasting SKU {selected_sku}..."):
            sku_hist    = daily[daily['StockCode'] == selected_sku].sort_values('Date').tail(60)
            forecast_df = forecast_sku(rf_model, feature_cols, daily, selected_sku, horizon)

            fig, ax = plt.subplots(figsize=(12,5), facecolor=FBG)
            ax.set_facecolor(ABG)
            hx = range(len(sku_hist))
            fx = range(len(sku_hist), len(sku_hist)+len(forecast_df))
            ax.plot(hx, sku_hist['DailyQty'].values, 'o-', color=COLORS[0], linewidth=2, label='Historical')
            ax.plot(fx, forecast_df['forecast'].values, 's--', color=COLORS[1], linewidth=2, label='Forecast')
            ax.fill_between(fx, forecast_df['forecast'].values, alpha=0.2, color=COLORS[1])
            ax.axvline(x=len(sku_hist), color=MC, linestyle=':', linewidth=1)
            ax.set_xlabel('Days', color=MC); ax.set_ylabel('Daily Quantity', color=MC)
            ax.set_title(f'SKU {selected_sku} — Last {len(sku_hist)} Days + {horizon}-Day Forecast',
                         color=TC, fontweight='bold')
            ax.legend(facecolor=ABG, edgecolor=MC); ax.tick_params(colors=MC)
            for s in ax.spines.values(): s.set_color(MC)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Forecast Table")
        st.dataframe(
            forecast_df.rename(columns={'date':'Date','forecast':'Forecast Qty'})
                       .style.format({'Forecast Qty':'{:.1f}'}),
            use_container_width=True)

    st.markdown("---")
    st.subheader("Model Performance")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Test MAE",   f"{rf_metrics['mae']:.2f}")
    c2.metric("Test RMSE",  f"{rf_metrics['rmse']:.2f}")
    c3.metric("Test R²",    f"{rf_metrics['r2']:.3f}")
    c4.metric("Train Size", f"{rf_metrics['train']:,}")
    c5.metric("Test Size",  f"{rf_metrics['test']:,}")

# ══════════════════════════════════════════════════════════════
# PAGE: Alert Filter
# ══════════════════════════════════════════════════════════════
elif page == "🚨 Alert Filter":
    st.title("🚨 Fuzzy Logic Reorder Alerts")

    col1, col2 = st.columns([3,1])
    with col1:
        level = st.radio("Alert Level", ['CRITICAL','WARNING','MONITOR','OK'], horizontal=True)
    with col2:
        top_n = st.slider("Top N", 5, 100, 20)

    filtered = alerts[alerts['AlertLevel'] == level].nlargest(top_n, 'FuzzyPriority')
    icons    = {'CRITICAL':'🔴','WARNING':'🟠','MONITOR':'🟡','OK':'🟢'}
    st.markdown(f"### {icons.get(level,'')} {level} — {len(filtered)} SKUs")

    st.dataframe(
        filtered[['StockCode','Description','Coverage','DemandCV',
                  'SafetyStock','ROP','EOQ','FuzzyPriority','AlertLevel']]
                .reset_index(drop=True)
                .style.format({'Coverage':'{:.2f}','DemandCV':'{:.2f}',
                               'FuzzyPriority':'{:.1f}','ROP':'{:.1f}','EOQ':'{:.0f}'}),
        use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Alerts CSV", csv,
                       f"alerts_{level}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════
# PAGE: Segment Explorer
# ══════════════════════════════════════════════════════════════
elif page == "👥 Segment Explorer":
    st.title("👥 Customer Segment Explorer")

    seg_list     = sorted(rfm['SegmentName'].dropna().unique().tolist())
    selected_seg = st.selectbox("Select Segment", seg_list)
    seg_data     = rfm[rfm['SegmentName'] == selected_seg]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers",     f"{len(seg_data):,}")
    c2.metric("Avg Recency",   f"{seg_data['Recency'].mean():.1f} days")
    c3.metric("Avg Frequency", f"{seg_data['Frequency'].mean():.1f}")
    c4.metric("Avg Monetary",  f"£{seg_data['Monetary'].mean():.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recency vs Monetary")
        fig, ax = plt.subplots(figsize=(7,4), facecolor=FBG)
        ax.set_facecolor(ABG)
        for sn in rfm['SegmentName'].dropna().unique():
            sd = rfm[rfm['SegmentName'] == sn]
            ax.scatter(sd['Recency'], np.log1p(sd['Monetary']),
                       alpha=0.8 if sn==selected_seg else 0.15,
                       s=60 if sn==selected_seg else 15, label=sn)
        ax.set_xlabel('Recency (days)', color=MC)
        ax.set_ylabel('log(Monetary)', color=MC)
        ax.legend(facecolor=ABG, edgecolor=MC); ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Segment Distribution")
        fig, ax = plt.subplots(figsize=(7,4), facecolor=FBG)
        sc  = rfm['SegmentName'].value_counts()
        exp = [0.08 if s==selected_seg else 0 for s in sc.index]
        ax.pie(sc.values, labels=sc.index, autopct='%1.1f%%',
               colors=COLORS[:len(sc)], explode=exp, textprops={'color':TC})
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader(f"Top Customers — {selected_seg}")
    st.dataframe(
        seg_data.nlargest(20,'Monetary')[['CustomerID','Recency','Frequency','Monetary']]
                .reset_index(drop=True)
                .style.format({'Monetary':'£{:.2f}','Recency':'{:.0f}','Frequency':'{:.0f}'}),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: Anomaly Detection
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(anom_df):,}")
    c2.metric("IQR Flagged",        f"{int(anom_df['IsAnomaly_IQR'].sum()):,}")
    c3.metric("IF Flagged",         f"{int(anom_df['IsAnomaly_IF'].sum()):,}")
    c4.metric("Both Agree",         f"{int((anom_df['IsAnomaly_IQR'] & anom_df['IsAnomaly_IF']).sum()):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(7,4), facecolor=FBG)
        ax.set_facecolor(ABG)
        ax.hist(anom_df[anom_df['IsAnomaly_IF']==0]['AnomalyScore'],
                bins=60, alpha=0.7, color=COLORS[0], label='Normal')
        ax.hist(anom_df[anom_df['IsAnomaly_IF']==1]['AnomalyScore'],
                bins=60, alpha=0.7, color='#ff4444', label='Anomaly')
        ax.set_xlabel('Anomaly Score', color=MC)
        ax.legend(facecolor=ABG, labelcolor=TC); ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Quantity vs Revenue")
        fig, ax = plt.subplots(figsize=(7,4), facecolor=FBG)
        ax.set_facecolor(ABG)
        n = anom_df[anom_df['IsAnomaly_IF']==0]
        a = anom_df[anom_df['IsAnomaly_IF']==1]
        ax.scatter(n['Quantity'], n['TotalRevenue'], alpha=0.2, s=8,  color=COLORS[0], label='Normal')
        ax.scatter(a['Quantity'], a['TotalRevenue'], alpha=0.7, s=30, color='#ff4444', marker='x', label='Anomaly')
        ax.set_xlabel('Quantity', color=MC); ax.set_ylabel('Total Revenue', color=MC)
        ax.legend(facecolor=ABG, labelcolor=TC); ax.tick_params(colors=MC)
        for s in ax.spines.values(): s.set_color(MC)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Top Anomalies")
    top_anom = anom_df[anom_df['IsAnomaly_IF']==1].nlargest(20,'AnomalyScore')
    st.dataframe(
        top_anom[['InvoiceNo','StockCode','Quantity','UnitPrice','TotalRevenue','AnomalyScore']]
                .reset_index(drop=True)
                .style.format({'AnomalyScore':'{:.3f}','TotalRevenue':'£{:.2f}'}),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: System Info
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ System Info":
    st.title("ℹ️ System Information")

    st.subheader("📦 Subsystem 1 — Data Ingestion")
    c1,c2,c3 = st.columns(3)
    c1.metric("Raw Records",   f"{len(df_raw):,}")
    c2.metric("Clean Records", f"{len(df_clean):,}")
    c3.metric("Removed",       f"{len(df_raw)-len(df_clean):,}")
    c1.metric("Unique SKUs",   f"{inv['StockCode'].nunique():,}")
    c2.metric("Countries",     f"{df_clean['Country'].nunique()}")
    c3.metric("Date Range",    f"{df_clean['InvoiceDate'].min().date()} → {df_clean['InvoiceDate'].max().date()}")

    st.subheader("🌲 Subsystem 2 — Demand Forecasting")
    c1,c2,c3 = st.columns(3)
    c1.metric("Test MAE",  f"{rf_metrics['mae']:.2f} units")
    c2.metric("Test RMSE", f"{rf_metrics['rmse']:.2f} units")
    c3.metric("Test R²",   f"{rf_metrics['r2']:.3f}")

    st.subheader("🔍 Subsystem 3 — Anomaly Detection")
    c1,c2,c3 = st.columns(3)
    c1.metric("Transactions",  f"{len(anom_df):,}")
    c2.metric("IQR Flagged",   f"{int(anom_df['IsAnomaly_IQR'].sum()):,}")
    c3.metric("IF Flagged",    f"{int(anom_df['IsAnomaly_IF'].sum()):,}")

    st.subheader("👥 Subsystem 4 — Customer Segmentation")
    for seg in rfm['SegmentName'].dropna().unique():
        cnt = len(rfm[rfm['SegmentName']==seg])
        st.write(f"**{seg}**: {cnt:,} customers ({100*cnt/len(rfm):.1f}%)")

    st.subheader("🚨 Subsystem 5 — Fuzzy Reorder System")
    for lvl in ['CRITICAL','WARNING','MONITOR','OK']:
        cnt = len(alerts[alerts['AlertLevel']==lvl])
        st.write(f"**{lvl}**: {cnt:,} SKUs ({100*cnt/len(alerts):.1f}%)")

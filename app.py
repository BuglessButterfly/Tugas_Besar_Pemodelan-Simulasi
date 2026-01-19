import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# SciPy untuk fitting (disarankan)
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# Config + Metadata
# ============================================================
st.set_page_config(page_title="SIR + Euler/RK4 (COVID-19 India)", layout="wide")

DATASET_NAME = "Kaggle – covid19-in-india (Sudalai Raj Kumar)"
DATASET_URL = "https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india"
DATASET_FILE = "covid19.csv"

st.title("Simulasi Penyebaran COVID-19 India — Model SIR (ODE) + Euler & RK4")
st.caption(
    "(Jalur 1 Kontinu): Load Data → EDA → Model ODE (SIR) → Implementasi Euler & RK4 → "
    "Estimasi Parameter → Validasi (Grafik & Metrik) → Deployment Streamlit (What-if)."
)

if not HAS_SCIPY:
    st.info(
        "Catatan: SciPy tidak tersedia. Mode fitting (Fitted) tidak bisa dipakai, "
        "tetapi mode What-if tetap bisa jalan."
    )


# ============================================================
# Utils: Data
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    paths = [DATASET_FILE, f"/mnt/data/{DATASET_FILE}"]
    data_path = next((p for p in paths if os.path.exists(p)), None)
    if data_path is None:
        return None, None
    df = pd.read_csv(data_path)
    return df, data_path


def _robust_parse_dates(df: pd.DataFrame) -> pd.Series:
    s1 = pd.to_datetime(df["Date"], errors="coerce")
    if float(s1.isna().mean()) > 0.2:
        s2 = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        if float(s2.isna().mean()) < float(s1.isna().mean()):
            return s2
    return s1


def prepare_timeseries(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    - Confirmed/Cured/Deaths adalah kumulatif.
    - Anti double-count: MAX per (Date, State), lalu SUM antar state (untuk All India).
    - Pastikan kumulatif tidak pernah turun (cummax).
    - Index harian + ffill agar konsisten untuk ODE harian.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    scope = str(scope).strip()

    if "Date" not in df.columns:
        raise ValueError("Kolom 'Date' tidak ditemukan di dataset.")

    if "State/UnionTerritory" in df.columns:
        df["State/UnionTerritory"] = df["State/UnionTerritory"].astype(str).str.strip()

    df["Date"] = _robust_parse_dates(df)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in ["Confirmed", "Cured", "Deaths"]:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan.")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "State/UnionTerritory" in df.columns:
        by_state = (
            df.groupby(["Date", "State/UnionTerritory"])[["Confirmed", "Cured", "Deaths"]]
              .max()
              .sort_index()
        )
        if scope == "All India":
            ts = by_state.groupby("Date")[["Confirmed", "Cured", "Deaths"]].sum().sort_index()
        else:
            states = df["State/UnionTerritory"].dropna().unique().tolist()
            if scope not in states:
                raise ValueError(f"Scope '{scope}' tidak ditemukan.")
            ts = by_state.xs(scope, level="State/UnionTerritory").sort_index()
    else:
        if scope != "All India":
            raise ValueError("Dataset tidak punya 'State/UnionTerritory' → hanya bisa All India.")
        ts = df.groupby("Date")[["Confirmed", "Cured", "Deaths"]].max().sort_index()

    for c in ["Confirmed", "Cured", "Deaths"]:
        ts[c] = ts[c].cummax()

    ts = ts.sort_index().asfreq("D")
    ts[["Confirmed", "Cured", "Deaths"]] = ts[["Confirmed", "Cured", "Deaths"]].ffill()

    ts["Active"] = (ts["Confirmed"] - ts["Cured"] - ts["Deaths"]).clip(lower=0)  # I_obs
    ts["Removed"] = (ts["Cured"] + ts["Deaths"]).clip(lower=0)                   # R_obs
    ts = ts.dropna()
    return ts


# ============================================================
# Utils: SIR (ODE) + Euler & RK4
# ============================================================
def sir_deriv(t, y, beta, gamma, N):
    S, I, R = y
    if N <= 0:
        raise ValueError("N harus > 0.")
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta dan gamma harus > 0.")

    S = max(float(S), 0.0)
    I = max(float(I), 0.0)
    R = max(float(R), 0.0)

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR], dtype=float)


def rk4_step(f, t, y, h, *params):
    y = np.asarray(y, dtype=float)
    k1 = f(t, y, *params)
    k2 = f(t + h/2, y + (h/2)*k1, *params)
    k3 = f(t + h/2, y + (h/2)*k2, *params)
    k4 = f(t + h,   y + h*k3,     *params)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_sir_rk4(beta, gamma, N, S0, I0, R0, days, h=1.0, conserve=True):
    if days < 1:
        raise ValueError("days minimal 1.")
    if h <= 0:
        raise ValueError("h harus > 0.")

    t = np.arange(0, days + 1, h, dtype=float)
    y = np.zeros((len(t), 3), dtype=float)
    y[0] = [float(S0), float(I0), float(R0)]

    for i in range(1, len(t)):
        y_next = rk4_step(sir_deriv, t[i-1], y[i-1], h, beta, gamma, N)
        y_next = np.maximum(y_next, 0.0)
        if conserve:
            tot = float(y_next.sum())
            if tot > 0:
                y_next = y_next * (float(N) / tot)
        y[i] = y_next

    return t, y


def euler_step(f, t, y, h, *params):
    y = np.asarray(y, dtype=float)
    return y + h * f(t, y, *params)


def simulate_sir_euler(beta, gamma, N, S0, I0, R0, days, h=1.0, conserve=True):
    if days < 1:
        raise ValueError("days minimal 1.")
    if h <= 0:
        raise ValueError("h harus > 0.")

    t = np.arange(0, days + 1, h, dtype=float)
    y = np.zeros((len(t), 3), dtype=float)
    y[0] = [float(S0), float(I0), float(R0)]

    for i in range(1, len(t)):
        y_next = euler_step(sir_deriv, t[i-1], y[i-1], h, beta, gamma, N)
        y_next = np.maximum(y_next, 0.0)
        if conserve:
            tot = float(y_next.sum())
            if tot > 0:
                y_next = y_next * (float(N) / tot)
        y[i] = y_next

    return t, y


# ============================================================
# Metrics
# ============================================================
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape_masked(y_true, y_pred, min_true=10.0, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= float(min_true)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0)


# ============================================================
# Plotly
# ============================================================
def plot_lines(dates, series_dict, title, y_label, height=380):
    fig = go.Figure()
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(x=dates, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


# ============================================================
# Fitting (Mode Fitted): beta, gamma, N_fit (N_eff)
# ============================================================
@st.cache_data(show_spinner=False)
def fit_beta_gamma_nfit(I_obs, R_obs, I0, R0_init, N_worldbank):
    """
    FIXED FITTING (konsisten dengan notebook final):
    - loss pakai log1p
    - gamma dibatasi realistis (durasi 5–21 hari) => gamma in [1/21, 1/5]
    - multi-start untuk stabilitas
    - fitting pakai conserve=False
    """
    if not HAS_SCIPY:
        return None

    I_obs = np.asarray(I_obs, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)

    IR_max = float(np.max(I_obs + R_obs))
    N_fit_min = float(1.05 * IR_max)
    N_fit_max = float(min(N_worldbank, 200.0 * IR_max))

    # guardrail: bounds harus valid
    if (not np.isfinite(N_fit_min)) or (not np.isfinite(N_fit_max)) or (N_fit_max <= N_fit_min):
        return None

    N_fit0 = float(max(N_fit_min * 1.5, 10.0 * IR_max))

    gamma_min = 1.0 / 21.0
    gamma_max = 1.0 / 5.0

    def loss(params):
        beta, gamma, N_fit = params

        # batas beta
        if beta <= 0 or beta > 2.0:
            return 1e18

        # batas gamma realistis
        if gamma < gamma_min or gamma > gamma_max:
            return 1e18

        # batas N_fit
        if not (N_fit_min <= N_fit <= N_fit_max):
            return 1e18

        S0_fit = N_fit - I0 - R0_init
        if S0_fit <= 0:
            return 1e18

        _, y = simulate_sir_rk4(
            beta, gamma, N_fit, S0_fit, I0, R0_init,
            days=len(I_obs) - 1, h=1.0, conserve=False  # fitting: conserve=False
        )
        I_pred = np.maximum(y[:, 1], 0.0)
        R_pred = np.maximum(y[:, 2], 0.0)
        I_true = np.maximum(I_obs, 0.0)
        R_true = np.maximum(R_obs, 0.0)

        # log-loss
        mse_I = np.mean((np.log1p(I_pred) - np.log1p(I_true)) ** 2)
        mse_R = np.mean((np.log1p(R_pred) - np.log1p(R_true)) ** 2)
        return float(mse_I + mse_R)

    # multi-start (lebih stabil)
    starts = [
        (0.2, 1.0/14.0, max(N_fit_min * 1.5, N_fit0)),
        (0.4, 1.0/10.0, max(N_fit_min * 2.0, N_fit0)),
        (0.8, 1.0/7.0,  max(N_fit_min * 3.0, N_fit0)),
        (1.2, 1.0/5.5,  max(N_fit_min * 4.0, N_fit0)),
    ]

    best_res = None
    best_fun = float("inf")

    for b0, g0, n0 in starts:
        # pastikan start gamma masuk bounds
        g0 = float(np.clip(g0, gamma_min, gamma_max))
        n0 = float(np.clip(n0, N_fit_min, N_fit_max))

        res = minimize(
            loss,
            x0=np.array([b0, g0, n0], dtype=float),
            bounds=[(1e-6, 2.0), (gamma_min, gamma_max), (N_fit_min, N_fit_max)],
            method="L-BFGS-B",
        )

        if getattr(res, "success", False) and np.isfinite(res.fun) and float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_res = res

    if best_res is None:
        return None

    beta_hat, gamma_hat, N_fit_hat = map(float, best_res.x)

    # guardrail final
    if beta_hat <= 0 or gamma_hat <= 0 or N_fit_hat <= (I0 + R0_init):
        return None

    return beta_hat, gamma_hat, N_fit_hat


# ============================================================
# Load dataset
# ============================================================
df_raw, data_path = load_data()
if df_raw is None:
    st.error(f"File `{DATASET_FILE}` tidak ditemukan. Taruh di folder app.py atau `/mnt/data/`.")
    st.stop()

df_raw.columns = [c.strip() for c in df_raw.columns]
if "State/UnionTerritory" in df_raw.columns:
    df_raw["State/UnionTerritory"] = df_raw["State/UnionTerritory"].astype(str).str.strip()

required_cols = {"Date", "Confirmed", "Cured", "Deaths"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()


# ============================================================
# Sidebar (Real-time)
# ============================================================
st.sidebar.header("Pengaturan")

# Scope
scopes = ["All India"]
if "State/UnionTerritory" in df_raw.columns:
    scopes += sorted(df_raw["State/UnionTerritory"].dropna().unique().tolist())
SCOPE = st.sidebar.selectbox("Cakupan", scopes, index=0)

# N metadata (pelaporan)
st.sidebar.markdown("---")
st.sidebar.subheader("Populasi Total (N) — untuk pelaporan (metadata)")
N_YEAR = st.sidebar.number_input("Tahun referensi N", 1900, 2100, 2020, 1)
N_SOURCE_NAME = st.sidebar.text_input("Sumber resmi (nama)", "World Bank – Population, total (SP.POP.TOTL)")
N_SOURCE_URL = st.sidebar.text_input("Sumber resmi (URL)", "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN")
DEFAULT_N_INDIA_2020 = 1402617695
N_worldbank = int(st.sidebar.number_input("Nilai N (bilangan bulat)", min_value=1, value=int(DEFAULT_N_INDIA_2020), step=1))

if not N_SOURCE_NAME.strip() or not N_SOURCE_URL.strip():
    st.sidebar.warning("Lengkapi nama & URL sumber N agar sesuai instruksi (N tidak ambigu).")

# Warning bila scope state tapi N masih India
if SCOPE != "All India":
    st.sidebar.warning(
        "Anda memilih scope state, tetapi nilai N default masih populasi India. "
        "Masukkan N state yang benar agar hasil valid."
    )

# Mode parameter
st.sidebar.markdown("---")
st.sidebar.subheader("Mode Parameter")
mode = st.sidebar.radio(
    "Pilih sumber parameter",
    ["Fitted dari data (beta_hat, gamma_hat, N_fit_hat)", "What-if (slider R0 & durasi infeksi)"],
    index=0,
)
if mode.startswith("Fitted") and not HAS_SCIPY:
    st.sidebar.warning("SciPy tidak tersedia → Mode Fitted tidak bisa dijalankan. Gunakan What-if.")

# What-if slider
st.sidebar.markdown("---")
st.sidebar.subheader("What-if Slider")
R0_in = st.sidebar.slider("R0", 0.5, 5.0, 2.0, 0.1)
inf_days = st.sidebar.slider("Durasi infeksi (hari)", 3, 21, 10, 1)
gamma_slider = 1.0 / float(inf_days)
beta_slider = float(R0_in) * gamma_slider

# Forecast horizon
st.sidebar.markdown("---")
extra_days = st.sidebar.slider("Forecast ke depan (hari)", 0, 180, 0, 7)

# Tampilan
st.sidebar.markdown("---")
plot_mode = st.sidebar.selectbox(
    "Tampilan Kurva SIR",
    ["Proporsi (S/N, I/N, R/N)", "Per 100.000 penduduk", "Jumlah (raw)"],
    index=0,
)

# Toggle pembanding Euler
st.sidebar.markdown("---")
show_euler = st.sidebar.checkbox("Tampilkan pembanding Euler (baseline)", value=True)

st.sidebar.caption("Aplikasi otomatis update saat input berubah (real-time).")

# ============================================================
# Prepare data
# ============================================================
try:
    ts = prepare_timeseries(df_raw, SCOPE)
except Exception as e:
    st.error(f"Gagal menyiapkan time series: {e}")
    st.stop()

if len(ts) < 30:
    st.error("Data terlalu sedikit setelah cleaning (butuh >= 30 hari).")
    st.stop()

dates = ts.index
I_obs = ts["Active"].to_numpy(dtype=float)
R_obs = ts["Removed"].to_numpy(dtype=float)

I0 = float(I_obs[0])
R0_init = float(R_obs[0])

days_data = len(ts) - 1
total_days = int(days_data + int(extra_days))
if total_days < 1:
    st.error("Total hari simulasi < 1. Data terlalu pendek.")
    st.stop()


# ============================================================
# Choose parameters based on mode
# ============================================================
fit_used = None
if mode.startswith("Fitted"):
    fit = fit_beta_gamma_nfit(I_obs, R_obs, I0, R0_init, N_worldbank=N_worldbank)
    if fit is None:
        st.error("Fitting gagal / tidak valid. Coba mode What-if atau pastikan SciPy tersedia.")
        st.stop()
    beta_use, gamma_use, N_use = fit  # N_use = N_fit_hat (N_eff)
    fit_used = True
else:
    beta_use, gamma_use, N_use = float(beta_slider), float(gamma_slider), float(N_worldbank)
    fit_used = False

S0_use = float(N_use) - I0 - R0_init
if S0_use <= 0:
    st.error("S0 <= 0. N (atau N_fit) terlalu kecil dibanding I0 + R0.")
    st.stop()


# ============================================================
# Simulate
# - Untuk metrik validasi: conserve=False (konsisten & fair)
# - Untuk kurva SIR/forecast display: conserve=True (rapi)
# ============================================================
# Simulasi khusus untuk METRIK (hanya sampai horizon data)
_, y_rk4_m = simulate_sir_rk4(beta_use, gamma_use, N_use, S0_use, I0, R0_init, days=days_data, h=1.0, conserve=False)
I_rk4_on_data = y_rk4_m[:, 1]
R_rk4_on_data = y_rk4_m[:, 2]

I_eul_on_data = None
R_eul_on_data = None
if show_euler:
    _, y_eul_m = simulate_sir_euler(beta_use, gamma_use, N_use, S0_use, I0, R0_init, days=days_data, h=1.0, conserve=False)
    I_eul_on_data = y_eul_m[:, 1]
    R_eul_on_data = y_eul_m[:, 2]

# Simulasi untuk DISPLAY (termasuk forecast)
_, y_rk4 = simulate_sir_rk4(beta_use, gamma_use, N_use, S0_use, I0, R0_init, days=total_days, h=1.0, conserve=True)
S_rk4, I_rk4, R_rk4 = y_rk4[:, 0], y_rk4[:, 1], y_rk4[:, 2]

S_eul = I_eul = R_eul = None
if show_euler:
    _, y_eul = simulate_sir_euler(beta_use, gamma_use, N_use, S0_use, I0, R0_init, days=total_days, h=1.0, conserve=True)
    S_eul, I_eul, R_eul = y_eul[:, 0], y_eul[:, 1], y_eul[:, 2]

dates_sim = pd.date_range(start=dates[0], periods=total_days + 1, freq="D")


# ============================================================
# Transform plot for SIR curves
# ============================================================
if plot_mode == "Proporsi (S/N, I/N, R/N)":
    S_plot, I_plot, R_plot = S_rk4 / N_use, I_rk4 / N_use, R_rk4 / N_use
    if show_euler:
        S_plot_e, I_plot_e, R_plot_e = S_eul / N_use, I_eul / N_use, R_eul / N_use
    y_label_sir = "Proporsi (0–1)"
elif plot_mode == "Per 100.000 penduduk":
    scale = 100_000.0 / N_use
    S_plot, I_plot, R_plot = S_rk4 * scale, I_rk4 * scale, R_rk4 * scale
    if show_euler:
        S_plot_e, I_plot_e, R_plot_e = S_eul * scale, I_eul * scale, R_eul * scale
    y_label_sir = "Kasus per 100.000 penduduk"
else:
    S_plot, I_plot, R_plot = S_rk4, I_rk4, R_rk4
    if show_euler:
        S_plot_e, I_plot_e, R_plot_e = S_eul, I_eul, R_eul
    y_label_sir = "Jumlah (raw)"


# ============================================================
# UI Tabs
# ============================================================
tab_eda, tab_sim, tab_about = st.tabs(["EDA", "Simulasi & Validasi", "Tentang"])

with tab_eda:
    st.subheader("EDA Time Series (ringkas)")
    fig_base = plot_lines(
        dates,
        {
            "Confirmed": ts["Confirmed"].values,
            "Active (I_obs)": ts["Active"].values,
            "Removed (R_obs)": ts["Removed"].values
        },
        f"Data Asli — {SCOPE}",
        "Jumlah kasus",
        height=420,
    )
    st.plotly_chart(fig_base, use_container_width=True)

    daily_new = ts["Confirmed"].diff().fillna(0).clip(lower=0)
    roll7 = daily_new.rolling(7).mean()
    fig_daily = plot_lines(
        dates,
        {"Daily New Confirmed": daily_new.values, "Rolling Mean (7d)": roll7.values},
        "Daily New Confirmed + Rolling Mean 7 Hari",
        "Kasus per hari",
        height=420,
    )
    st.plotly_chart(fig_daily, use_container_width=True)


with tab_sim:
    colA, colB = st.columns([1.2, 1.0])

    with colA:
        st.subheader("Ringkasan")
        st.write(f"**Dataset:** `{os.path.basename(data_path)}`")
        st.write(f"**Cakupan:** {SCOPE}")
        st.write(f"**N untuk pelaporan (tahun {int(N_YEAR)}):** {int(N_worldbank):,}")
        st.write(f"**Sumber N:** {N_SOURCE_NAME}")
        st.write(f"**URL:** {N_SOURCE_URL}")
        st.write(f"**Forecast tambahan:** {int(extra_days)} hari")
        st.write(f"**Mode:** {mode}")

        st.markdown("---")
        if fit_used:
            st.success(f"Parameter hasil fit: β={beta_use:.4f}, γ={gamma_use:.4f}, N_fit={int(N_use):,}")
            st.write(f"**R0 (β/γ):** {(beta_use / gamma_use):.3f}")
            st.caption(
                "Catatan: N_fit adalah populasi efektif (N_eff) yang terlibat dalam dinamika pada periode data. "
                "N (World Bank) dicantumkan untuk pelaporan (metadata). "
                "Fitting memakai log-loss dan batas gamma realistis (durasi 5–21 hari)."
            )
        else:
            st.info(f"Parameter what-if: β={beta_use:.4f}, γ={gamma_use:.4f}, R0={beta_use/gamma_use:.3f}")

        st.markdown("---")
        st.subheader("Metrik validasi (Data vs Model)")

        rows = [
            {"Method": "RK4", "Series": "I (Active)", "MAE": mae(I_obs, I_rk4_on_data), "RMSE": rmse(I_obs, I_rk4_on_data), "MAPE_masked(%)": mape_masked(I_obs, I_rk4_on_data)},
            {"Method": "RK4", "Series": "R (Removed)", "MAE": mae(R_obs, R_rk4_on_data), "RMSE": rmse(R_obs, R_rk4_on_data), "MAPE_masked(%)": mape_masked(R_obs, R_rk4_on_data)},
        ]
        if show_euler:
            rows += [
                {"Method": "Euler", "Series": "I (Active)", "MAE": mae(I_obs, I_eul_on_data), "RMSE": rmse(I_obs, I_eul_on_data), "MAPE_masked(%)": mape_masked(I_obs, I_eul_on_data)},
                {"Method": "Euler", "Series": "R (Removed)", "MAE": mae(R_obs, R_eul_on_data), "RMSE": rmse(R_obs, R_eul_on_data), "MAPE_masked(%)": mape_masked(R_obs, R_eul_on_data)},
            ]

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)
        st.caption("MAPE_masked dihitung hanya saat nilai aktual ≥ 10 agar tidak meledak saat awal data kecil.")

    with colB:
        st.subheader("Validasi grafik")
        series_I = {"I_obs": I_obs, "I_pred (RK4)": I_rk4_on_data}
        if show_euler:
            series_I["I_pred (Euler)"] = I_eul_on_data
        figI = plot_lines(dates, series_I, "Validasi I(t) (horizon data)", "Active")
        st.plotly_chart(figI, use_container_width=True)

        series_R = {"R_obs": R_obs, "R_pred (RK4)": R_rk4_on_data}
        if show_euler:
            series_R["R_pred (Euler)"] = R_eul_on_data
        figR = plot_lines(dates, series_R, "Validasi R(t) (horizon data)", "Removed")
        st.plotly_chart(figR, use_container_width=True)

    st.markdown("---")
    st.subheader("Kurva SIR hasil simulasi (termasuk forecast)")

    series_sir = {"S (RK4)": S_plot, "I (RK4)": I_plot, "R (RK4)": R_plot}
    if show_euler:
        series_sir.update({"S (Euler)": S_plot_e, "I (Euler)": I_plot_e, "R (Euler)": R_plot_e})

    figSIR = plot_lines(
        dates_sim,
        series_sir,
        f"Kurva SIR | Mode: {plot_mode}",
        y_label_sir,
        height=480,
    )
    st.plotly_chart(figSIR, use_container_width=True)


with tab_about:
    st.subheader("Tentang aplikasi ini")
    st.write(
        "Aplikasi ini memodelkan COVID-19 dengan SIR (ODE) dan menyelesaikannya secara numerik dengan Euler & RK4. "
        "Aplikasi menyediakan mode fitting parameter dari data serta mode what-if (slider) untuk eksplorasi."
    )
    st.markdown("### Dataset")
    st.write(f"**Sumber:** {DATASET_NAME}")
    st.write(f"**Link:** {DATASET_URL}")
    st.write(f"**File:** `{DATASET_FILE}`")
    st.markdown(
        "**Observasi SIR:**\n"
        "- I_obs(t) = Active = Confirmed − Cured − Deaths\n"
        "- R_obs(t) = Removed = Cured + Deaths\n"
        "- S(t) = N − I(t) − R(t)"
    )
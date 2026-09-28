# -*- coding: utf-8 -*-
"""ACL 수소 수송 시뮬레이터 (Streamlit)
비정질 탄소 하부막(ACL) 3D 셀에서 중성 수소 H0와 수소 이온 H+의
침투·포획·방출을 실단위(초, nm)로 계산·시각화한다.
모든 기본값은 트렌드 수준 가정이며, 가정표(하단 expander)에 근거를 명시한다.
"""
import io
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

st.set_page_config(page_title="ACL 수소 수송 시뮬레이터", layout="wide")

KB = 8.617e-5      # eV/K
T0 = 473.15        # 200 C 기준
C_H0, C_HP = "#2f5fa8", "#b06c1a"
CMAP_H0 = LinearSegmentedColormap.from_list("h0", ["#f6f8fb", C_H0, "#12233f"])
CMAP_HP = LinearSegmentedColormap.from_list("hp", ["#fbf7f1", C_HP, "#3f2708"])
matplotlib.rcParams["font.family"] = ["Arial Narrow", "Archivo Narrow", "Arial", "sans-serif"]


# ---------------------------------------------------------------- solver
def make_xi(shape, seed, corr_passes):
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(shape).astype(np.float32)
    for _ in range(max(2, corr_passes)):
        s = xi.copy()
        for ax in range(3):
            s += np.concatenate([xi.take([0], ax), xi], ax).take(range(shape[ax]), ax)
            s += np.concatenate([xi, xi.take([-1], ax)], ax).take(range(1, shape[ax] + 1), ax)
        xi = s / 7.0
    xi -= xi.mean()
    xi /= xi.std() + 1e-12
    return xi


def arrh(T_c, ea):
    return float(np.exp(-ea / KB * (1.0 / (T_c + 273.15) - 1.0 / T0)))


@st.cache_data(show_spinner=False, max_entries=8)
def simulate(R, sG, Nt0, seed, T_c, logD0, Ea, drift, src, sigma_nm, t_peb,
             top_open, fine, t_total, nframes=60):
    """두 종(H0, H+)을 같은 무질서장 위에서 적분. 프레임 스냅샷과 시계열을 반환."""
    NX = NY = 48 if fine else 32
    NZ = 72 if fine else 48
    H = 36.0 / NZ
    shape = (NZ, NY, NX)      # z, y, x
    xi = make_xi(shape, int(seed), int(round(1.5 / H)))

    sp2 = R / (1.0 + R)
    sp3 = 1.0 - sp2
    D0a = 10.0 ** logD0
    A = arrh(T_c, Ea)
    D0h = D0a * np.exp(-2.2 * max(sp3 - 0.15, 0.0)) * A
    D0i = 0.5 * D0a * A
    s_site = np.clip(sp2 * (1.0 + 0.8 * xi), 0.0, 1.0).astype(np.float32)
    D = [
        (D0h * np.exp(0.9 * sG * np.clip(xi, -1.8, 1.8))).astype(np.float32),
        (D0i * s_site ** 2).astype(np.float32),
    ]
    kt = [np.full(shape, 0.05, np.float32), (3.0 * (1.0 - s_site)).astype(np.float32)]
    Nt = [
        (Nt0 * (1.0 + 0.3 * xi)).clip(0.05).astype(np.float32),
        (Nt0 * (1.0 - s_site + 0.15)).clip(0.05).astype(np.float32),
    ]
    kd = [1.0 * arrh(T_c, 0.4), 0.02 * arrh(T_c, 0.9)]

    Dm = []   # per species, per axis: harmonic-mean face D
    for spx in (0, 1):
        Df = D[spx]; per = []
        for ax in (0, 1, 2):
            n = Df.shape[ax]
            Da = Df.take(range(0, n - 1), ax); Db = Df.take(range(1, n), ax)
            per.append((2 * Da * Db / (Da + Db + 1e-30)).astype(np.float32))
        Dm.append(per)
    Dmax = max(float(D[0].max()), float(D[1].max()), 1e-9)
    vd_c = abs(drift) / H
    dt = min(0.6 * 0.16 * H * H / Dmax,
             (0.5 / vd_c) if vd_c > 0 else 1e9, 0.2 / 3.0, 0.05)

    u = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
    w = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
    esc = [0.0, 0.0]; og = [0.0, 0.0]; inj = [0.0, 0.0]
    zsrc = max(2, int(round(1.5 / H)))
    yy, xx_ = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    r2xy = (xx_ - NX / 2) ** 2 + (yy - NY / 2) ** 2
    sig_c2 = (sigma_nm / H) ** 2
    if src != "flux":
        for sp in (0, 1):
            if src == "point":
                u[sp][zsrc, NY // 2, NX // 2] = 1000.0
                inj[sp] = 1000.0
            else:
                zz = np.arange(NZ)[:, None, None]
                r2 = (zz - zsrc) ** 2 + r2xy[None]
                g = np.exp(-r2 / (2 * sig_c2)).astype(np.float32)
                g[g < 1e-4] = 0
                u[sp][:] = g
                inj[sp] = float(g.sum())
    flux_shape = (30.0 * np.exp(-r2xy / (2 * sig_c2))).astype(np.float32)

    # --- straightforward (readable) flux update, vectorized per axis ---
    h2 = H * H
    def step(sp, t_now):
        c = u[sp]; W = w[sp]; Df = D[sp]
        acc = np.zeros_like(c)
        for ax in (0, 1, 2):
            n = c.shape[ax]
            ca = c.take(range(0, n - 1), ax); cb = c.take(range(1, n), ax)
            f = Dm[sp][ax] * (cb - ca) / h2             # into lower cell
            pad = np.zeros_like(c.take([0], ax))
            acc += np.concatenate([f, pad], ax)         # + to lower-index cell
            acc -= np.concatenate([pad, f], ax)         # - from upper-index cell
        # top (z=0) boundary
        if top_open and t_now > t_peb:
            leak_t = float((D[sp][0] * c[0] / h2).sum()) * dt
            acc[0] -= D[sp][0] * c[0] / h2
        else:
            leak_t = 0.0
        # bottom absorbing
        leak_b = float((D[sp][-1] * c[-1] / h2).sum()) * dt
        acc[-1] -= D[sp][-1] * c[-1] / h2
        # drift (H+ only), upwind along z
        if sp == 1 and drift != 0.0:
            v = drift / H
            if v > 0:
                upw = np.concatenate([np.zeros_like(c[:1]), c[:-1]], 0)
            else:
                upw = np.concatenate([c[1:], np.zeros_like(c[:1])], 0)
            acc += abs(v) * (upw - c)
        srcterm = 0.0
        if src == "flux" and t_now < t_peb:
            acc[zsrc] += flux_shape
            srcterm = float(flux_shape.sum()) * dt
        cap = np.clip(1.0 - W / Nt[sp], 0.0, 1.0)
        trap = kt[sp] * c * cap - kd[sp] * W
        cn = c + dt * (acc - trap)
        np.clip(cn, 0, None, out=cn)
        W += dt * trap
        np.clip(W, 0, None, out=W)
        u[sp][:] = cn
        esc[sp] += leak_b; og[sp] += leak_t; inj[sp] += srcterm

    nsteps = max(1, int(round(t_total / dt)))
    snap_at = np.unique(np.linspace(0, nsteps, nframes).astype(int))
    frames = []          # per frame: dict per species
    times = []
    zaxis = np.arange(NZ) * H
    chk_t = [0.25, 0.5, 0.75, 1.0]
    chk3d = {}

    def record(k):
        rec = {"t": k * dt}
        for sp in (0, 1):
            tot_field = u[sp] + w[sp]
            prof = tot_field.mean(axis=(1, 2))
            tu = float(u[sp].sum()); tw = float(w[sp].sum())
            tot = tu + tw + esc[sp] + og[sp]
            mz = float((u[sp].sum(axis=(1, 2)) * zaxis).sum())
            rec[sp] = {
                "xz": tot_field[:, NY // 2, :].copy(),
                "prof": prof.copy(),
                "pen": mz / tu if tu > 0 else 0.0,
                "trap": tw / tot if tot > 0 else 0.0,
                "bt": esc[sp] / tot if tot > 0 else 0.0,
                "og": og[sp] / tot if tot > 0 else 0.0,
                "cons": abs(tot - inj[sp]) / inj[sp] if inj[sp] > 0 else 0.0,
            }
        times.append(rec["t"]); frames.append(rec)

    record(0)
    prog = st.progress(0.0, text="시뮬레이션 계산 중…")
    si = 1
    for k in range(1, nsteps + 1):
        t_now = k * dt
        step(0, t_now); step(1, t_now)
        if si < len(snap_at) and k >= snap_at[si]:
            record(k); si += 1
            prog.progress(k / nsteps, text=f"시뮬레이션 계산 중… t = {t_now:.1f}s / {t_total:.0f}s")
        for i_c, fr in enumerate(chk_t):
            if abs(k - fr * nsteps) < 1:
                chk3d[fr] = [(u[0] + w[0]).astype(np.float16),
                             (u[1] + w[1]).astype(np.float16)]
    prog.empty()
    meta = dict(NX=NX, NY=NY, NZ=NZ, H=H, dt=dt, nsteps=nsteps)
    return frames, np.array(times), chk3d, meta


# ---------------------------------------------------------------- UI
st.title("ACL 수소 수송 시뮬레이터")
st.caption("비정질 탄소 하부막 3D 셀 · H⁰(중성) vs H⁺(이온) · 실단위(초, nm) · "
           "기본값은 트렌드 수준 가정 — 하단 가정표 참조")

with st.sidebar:
    st.header("재료")
    R = st.slider("sp²/sp³ 비", 0.2, 5.0, 1.5, 0.1)
    st.caption(f"sp² 분율 {R/(1+R)*100:.0f}%")
    sG = st.slider("연결 복잡도 엔트로피 S_G", 0.0, 2.0, 1.0, 0.05)
    Nt0 = st.slider("트랩 사이트 밀도 (상대)", 0.2, 3.0, 1.0, 0.1)
    seed = st.number_input("무질서장 시드", 1, 9999, 929)
    st.header("공정")
    T_c = st.slider("온도 (°C)", 100, 300, 200, 5)
    logD0 = st.slider("기준 D₀ @200°C (log₁₀ nm²/s)", 0.0, 2.0, 1.0, 0.05)
    st.caption(f"D₀ = {10**logD0:.1f} nm²/s")
    Ea = st.slider("활성화에너지 E_a (eV)", 0.3, 1.5, 0.8, 0.05)
    drift = st.slider("H⁺ 드리프트 (nm/s, +기판쪽)", -8.0, 8.0, 0.0, 0.5)
    st.header("소스·경계")
    src = st.selectbox("유입 모드", ["flux", "gauss", "point"],
                       format_func=lambda v: {"flux": "지속 유입 (PEB 동안)",
                                              "gauss": "임펄스 · Gaussian",
                                              "point": "임펄스 · point"}[v])
    sigma_nm = st.slider("Gaussian σ (nm)", 1.5, 10.0, 5.0, 0.5)
    t_peb = st.slider("PEB 시간 (s)", 10, 120, 60, 5)
    top_open = st.selectbox("상부 경계 (PEB 후)", [False, True],
                            format_func=lambda v: "개방 (outgassing)" if v else "밀폐 (캡)")
    st.header("수치")
    fine = st.selectbox("해상도", [False, True],
                        format_func=lambda v: "정밀 48급 (느림)" if v else "빠름 32급")
    t_total = st.slider("총 모사 시간 (s)", 10, 120, 30, 5)
    st.caption("계산은 조건당 1회 수행 후 캐시됩니다")

frames, times, chk3d, meta = simulate(R, sG, Nt0, seed, T_c, logD0, Ea, drift,
                                      src, sigma_nm, t_peb, top_open, fine, float(t_total))
H = meta["H"]; NZ = meta["NZ"]

it = st.slider("시간 t (s)", 0.0, float(times[-1]), float(times[-1]),
               step=float(max(times[1], 0.1)))
fi = int(np.argmin(np.abs(times - it)))
fr = frames[fi]
phase = ""
if src == "flux":
    phase = " · PEB 유입 중" if fr["t"] < t_peb else " · PEB 종료 후"
st.markdown(f"**t = {fr['t']:.1f} s{phase}**")

mc = st.columns(8)
for sp, name in ((0, "H⁰"), (1, "H⁺")):
    d = fr[sp]
    mc[0 + 4 * sp].metric(f"{name} 침투 깊이", f"{d['pen']:.1f} nm")
    mc[1 + 4 * sp].metric(f"{name} 포획 분율", f"{d['trap']*100:.1f}%")
    mc[2 + 4 * sp].metric(f"{name} 기판 도달", f"{d['bt']*100:.2f}%")
    mc[3 + 4 * sp].metric(f"{name} 보존 오차", f"{d['cons']*100:.2f}%",
                          delta="정상" if d["cons"] < 0.005 else "확인 필요",
                          delta_color="normal" if d["cons"] < 0.005 else "inverse")

c1, c2 = st.columns(2)
for sp, (col, name, cm) in zip((0, 1), ((c1, "H⁰ 중성 — 자유부피 홉핑", CMAP_H0),
                                        (c2, "H⁺ 이온 — sp² 사이트 홉핑·깊은 포획", CMAP_HP))):
    with col:
        st.subheader(name)
        xz = fr[sp]["xz"]
        vmax = max(float(xz.max()), 1e-9)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(np.maximum(xz, vmax * 1e-3), cmap=cm,
                  norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax),
                  extent=[0, 24, 36, 0], aspect="auto")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("depth z (nm)  (top = MOR interface)")
        st.pyplot(fig, clear_figure=True)

st.subheader("깊이 프로파일 c̄(z) — SIMS 비교용")
fig, ax = plt.subplots(figsize=(9, 2.8))
zax = np.arange(NZ) * H
for sp, (c, lab) in enumerate(((C_H0, "H⁰"), (C_HP, "H⁺"))):
    p = np.maximum(fr[sp]["prof"], 1e-12)
    ax.semilogy(zax, p / max(frames[0][sp]["prof"].max(), 1e-12), color=c, lw=2, label=lab)
ax.set_xlabel("depth z (nm)"); ax.set_ylabel("relative concentration"); ax.set_ylim(1e-4, 2)
ax.legend(frameon=False); ax.grid(alpha=0.25)
st.pyplot(fig, clear_figure=True)

st.subheader("시계열")
fig, axs = plt.subplots(1, 3, figsize=(11, 2.6))
series = [("pen", "mean penetration (nm)", 1), ("trap", "trapped fraction (%)", 100), ("bt", "substrate arrival (%)", 100)]
for ax, (key, lab, sc) in zip(axs, series):
    for sp, c in ((0, C_H0), (1, C_HP)):
        ax.plot(times, [f[sp][key] * sc for f in frames], color=c, lw=2)
    ax.axvline(fr["t"], color="#888888", lw=1, ls=":")
    ax.set_title(lab, fontsize=10); ax.grid(alpha=0.25); ax.set_xlabel("t (s)")
st.pyplot(fig, clear_figure=True)

with st.expander("XY 수평 단면 (체크포인트 시점)"):
    frsel = st.select_slider("시점 (총 시간 대비)", options=[0.25, 0.5, 0.75, 1.0], value=1.0)
    zsel = st.slider("깊이 z (nm)", 0.5, 35.5, 12.0, 0.5)
    zi = min(NZ - 1, int(round(zsel / H)))
    cc1, cc2 = st.columns(2)
    for sp, (col, cm) in zip((0, 1), ((cc1, CMAP_H0), (cc2, CMAP_HP))):
        with col:
            f3 = chk3d.get(frsel)
            if f3 is None:
                st.info("이 시점의 체크포인트가 없습니다."); continue
            sl = f3[sp][zi].astype(np.float32)
            vmax = max(float(sl.max()), 1e-9)
            fig, ax = plt.subplots(figsize=(3.6, 3.4))
            ax.imshow(np.maximum(sl, vmax * 1e-3), cmap=cm,
                      norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), extent=[0, 24, 24, 0])
            ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
            st.pyplot(fig, clear_figure=True)

rep = "\n".join([
    f"ACL 수소 수송 시뮬레이터 리포트 (트렌드 수준, t={fr['t']:.1f} s)",
    f"조건: sp2/sp3={R} (sp2 {R/(1+R)*100:.0f}%), S_G={sG}, N_t={Nt0}x, seed={seed}",
    f"공정: T={T_c}C, D0={10**logD0:.1f} nm2/s, Ea={Ea} eV, drift(H+)={drift} nm/s, "
    f"유입={src}, t_PEB={t_peb} s, 상부={'개방' if top_open else '밀폐'}, 격자={meta['NX']}x{meta['NY']}x{meta['NZ']}",
    *(f"{n}: 침투 {fr[sp]['pen']:.1f} nm, 포획 {fr[sp]['trap']*100:.1f}%, "
      f"기판 도달 {fr[sp]['bt']*100:.2f}%, 상부 방출 {fr[sp]['og']*100:.2f}%, "
      f"보존오차 {fr[sp]['cons']*100:.2f}%" for sp, n in ((0, "H0"), (1, "H+"))),
])
csv = io.StringIO()
csv.write("t_s," + ",".join(f"{k}_{n}" for n in ("H0", "Hp") for k in ("pen_nm", "trap", "bt")) + "\n")
for i, f in enumerate(frames):
    csv.write(f"{times[i]:.2f}," + ",".join(
        f"{f[sp][k]:.5g}" for sp in (0, 1) for k in ("pen", "trap", "bt")) + "\n")
d1, d2 = st.columns(2)
d1.download_button("리포트 (txt)", rep, "acl_hsim_report.txt")
d2.download_button("시계열 (csv)", csv.getvalue(), "acl_hsim_series.csv")

with st.expander("가정표 — 수식·기본값·근거·조정 경로"):
    st.markdown(r"""
종별 2-장 모델 (h = 격자 간격, 막 두께 36 nm):

$\partial u/\partial t = \nabla\!\cdot\!(D(x,T)\nabla u) - v_d\,\partial u/\partial z - k_t(x)\,u\,(1-w/N_t) + k_d(T)\,w$
&nbsp;&nbsp;&nbsp; $\partial w/\partial t = +k_t u (1-w/N_t) - k_d w$

| 항목 | 기본값 | 성격 | 비고 |
|---|---|---|---|
| D₀(H⁰) @200°C | 10 nm²/s (1–100) | 가정 | fab 캘리브레이션 대상 (요청 ⑥ 연계) |
| E_a (확산) | 0.8 eV | 가정 | Arrhenius, T₀ = 473 K |
| sp³ 의존 | D₀ ∝ exp(−2.2(sp³−0.15)) | 경향 (Track B) | 치밀 사면체 네트워크가 채널을 닫음 |
| S_G 의존 | D(x) = D₀·exp(0.9·S_G·ξ) | 경향 (Track B) | ξ: 상관길이 ~1.5 nm 무질서장 |
| H⁺ 이동 | D ∝ s(x)², 기준 0.5·D₀ | 경향 | π-사이트 홉핑; v_d는 스택 대전 편향 대용 |
| 포획 H⁰ / H⁺ | k_d@200°C = 1 / 0.02 s⁻¹ (E_a 0.4 / 0.9 eV) | 가정 | 얕은 물리흡착형 vs 깊은 결함형 |
| 트랩 용량 | N_t (Langmuir 포화) | 가정 | 재고 포화 시 투과 가속 |

수치: 명시적 유한체적, 면 확산계수 조화평균, H⁺ 드리프트 상류차분, dt = 안정한계의 60%.
질량 보존 오차(주입 대비 재고+유출)는 상단 metric에 상시 표시. 절대 보정은 SIMS 프로파일
대조와 하이닉스 계측 상관 세트로 수행하는 것이 본 과제의 경로다.
""")

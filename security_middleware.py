"""
security_middleware.py - Near-RT RIC xApp: jamming detection + RIS null-steering.

Pipeline per slot:
    monitor(...)      -> AttackDetector looks at SINR/CQI degradation, estimates P_j
    reconfigure(...)  -> differentiable PyTorch optimizer picks (quantized) RIS phases

Notation follows environment.py:
    h_eff   = h_d^H  + h_r^H Theta G
    h_j_eff = h_ju^H + h_r^H Theta G_j
    SINR    = P_t ||h_eff||^2 / (P_j |h_j_eff|^2 + sigma^2)      (MRT beamforming at the BS)
"""
import math

import numpy as np
import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


# ======================================================================
# Helpers (numpy only)
# ======================================================================
def _db(x):
    return 10.0 * np.log10(x) if x > 0 else -np.inf


def sinr_db_to_cqi(sinr_db):
    """Approximate linear SINR->CQI mapping (about 1.9 dB per CQI step), clipped to 1..15."""
    return int(np.clip(round((sinr_db + 6.7) / 1.9), 1, 15))


def cqi_to_sinr_db(cqi):
    """Inverse of the approximate mapping above."""
    return 1.9 * float(cqi) - 6.7


CQI_MAX_SINR_DB = cqi_to_sinr_db(15)


def phase_levels(bits):
    """Discrete RIS phase set: 1 bit -> {0, pi}, 2 bit -> {0, pi/2, pi, 3pi/2}."""
    return np.arange(2 ** bits) * TWO_PI / (2 ** bits)


def quantize_phase(theta, bits):
    """Nearest-level projection onto the b-bit phase set (bits=None -> only wrap to [0, 2pi))."""
    theta = np.asarray(theta, dtype=np.float64)
    if bits is None:
        return np.mod(theta, TWO_PI)
    step = TWO_PI / (2 ** bits)
    return np.mod(np.round(theta / step) * step, TWO_PI)


def effective_channels_np(ch, phases):
    """Returns h_eff (M,) and h_j_eff (scalar) for given RIS phases."""
    e = np.exp(1j * np.asarray(phases, dtype=np.float64).reshape(-1))
    w = e * np.conj(np.asarray(ch["h_r"]).reshape(-1))  # h_r^H Theta
    h_eff = np.conj(np.asarray(ch["h_d"]).reshape(-1)) + w @ np.asarray(ch["G"])
    h_j = np.conj(np.asarray(ch["h_ju"]).reshape(-1))[0] + w @ np.asarray(ch["G_j"]).reshape(-1)
    return h_eff, h_j


def ris_metrics_np(ch, phases, p_t, p_j, sigma2):
    """Reference (non-differentiable) SINR evaluation, identical to environment.py."""
    h_eff, h_j = effective_channels_np(ch, phases)
    signal = p_t * float(np.sum(np.abs(h_eff) ** 2))
    interf = p_j * float(np.abs(h_j) ** 2)
    sinr = signal / (interf + sigma2)
    return {
        "sinr": sinr, "sinr_db": _db(sinr), "rate": math.log2(1.0 + sinr),
        "snr_db": _db(signal / sigma2), "inr_db": _db(interf / sigma2),
        "signal": signal, "interference": interf,
    }


# ======================================================================
# 1) Anomaly / attack detector
# ======================================================================
class AttackDetector:
    """
    Flags a jamming attack on sudden SINR (or CQI) degradation.

    degradation = reference - measured, where reference is
      - expected_sinr_db (jam-free SINR predicted from the middleware's own channel
        knowledge) if provided: robust to fast fading, or
      - an EWMA baseline of past SINR otherwise (updated only while not under attack).

    Hysteresis: trigger after `trigger_count` consecutive drops >= drop_db; release after
    `release_count` consecutive informative measurements with degradation <= release_db.

    `informative=False` marks a measurement taken while the RIS was already nulling the
    jammer: it says nothing about whether the jammer is still there (a good null looks
    the same as no jammer), so it never releases the attack state.
    """

    def __init__(self, drop_db=10.0, release_db=4.0, trigger_count=1, release_count=1, ewma_alpha=0.1):
        self.drop_db = drop_db
        self.release_db = release_db
        self.trigger_count = trigger_count
        self.release_count = release_count
        self.alpha = ewma_alpha
        self.under_attack = False
        self.baseline_db = None
        self.degradation_db = 0.0
        self._viol = 0
        self._ok = 0

    def update(self, sinr_db=None, expected_sinr_db=None, cqi=None, informative=True):
        from_cqi = sinr_db is None
        if from_cqi:
            if cqi is None:
                raise ValueError("Provide sinr_db or cqi.")
            sinr_db = cqi_to_sinr_db(cqi)

        if expected_sinr_db is not None:
            ref = expected_sinr_db
        else:
            if self.baseline_db is None:
                self.baseline_db = sinr_db
            ref = self.baseline_db
        if from_cqi:  # CQI saturates at 15
            ref = min(ref, CQI_MAX_SINR_DB)

        self.degradation_db = ref - sinr_db

        if not self.under_attack:
            if self.degradation_db >= self.drop_db:
                self._viol += 1
                if self._viol >= self.trigger_count:
                    self.under_attack, self._viol, self._ok = True, 0, 0
            else:
                self._viol = 0
                if expected_sinr_db is None:
                    self.baseline_db += self.alpha * (sinr_db - self.baseline_db)
        elif informative:
            if self.degradation_db <= self.release_db:
                self._ok += 1
                if self._ok >= self.release_count:
                    self.under_attack, self._ok = False, 0
            else:
                self._ok = 0
        return self.under_attack


# ======================================================================
# 2) Differentiable SINR layer + optimizer (PyTorch)
# ======================================================================
def quantize_ste(theta, bits):
    """Round to the b-bit phase grid in the forward pass, identity gradient in the backward pass."""
    step = TWO_PI / (2 ** bits)
    q = torch.round(theta / step) * step
    return theta + (q - theta).detach()


class RISSINRLayer(nn.Module):
    """
    Holds R candidate RIS phase vectors theta (R, N) as parameters and evaluates the exact
    SINR formula with complex torch ops, so d(SINR)/d(theta) flows through
        h_eff = h_d^H + h_r^H diag(e^{j theta}) G,   h_j_eff = h_ju^H + h_r^H diag(e^{j theta}) G_j.
    With `bits` set, the forward pass uses STE-quantized phases (quantization-aware training).
    """

    def __init__(self, ch, p_t, p_j, sigma2, init_phases, bits=None):
        super().__init__()
        cplx = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.complex128)
        self.register_buffer("h_d_c", cplx(np.conj(ch["h_d"]).reshape(-1)))  # (M,)  h_d^H
        self.register_buffer("G", cplx(ch["G"]))  # (N, M)
        self.register_buffer("h_r_c", cplx(np.conj(ch["h_r"]).reshape(-1)))  # (N,)  h_r^H
        self.register_buffer("h_ju_c", cplx(np.conj(ch["h_ju"]).reshape(-1)))  # (1,)  h_ju^H
        self.register_buffer("G_j", cplx(np.asarray(ch["G_j"]).reshape(-1)))  # (N,)
        self.p_t, self.p_j, self.sigma2 = float(p_t), float(p_j), float(sigma2)
        self.bits = bits
        self.theta = nn.Parameter(torch.as_tensor(np.asarray(init_phases), dtype=torch.float64).clone())

    def phases(self):
        return self.theta if self.bits is None else quantize_ste(self.theta, self.bits)

    def evaluate(self, phases):
        """phases: (R, N) real tensor -> (sinr, signal, interference), each (R,)."""
        w = torch.exp(1j * phases) * self.h_r_c  # (R, N)  h_r^H Theta
        h_eff = self.h_d_c + w @ self.G  # (R, M)
        h_j = self.h_ju_c + w @ self.G_j  # (R,)
        signal = self.p_t * (h_eff.real ** 2 + h_eff.imag ** 2).sum(-1)
        interf = self.p_j * (h_j.real ** 2 + h_j.imag ** 2)
        return signal / (interf + self.sigma2), signal, interf

    def forward(self):
        return self.evaluate(self.phases())[0]


def _objective(sinr, interf, sigma2, null_weight):
    """Per-candidate loss: -log2(1+SINR) (+ optional extra penalty on jammer leakage)."""
    return -torch.log2(1.0 + sinr) + null_weight * torch.log1p(interf / sigma2)


def _run_adam(layer, steps, lr, null_weight):
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        sinr, _, interf = layer.evaluate(layer.phases())
        loss = _objective(sinr, interf, layer.sigma2, null_weight).sum()  # restarts are independent
        loss.backward()
        opt.step()


def _greedy_polish(layer, phases, levels, sweeps, null_weight):
    """Optional discrete coordinate-descent pass over the allowed levels (not differentiable)."""
    cur = phases.copy()
    n_el, n_lv = cur.size, len(levels)
    with torch.no_grad():
        def obj(batch):
            s, _, i = layer.evaluate(torch.as_tensor(batch, dtype=torch.float64))
            return _objective(s, i, layer.sigma2, null_weight).numpy()

        cur_obj = obj(cur[None, :])[0]
        for _ in range(sweeps):
            improved = False
            for n in range(n_el):
                batch = np.tile(cur, (n_lv, 1))
                batch[:, n] = levels
                vals = obj(batch)
                k = int(np.argmin(vals))
                if vals[k] < cur_obj - 1e-12:
                    cur, cur_obj, improved = batch[k].copy(), vals[k], True
            if not improved:
                break
    return cur


def optimize_ris_phases(
        ch, p_t, p_j, sigma2, bits=None, init_phases=None, num_restarts=8,
        cont_steps=200, qat_steps=100, lr=0.1, null_weight=0.0, polish_sweeps=2, seed=0,
):
    """
    Gradient-based RIS phase design.
      1. Continuous Adam stage on R random restarts (+ warm start in row 0).
      2. If bits is set: STE quantization-aware fine-tuning.
      3. Pick the best exactly-quantized candidate; optionally polish greedily on the grid.
    p_j is the assumed jammer power (W); p_j = 0 gives the plain max-beam-gain design.
    null_weight > 0 adds an explicit penalty on residual jammer power on top of the SINR objective.
    Returns (phases float32 (N,), metrics dict).
    """
    n_el = np.asarray(ch["h_r"]).size
    rng = np.random.default_rng(seed)
    init = rng.uniform(0.0, TWO_PI, size=(num_restarts, n_el))
    if init_phases is not None:
        init[0] = np.asarray(init_phases, dtype=np.float64).reshape(-1)

    layer = RISSINRLayer(ch, p_t, p_j, sigma2, init, bits=None)
    _run_adam(layer, cont_steps, lr, null_weight)
    candidates = [quantize_phase(layer.theta.detach().numpy(), bits)]

    if bits is not None and qat_steps > 0:
        layer.bits = bits
        _run_adam(layer, qat_steps, lr * 0.3, null_weight)
        candidates.append(quantize_phase(layer.theta.detach().numpy(), bits))

    cands = np.concatenate(candidates, axis=0)
    with torch.no_grad():
        sinr, _, interf = layer.evaluate(torch.as_tensor(cands, dtype=torch.float64))
        best = cands[int(torch.argmin(_objective(sinr, interf, layer.sigma2, null_weight)))].copy()

    if bits is not None and polish_sweeps > 0:
        best = _greedy_polish(layer, best, phase_levels(bits), polish_sweeps, null_weight)

    return best.astype(np.float32), ris_metrics_np(ch, best, p_t, p_j, sigma2)


# ======================================================================
# 3) The xApp: detector + optimizer + mode logic
# ======================================================================
class SecurityMiddleware:
    """
    Normal mode     : optimize phases for beam gain only (assumed P_j = 0).
    Mitigation mode : optimize the full SINR with an estimated P_j, which steers a null at the jammer.

    Typical loop (see __main__):
        phases = mw.reconfigure(channels_t)
        ... apply phases, get measured SINR / CQI ...
        mw.monitor(channels_t, measured_sinr_db=...)

    Every `probe_interval` mitigation slots, one slot runs the normal-mode phases as a probe. Only
    jammer-unaware slots reveal whether the jammer is still present, so only they can release the
    attack state (set probe_interval=0 to disable probing and stay in mitigation once triggered).
    """

    def __init__(self, num_elements, p_t, sigma2, bits=2, detector=None, default_pj_dbm=30.0,
                 probe_interval=10, seed=0, **opt_kwargs):
        self.N, self.p_t, self.sigma2, self.bits = num_elements, p_t, sigma2, bits
        self.detector = detector or AttackDetector()
        self.default_pj_w = 10 ** ((default_pj_dbm - 30.0) / 10.0)
        self.probe_interval = probe_interval
        self.opt_kwargs = opt_kwargs
        self.seed = seed
        self.phases = np.zeros(num_elements, dtype=np.float32)
        self.p_j_hat_w = None
        self.last_info = {"mode": "normal"}
        self._phases_mode = "normal"
        self._since_probe = 0
        self._calls = 0

    # ---- monitoring ----
    def monitor(self, channels, measured_sinr_db=None, cqi=None, phases=None):
        """Feed one SINR/CQI measurement taken with `phases` (default: last applied) on `channels`."""
        phases = self.phases if phases is None else phases
        informative = self._phases_mode == "normal"
        expected = ris_metrics_np(channels, phases, self.p_t, 0.0, self.sigma2)["snr_db"]  # jam-free prediction
        attack = self.detector.update(sinr_db=measured_sinr_db, expected_sinr_db=expected,
                                      cqi=cqi, informative=informative)
        if attack and informative:
            meas = measured_sinr_db if measured_sinr_db is not None else cqi_to_sinr_db(cqi)
            self._estimate_jamming_power(channels, phases, meas)
        if not attack:
            self.p_j_hat_w = None
        return attack

    def _estimate_jamming_power(self, channels, phases, measured_sinr_db):
        """From I = S/SINR - sigma^2 and |h_j_eff|^2 under the applied phases; clipped to 10-40 dBm."""
        m = ris_metrics_np(channels, phases, self.p_t, 0.0, self.sigma2)
        _, h_j = effective_channels_np(channels, phases)
        interference = m["signal"] / (10 ** (measured_sinr_db / 10.0)) - self.sigma2
        gain = float(np.abs(h_j) ** 2)
        if interference <= 0 or gain < 1e-30:
            return
        p_dbm = float(np.clip(10 * np.log10(interference / gain) + 30.0, 10.0, 40.0))
        if self.p_j_hat_w is not None:  # light smoothing in the dB domain
            p_dbm = 0.5 * p_dbm + 0.5 * (10 * np.log10(self.p_j_hat_w) + 30.0)
        self.p_j_hat_w = 10 ** ((p_dbm - 30.0) / 10.0)

    # ---- reconfiguration ----
    def reconfigure(self, channels):
        """Returns the (quantized) RIS phases for the current channel estimates, float32 (N,)."""
        probe = False
        if self.detector.under_attack:
            if self.probe_interval > 0 and self._since_probe >= self.probe_interval:
                probe, self._since_probe = True, 0
            else:
                self._since_probe += 1
        mitigate = self.detector.under_attack and not probe

        p_j = (self.p_j_hat_w or self.default_pj_w) if mitigate else 0.0
        self._calls += 1
        phases, metrics = optimize_ris_phases(
            channels, self.p_t, p_j, self.sigma2, bits=self.bits, init_phases=self.phases,
            seed=self.seed + self._calls, **self.opt_kwargs)

        self.phases = phases
        self._phases_mode = "mitigation" if mitigate else "normal"
        self.last_info = {
            "mode": self._phases_mode, "probe": probe,
            "p_j_assumed_dbm": (10 * np.log10(p_j) + 30.0) if p_j > 0 else None,
            "predicted_sinr_db": metrics["sinr_db"], "predicted_inr_db": metrics["inr_db"],
        }
        return phases


# ======================================================================
# Demo: attack onset in the security-aware environment
# ======================================================================
if __name__ == "__main__":
    from environment import RISEnvironment
    from jammer_agent import BarrageJammer

    def channels_from_env(env):
        # Legitimate channels are known; interference channels are the (noisy) estimates.
        return {"h_d": env.h_d, "G": env.G, "h_r": env.h_r, "h_ju": env.h_ju_hat, "G_j": env.G_j_hat}

    def run(bits, defend, n_slots=30, attack_slot=10, p_j_dbm=30.0, nmse=0.05, seed=0):
        np.random.seed(seed)  # same channel sequence for every run
        env = RISEnvironment(num_antennas=4, num_elements=16, jammer_type="none",
                             interference_csi_nmse=nmse)
        env.reset(seed=seed)
        mw = SecurityMiddleware(env.N, env.P_t, env.sigma2, bits=bits, seed=seed)
        rates, modes = [], []
        for t in range(n_slots):
            if t == attack_slot:  # attack onset
                env.jammer = BarrageJammer(power_dbm=p_j_dbm, num_symbols=1, seed=seed)
            ch = channels_from_env(env)
            phases = mw.reconfigure(ch)
            _, reward, _, _, info = env.step(phases)
            rates.append(reward)
            if defend:
                mw.monitor(ch, measured_sinr_db=info["sinr_db"])
            modes.append(mw.last_info["mode"])
        return np.array(rates), modes

    print("Attack onset at slot 10 (barrage, 30 dBm). Mean rate in bps/Hz:")
    print(f"{'phase res.':>10} | {'pre-attack':>10} | {'no defense':>10} | {'defended':>10} | mitigation slots")
    for bits in (None, 2, 1):
        r_nd, _ = run(bits, defend=False)
        r_d, modes = run(bits, defend=True)
        label = "continuous" if bits is None else f"{bits}-bit"
        print(f"{label:>10} | {r_nd[:10].mean():10.2f} | {r_nd[10:].mean():10.2f} | "
              f"{r_d[10:].mean():10.2f} | {modes[10:].count('mitigation')}/20")
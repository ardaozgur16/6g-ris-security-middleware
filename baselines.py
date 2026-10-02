import numpy as np
from channel_model import WirelessChannel

TWO_PI = 2.0 * np.pi


def _db(x):
    return 10.0 * np.log10(max(x, 1e-30))  # floored so an exact null reads -300 dB instead of -inf


def quantize_phase(theta, bits):
    """Nearest-level projection onto the b-bit phase grid (bits=None -> only wrap to [0, 2pi))."""
    theta = np.asarray(theta, dtype=np.float64)
    if bits is None:
        return np.mod(theta, TWO_PI)
    step = TWO_PI / (2 ** bits)
    return np.mod(np.round(theta / step) * step, TWO_PI)


# ----------------------------------------------------------------------
# Helpers for the analytical null-steering solver
# ----------------------------------------------------------------------
def _null_residual(theta, c, t):
    """g(theta) = sum_n c_n e^{j theta_n} - t  (g = 0  <=>  h_j_eff = 0)."""
    return np.abs(np.sum(c * np.exp(1j * theta)) - t)


def _restore_null(theta, c, t, iters=100, tol=1e-12):
    """Damped Gauss-Newton on the two real equations Re(g) = Im(g) = 0 (also minimizes |g| if infeasible)."""
    theta = theta.copy()
    for _ in range(iters):
        z = c * np.exp(1j * theta)
        g = z.sum() - t
        if abs(g) < tol:
            break
        J = np.vstack([-z.imag, z.real])  # d[Re g, Im g]/d theta, shape (2, N)
        JJt = J @ J.T
        JJt += 1e-9 * np.trace(JJt) * np.eye(2)
        try:
            step = J.T @ np.linalg.solve(JJt, np.array([g.real, g.imag]))
        except np.linalg.LinAlgError:
            # Fallback mechanism if the matrix is singular
            step = np.zeros_like(theta)
            break
        theta -= np.clip(step, -0.5, 0.5)
    return theta


def _maximize_gain_on_null(theta, A, h0, c, t, max_iters=300, step0=0.3):
    """
    Projected gradient ascent of f(theta) = ||h0 + sum_n e^{j theta_n} A_n||^2 on the manifold
    {theta : sum_n c_n e^{j theta_n} = t}. Ascent direction is projected onto the tangent space of the
    constraint, then the constraint is restored by Gauss-Newton; steps are accepted only if f increases.
    """

    def gain(th):
        return float(np.sum(np.abs(h0 + np.exp(1j * th) @ A) ** 2))

    step, best = step0, gain(theta)
    for _ in range(max_iters):
        e = np.exp(1j * theta)
        h_eff = h0 + e @ A
        grad = -2.0 * np.imag((e[:, None] * A) @ np.conj(h_eff))  # df/dtheta_n

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm

        z = c * e
        J = np.vstack([-z.imag, z.real])
        JJt = J @ J.T + 1e-12 * np.trace(J @ J.T) * np.eye(2)

        try:
            d = grad - J.T @ np.linalg.solve(JJt, J @ grad)  # tangent-space projection
        except np.linalg.LinAlgError:
            break

        nd = np.linalg.norm(d)
        if nd < 1e-10:
            break
        d /= nd

        improved = False
        while step > 1e-8:
            cand = _restore_null(theta + step * d, c, t, iters=20)
            if _null_residual(cand, c, t) < 1e-9 and gain(cand) > best:
                theta, best, improved = cand, gain(cand), True
                step = min(step * 1.3, 1.0)
                break
            step *= 0.5
        if not improved:
            break
    return theta


class BaselineSolvers:
    """
    Reference algorithms for validating the jamming defense. Every method takes the five channels
    returned by WirelessChannel.get_channel_realization() -> (h_d, G, h_r, h_ju, G_j)
    and returns (metrics, phases); metrics always come from the true SINR

        SINR = P_t ||h_eff||^2 / (P_j |h_j_eff|^2 + sigma^2).
    """

    def __init__(self, num_antennas=4, num_elements=16, tx_power_dbm=30.0, noise_power_dbm=-90.0,
                 jammer_power_dbm=30.0, phase_bits=None):
        self.M = num_antennas
        self.N = num_elements
        self.P_t = 10 ** ((tx_power_dbm - 30.0) / 10.0)
        self.sigma2 = 10 ** ((noise_power_dbm - 30.0) / 10.0)
        self.P_j = 10 ** ((jammer_power_dbm - 30.0) / 10.0)
        self.bits = phase_bits  # hardware phase resolution for the "practical" baselines (None = continuous)

    # ------------------------------------------------------------------
    def evaluate(self, h_d, G, h_r, h_ju, G_j, phases=None):
        """Exact SINR under jamming for given RIS phases (phases=None -> no RIS at all)."""
        h_eff = np.conj(h_d).reshape(-1)
        h_j = np.conj(h_ju).reshape(-1)[0]
        if phases is not None:
            w = np.exp(1j * np.asarray(phases, dtype=np.float64)) * np.conj(h_r).reshape(-1)  # h_r^H Theta
            h_eff = h_eff + w @ G
            h_j = h_j + w @ G_j.reshape(-1)
        signal = self.P_t * float(np.linalg.norm(h_eff) ** 2)  # MRT beamforming
        interference = self.P_j * float(np.abs(h_j) ** 2)
        sinr = signal / (interference + self.sigma2)
        return {
            "rate": float(np.log2(1.0 + sinr)),
            "sinr_db": _db(sinr),
            "inr_db": _db(interference / self.sigma2),
            "snr_db": _db(signal / self.sigma2),  # same phases, jammer off
            "rate_no_jam": float(np.log2(1.0 + signal / self.sigma2)),
        }

    # ------------------------------------------------------------------
    def no_ris(self, h_d, G, h_r, h_ju, G_j):
        """
        1. No-RIS baseline: BS -> user direct path only, jammer reaches the user only through h_ju.
        (In channel_model.py the direct BS->user path is Rayleigh/NLoS, not line-of-sight.)
        """
        return self.evaluate(h_d, G, h_r, h_ju, G_j, phases=None), None

    def random_phase(self, h_d, G, h_r, h_ju, G_j):
        """2. Uncoordinated RIS: random phases (on the b-bit grid if phase_bits is set), under jamming."""
        angles = quantize_phase(np.random.uniform(0, TWO_PI, size=self.N), self.bits)
        return self.evaluate(h_d, G, h_r, h_ju, G_j, phases=angles), angles

    def _ao_phases(self, h_d, G, h_r, max_iters=20):
        """
        Jam-free coordinate-wise phase alignment (the original alternating optimization):
        each element is rotated to add constructively to the sum of all the others.
        """
        theta_angles = np.zeros(self.N)
        for _ in range(max_iters):
            for n in range(self.N):
                Theta_temp = np.diag(np.exp(1j * theta_angles))
                Theta_temp[n, n] = 0  # drop element n
                h_other = h_d.conj().T + np.dot(np.dot(h_r.conj().T, Theta_temp), G)
                a_n = h_r[n].conj() * G[n:n + 1, :]
                inner_prod = np.dot(h_other, a_n.conj().T)[0, 0]
                theta_angles[n] = np.angle(inner_prod) % TWO_PI
        return theta_angles

    def unprotected_ris(self, h_d, G, h_r, h_ju, G_j, max_iters=20):
        """
        3. Unprotected AI-RIS: phases optimized only for the legitimate link (jammer-unaware), quantized to
        the hardware resolution, then evaluated under attack. metrics["rate_no_jam"] is its jammer-free rate,
        so rate_no_jam - rate is the damage done by the attack.
        """
        angles = quantize_phase(self._ao_phases(h_d, G, h_r, max_iters), self.bits)
        return self.evaluate(h_d, G, h_r, h_ju, G_j, phases=angles), angles

    # ------------------------------------------------------------------
    def analytical_null_steering(self, h_d, G, h_r, h_ju, G_j, num_starts=8, max_iters=300, seed=0):
        """
        4. Analytical null-steering / zero-forcing (continuous phases, perfect CSI).

        Exact cancellation h_j_eff = 0 needs  sum_n c_n e^{j theta_n} = -h_ju^H,  c_n = conj(h_r[n]) G_j[n].
        With unit-modulus phases the reachable magnitudes of the left side are
        [max(0, 2 max|c_n| - sum|c_n|), sum|c_n|], so a null exists iff |h_ju| lies in that interval
        (null_margin_db = 20 log10(sum|c_n| / |h_ju|) must be >= 0).

        If a null exists: find feasible phases (damped Gauss-Newton from random starts), then maximize the
        legitimate gain ||h_eff||^2 on the null manifold with projected gradient ascent (best of num_starts;
        the problem is non-convex, so this is a local optimum per start).
        If not: best-effort fallback that minimizes |h_j_eff| (reflected path set directly against the direct
        path); metrics["null_feasible"] is False and this is NOT a valid upper bound.
        """
        rng = np.random.default_rng(seed)
        hr_c = np.conj(h_r).reshape(-1)
        c = hr_c * G_j.reshape(-1)  # reflected-jammer coefficients
        t = -np.conj(h_ju).reshape(-1)[0]  # required reflected sum
        mags = np.abs(c)
        total, lower = mags.sum(), max(0.0, 2.0 * mags.max() - mags.sum())
        margin_db = 20.0 * np.log10(max(total / abs(t), 1e-30)) if abs(t) > 0 else np.inf

        A = hr_c[:, None] * G  # (N, M) legitimate cascaded rows
        h0 = np.conj(h_d).reshape(-1)
        scale = max(mags.mean(), 1e-15)  # normalize so tolerances are scale-free
        cn, tn = c / scale, t / scale

        theta, best_gain = None, -np.inf
        if lower <= abs(t) <= total:
            for _ in range(num_starts):
                th = _restore_null(rng.uniform(0, TWO_PI, self.N), cn, tn, iters=200)
                if _null_residual(th, cn, tn) > 1e-9:
                    continue
                th = _maximize_gain_on_null(th, A / scale, h0 / scale, cn, tn, max_iters)
                g = float(np.sum(np.abs(h0 + np.exp(1j * th) @ A) ** 2))
                if g > best_gain:
                    theta, best_gain = th, g
        feasible = theta is not None

        if not feasible:  # best-effort: minimize |h_j_eff|
            starts = [np.angle(t) - np.angle(c)] + [rng.uniform(0, TWO_PI, self.N) for _ in range(num_starts)]
            starts = [_restore_null(s, cn, tn, iters=100) for s in starts]
            theta = min(starts, key=lambda s: _null_residual(s, cn, tn))

        theta = np.mod(theta, TWO_PI)
        metrics = self.evaluate(h_d, G, h_r, h_ju, G_j, phases=theta)
        metrics.update(null_feasible=bool(feasible), null_margin_db=float(margin_db))
        return metrics, theta


# ----------------------------------------------------------------------
# MONTE CARLO COMPARISON
# ----------------------------------------------------------------------
def _noisy_estimate(x, nmse, rng):
    """Channel estimate with the given NMSE (same model as environment.py)."""
    if nmse <= 0:
        return x
    err = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2.0)
    return x + np.sqrt(nmse * np.mean(np.abs(x) ** 2)) * err


if __name__ == "__main__":
    M = 4
    NUM_TRIALS = 50
    P_J_DBM = 30.0
    PHASE_BITS = 2  # practical baselines and the proposed optimizer use 2-bit hardware
    CSI_NMSE = 0.05  # interference-CSI error seen by the proposed optimizer

    SCENARIOS = [
        {"name": "A) jammer close to user (55,-15,1.5), N=32", "N": 32, "jammer_pos": (55, -15, 1.5)},
        {"name": "B) distant jammer (30,-60,1.5), N=64", "N": 64, "jammer_pos": (30, -60, 1.5)},
    ]

    try:  # optional: needs PyTorch
        from security_middleware import optimize_ris_phases
    except ImportError:
        optimize_ris_phases = None

    np.random.seed(0)
    rng = np.random.default_rng(0)

    for sc in SCENARIOS:
        N = sc["N"]
        channel_sim = WirelessChannel(num_antennas=M, num_elements=N, jammer_pos=sc["jammer_pos"])
        solvers = BaselineSolvers(M, N, jammer_power_dbm=P_J_DBM, phase_bits=PHASE_BITS)

        methods = {
            "No-RIS (direct path)": solvers.no_ris,
            f"Random phase RIS ({PHASE_BITS}-bit)": solvers.random_phase,
            f"Unprotected AI-RIS ({PHASE_BITS}-bit, jam-unaware)": solvers.unprotected_ris,
            "Null-steering ZF (continuous, perfect CSI)": solvers.analytical_null_steering,
        }
        results = {k: [] for k in methods}
        proposed, feas, margins, no_jam = [], [], [], []
        if optimize_ris_phases is not None:
            key = f"Proposed jammer-aware ({PHASE_BITS}-bit, NMSE={CSI_NMSE})"
            results[key] = []

        print(f"\n=== {sc['name']} | P_j = {P_J_DBM:.0f} dBm | {NUM_TRIALS} trials ===")
        for trial in range(NUM_TRIALS):
            ch = channel_sim.get_channel_realization()  # (h_d, G, h_r, h_ju, G_j)
            for name, fn in methods.items():
                m, _ = fn(*ch)
                results[name].append(m)
                if name.startswith("Null"):
                    feas.append(m["null_feasible"])
                    margins.append(m["null_margin_db"])
                if name.startswith("Unprotected"):
                    no_jam.append(m["rate_no_jam"])
            if optimize_ris_phases is not None:
                h_d, G, h_r, h_ju, G_j = ch
                est = {"h_d": h_d, "G": G, "h_r": h_r,
                       "h_ju": _noisy_estimate(h_ju, CSI_NMSE, rng), "G_j": _noisy_estimate(G_j, CSI_NMSE, rng)}
                ph, _ = optimize_ris_phases(est, solvers.P_t, solvers.P_j, solvers.sigma2,
                                            bits=PHASE_BITS, seed=trial)
                results[key].append(solvers.evaluate(*ch, phases=ph))

        print(f"{'Method':<52} {'Rate (bps/Hz)':>13} {'SINR med.':>10} {'INR med.':>9}")
        print(f"{'Unprotected AI-RIS, jammer OFF (reference)':<52} {np.mean(no_jam):13.3f} {'-':>10} {'-':>9}")
        for name, ms in results.items():
            print(f"{name:<52} {np.mean([m['rate'] for m in ms]):13.3f} "
                  f"{np.median([m['sinr_db'] for m in ms]):10.2f} {np.median([m['inr_db'] for m in ms]):9.1f}")
        zf = results["Null-steering ZF (continuous, perfect CSI)"]
        zf_ok = [m["rate"] for m in zf if m["null_feasible"]]
        if zf_ok:
            print(f"ZF rate over feasible-null trials only: {np.mean(zf_ok):.3f} bps/Hz")
        print(f"Exact null feasible in {100 * np.mean(feas):.0f}% of trials "
              f"(mean RIS-vs-direct margin {np.mean(margins):+.1f} dB; need >= 0 dB for a null)")
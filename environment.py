import gymnasium as gym
from gymnasium import spaces
import numpy as np

from channel_model import WirelessChannel
from jammer_agent import BarrageJammer, ReactiveJammer, dbm_to_watt, watt_to_dbm


class RISEnvironment(gym.Env):
    """
    RIS Destekli MISO Sistemi için Güvenlik Odaklı (Jammer Farkındalıklı) Gymnasium Ortamı.
    Pekiştirmeli Öğrenme ajanları (DDPG, TD3, PPO, SAC vb.) ile tam uyumludur.

    Performans metriği SNR yerine SINR'dir:
        h_eff   = h_d^H  + h_r^H Theta G
        h_j_eff = h_ju^H + h_r^H Theta G_j
        SINR    = P_t ||h_eff||^2 / (P_j |h_j_eff|^2 + sigma^2)
        Ödül    = log2(1 + SINR)
    """
    metadata = {"render_modes": []}

    def __init__(
            self,
            num_antennas=4,  # M: Baz İstasyonu anten sayısı
            num_elements=16,  # N: RIS eleman sayısı
            tx_power_dbm=30.0,  # İletim gücü (dBm) -> 30 dBm = 1 Watt
            noise_power_dbm=-90.0,  # Gürültü gücü (dBm) -> -90 dBm = 1e-9 Watt
            rician_factor_k=3.0,
            jammer_pos=(55.0, -15.0, 1.5),  # Jammer 3D koordinatı (metre)
            jammer_type="barrage",  # "barrage" | "reactive" | "none"
            jammer_power_dbm=30.0,  # P_j: 10-40 dBm aralığında
            jammer_kwargs=None,  # Jammer'a özel parametreler (reaction_delay, burst_len, seed ...)
            interference_csi_nmse=0.0  # Girişim kanalı tahmin hatası (NMSE, 0 = mükemmel CSI)
    ):
        super(RISEnvironment, self).__init__()

        self.M = num_antennas
        self.N = num_elements

        # Güç birimlerini dBm'den Watt'a (Lineer) dönüştür
        self.P_t = dbm_to_watt(tx_power_dbm)
        self.sigma2 = dbm_to_watt(noise_power_dbm)

        # 1. DÜZELTME: Güncellenmiş WirelessChannel parametrelerine uyum sağlandı.
        self.channel_sim = WirelessChannel(
            num_antennas=self.M,
            num_elements=self.N,
            jammer_pos=jammer_pos,
            k_factor_bs_ris=10.0,  # Kuleler arası varsayılan güçlü LoS
            k_factor_ris_user=rician_factor_k,  # Kullanıcının parametresi
            k_factor_jammer_ris=2.0  # Yer seviyesi jammer için zayıf LoS
        )

        # Jammer ajanını başlat
        self.jammer_type = jammer_type
        self.jammer_power_dbm = jammer_power_dbm
        self.jammer_kwargs = dict(jammer_kwargs or {})
        self.nmse = interference_csi_nmse
        self.reset_jammer()

        # AKSİYON UZAYI: N adet RIS elemanı için faz açıları [0, 2*pi]
        self.action_space = spaces.Box(
            low=0.0,
            high=2.0 * np.pi,
            shape=(self.N,),
            dtype=np.float32
        )

        # GÖZLEM UZAYI: h_d, G, h_r, h_ju, G_j (Reel ve İmajiner birleşik)
        self.obs_dim = 2 * (self.M + self.N * self.M + self.N + 1 + self.N)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Anlık kanal durumları (gerçek değerler)
        self.h_d = None
        self.G = None
        self.h_r = None
        self.h_ju = None
        self.G_j = None
        # Ajanın gördüğü girişim kanalı tahminleri
        self.h_ju_hat = None
        self.G_j_hat = None

    # ------------------------------------------------------------------
    # Jammer yönetimi
    # ------------------------------------------------------------------
    def reset_jammer(self):
        """Jammer durumunu ve slot sayacını sıfırlar."""
        kw = dict(self.jammer_kwargs)
        kw.setdefault("num_symbols", 1)  # SINR formülü yalnızca P_j kullanır
        if self.jammer_type == "barrage":
            self.jammer = BarrageJammer(power_dbm=self.jammer_power_dbm, **kw)
        elif self.jammer_type == "reactive":
            self.jammer = ReactiveJammer(power_dbm=self.jammer_power_dbm, **kw)
        elif self.jammer_type == "none":
            self.jammer = None
        else:
            raise ValueError(f"Unknown jammer_type: {self.jammer_type}")
        self._slot = 0

    def _estimate(self, x):
        """Girişim kanalının gürültülü tahminini üretir (NMSE = nmse)."""
        if self.nmse <= 0.0:
            return x
        
        # 2. DÜZELTME: Güvenli rastgele sayı üreticisi referansı (reset'ten önce çağrılma ihtimaline karşı)
        rng = getattr(self, 'np_random', np.random.default_rng())
        err = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2.0)
        return x + np.sqrt(self.nmse * np.mean(np.abs(x) ** 2)) * err

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Meşru kanal katsayıları ve girişim kanalı tahminlerini birleştirir."""
        scale_factor = 1e3  # Gözlemleri daha sayısal kararlı aralığa çeker

        obs = np.concatenate([
            self.h_d.flatten().real * scale_factor,
            self.h_d.flatten().imag * scale_factor,
            self.G.flatten().real * scale_factor,
            self.G.flatten().imag * scale_factor,
            self.h_r.flatten().real * scale_factor,
            self.h_r.flatten().imag * scale_factor,
            self.h_ju_hat.flatten().real * scale_factor,
            self.h_ju_hat.flatten().imag * scale_factor,
            self.G_j_hat.flatten().real * scale_factor,
            self.G_j_hat.flatten().imag * scale_factor,
        ])
        return obs.astype(np.float32)

    def _new_channel_state(self):
        """Yeni bir kanal gerçekleşimi üretir ve tahminleri hazırlar."""
        self.h_d, self.G, self.h_r, self.h_ju, self.G_j = self.channel_sim.get_channel_realization()
        self.h_ju_hat = self._estimate(self.h_ju)
        self.G_j_hat = self._estimate(self.G_j)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._new_channel_state()

        if options and options.get("reset_jammer", False):
            self.reset_jammer()

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Action: (N,) boyutlu dizi [0, 2*pi] arasındaki faz açıları"""
        phase_shifts = np.clip(np.asarray(action, dtype=np.float32), 0.0, 2.0 * np.pi)
        Theta = np.diag(np.exp(1j * phase_shifts))

        h_r_H = self.h_r.conj().T
        h_eff = self.h_d.conj().T + h_r_H @ Theta @ self.G
        h_j_eff = self.h_ju.conj().T + h_r_H @ Theta @ self.G_j

        norm_h_eff = np.linalg.norm(h_eff)
        if norm_h_eff > 1e-12:
            w = np.sqrt(self.P_t) * (h_eff.conj().T / norm_h_eff)
        else:
            w = np.zeros((self.M, 1), dtype=complex)

        if self.jammer is None:
            p_j = 0.0
        else:
            p_j, _ = self.jammer.step(self._slot, user_active=True)
        self._slot += 1

        signal_power = np.abs(np.dot(h_eff, w)[0, 0]) ** 2
        interference_power = p_j * np.abs(h_j_eff[0, 0]) ** 2
        sinr = signal_power / (interference_power + self.sigma2)
        snr = signal_power / self.sigma2  # Jammer yokken referans SNR
        rate = np.log2(1.0 + sinr)

        reward = float(rate)

        terminated = True
        truncated = False

        db = lambda x: 10.0 * np.log10(x) if x > 0 else -np.inf
        info = {
            "rate": rate,
            "sinr": sinr,
            "sinr_db": db(sinr),
            "snr_db": db(snr),
            "inr_db": db(interference_power / self.sigma2),
            "rate_no_jam": np.log2(1.0 + snr),
            "jammer_power_dbm": watt_to_dbm(p_j),
            "jammer_active": p_j > 0.0,
        }

        # 3. DÜZELTME: step() sonunda _new_channel_state() kaldırıldı.
        # Gym standartlarında terminated=True olduğunda dönen observation, 
        # o anki (terminal) duruma ait olmalıdır. Yeni kanal reset() ile üretilir.
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


# --- ORTAM TESTİ ---
if __name__ == "__main__":
    np.random.seed(0)

    env = RISEnvironment(num_antennas=4, num_elements=16, jammer_power_dbm=30.0)
    obs, _ = env.reset(seed=0)
    print(f"Gözlem Boyutu: {obs.shape} (beklenen: {env.obs_dim})")
    print(f"Aksiyon Boyutu: {env.action_space.shape}")

    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("\nRastgele Faz, Barrage Jammer @ 30 dBm:")
    print(f"  - SINR: {info['sinr_db']:.2f} dB | Jammer'sız SNR: {info['snr_db']:.2f} dB | INR: {info['inr_db']:.2f} dB")
    print(f"  - Spektral Verimlilik: {info['rate']:.4f} bps/Hz (jammer'sız: {info['rate_no_jam']:.4f})")

    print("\nP_j taraması (ortalama rate, bps/Hz):")
    for p_dbm in (10, 20, 30, 40):
        e = RISEnvironment(jammer_power_dbm=p_dbm)
        e.reset(seed=1)
        rates = [e.step(e.action_space.sample())[4]["rate"] for _ in range(200)]
        print(f"  P_j = {p_dbm} dBm -> {np.mean(rates):.3f}")

    e = RISEnvironment(jammer_type="reactive", jammer_power_dbm=30.0,
                       jammer_kwargs={"reaction_delay": 1, "burst_len": 1, "seed": 0})
    e.reset(seed=2)
    flags = [e.step(e.action_space.sample())[4]["jammer_active"] for _ in range(100)]
    print(f"\nReaktif jammer görev döngüsü (user her slotta aktif): {np.mean(flags):.2f}")
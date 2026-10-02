import numpy as np


def dbm_to_watt(p_dbm):
    """dBm -> Watt dönüşümü."""[cite: 11]
    return 10 ** ((p_dbm - 30.0) / 10.0)[cite: 11]


def watt_to_dbm(p_w):
    """Watt -> dBm dönüşümü (0 veya negatif güç için -inf döner)."""[cite: 11]
    return -np.inf if p_w <= 0 else 10.0 * np.log10(p_w) + 30.0[cite: 11]


class BaseJammer:
    """
    Tüm Jammer (Karıştırıcı) sınıfları için ortak temel arayüz.[cite: 11]

    step() fonksiyonu (p_j_watt, s_j) döner:[cite: 11]
      - p_j_watt : Watt cinsinden anlık karıştırma iletim gücü (sessizken 0.0)[cite: 11]
      - s_j      : Karmaşık (complex) semboller, şekil: (num_symbols,), birim güçte (E|s|^2 = 1)[cite: 11]

    Gerçek iletilen sinyal: x_j = sqrt(p_j_watt) * s_j[cite: 11]
    Kullanıcının maruz kaldığı toplam efektif girişim sinyali:
        i = (h_ju + h_r^H @ Theta @ G_j) * x_j
    """

    P_MIN_DBM = 10.0[cite: 11]
    P_MAX_DBM = 40.0[cite: 11]

    def __init__(self, power_dbm=30.0, num_symbols=64, seed=None):
        self.num_symbols = num_symbols[cite: 11]
        self.rng = np.random.default_rng(seed)[cite: 11]
        self.set_power_dbm(power_dbm)[cite: 11]

    def set_power_dbm(self, power_dbm):
        """Jammer iletim gücünü dBm olarak günceller."""
        if not (self.P_MIN_DBM <= power_dbm <= self.P_MAX_DBM):
            raise ValueError(
                f"Jamming power must be within [{self.P_MIN_DBM}, {self.P_MAX_DBM}] dBm, got {power_dbm}"
            )[cite: 11]
        self.power_dbm = float(power_dbm)[cite: 11]
        self.power_w = dbm_to_watt(self.power_dbm)[cite: 11]

    def _gaussian_symbols(self):
        """Birim ortalama güçte dairesel simetrik karmaşık Gauss örnekleri CN(0, 1)."""[cite: 11]
        re = self.rng.standard_normal(self.num_symbols)[cite: 11]
        im = self.rng.standard_normal(self.num_symbols)[cite: 11]
        return (re + 1j * im) / np.sqrt(2.0)[cite: 11]

    def _silent(self):
        """Jammer sustuğunda dönecek sıfır güç ve boş semboller."""
        return 0.0, np.zeros(self.num_symbols, dtype=complex)[cite: 11]

    def step(self, slot_idx=0, **sensing):
        raise NotImplementedError[cite: 11]


class BarrageJammer(BaseJammer):
    """
    Sabit (Barrage/Geniş Bant) Karıştırıcı:
    Her zaman diliminde (time slot) sabit bir P_j gücünde [10, 40] dBm
    kesintisiz Gauss gürültüsü yayar.[cite: 11]
    """

    def step(self, slot_idx=0, **sensing):
        return self.power_w, self._gaussian_symbols()[cite: 11]


class ReactiveJammer(BaseJammer):
    """
    Akıllı Reaktif Karıştırıcı:
    Meşru iletim kanalını dinler; yalnızca iletim algıladığında hedefin
    spektral bandına yönelik darbe/patlama (burst) şeklinde gürültü yayar.[cite: 11]

    Algılama Modelleri:
      1. Enerji Dedektörü: sensed_power_w > threshold_factor * noise_power_w[cite: 11]
      2. İstatistiksel Dedektör: user_active durumuna göre p_detect ve p_false_alarm olasılıkları.[cite: 11]
    """

    def __init__(
            self,
            power_dbm=30.0,
            num_symbols=64,
            seed=None,
            band_fraction=0.25,
            band_center=0.0,
            reaction_delay=0,
            burst_len=1,
            p_detect=0.95,
            p_false_alarm=0.02,
            threshold_factor=2.0,
    ):
        super().__init__(power_dbm, num_symbols, seed)[cite: 11]
        if not (0.0 < band_fraction <= 1.0):
            raise ValueError("band_fraction must be in (0, 1]")[cite: 11]
        self.band_fraction = band_fraction[cite: 11]
        self.band_center = band_center[cite: 11]
        self.reaction_delay = int(reaction_delay)[cite: 11]
        self.burst_len = int(burst_len)[cite: 11]
        self.p_detect = p_detect[cite: 11]
        self.p_false_alarm = p_false_alarm[cite: 11]
        self.threshold_factor = threshold_factor[cite: 11]

        self._pending = {}  # slot_idx -> tetiklenmiş bekleyen burst sayısı[cite: 11]
        self._burst_remaining = 0[cite: 11]
        self.slots_active = 0[cite: 11]
        self.slots_total = 0[cite: 11]

    # ---- Kanal Algılama / Sensing ----
    def _detect(self, user_active, sensed_power_w, noise_power_w):
        if sensed_power_w is not None:
            return sensed_power_w > self.threshold_factor * noise_power_w[cite: 11]
        if user_active is None:
            raise ValueError("Provide either sensed_power_w or user_active.")[cite: 11]
        p = self.p_detect if user_active else self.p_false_alarm[cite: 11]
        return self.rng.random() < p[cite: 11]

    # ---- Hedefe Yönelik Bant Sınırlı Dalga Şekli ----
    def _band_limited_symbols(self):
        n = self.num_symbols[cite: 11]
        spectrum = np.fft.fftshift(np.fft.fft(self._gaussian_symbols()))[cite: 11]
        freqs = np.fft.fftshift(np.fft.fftfreq(n))[cite: 11]
        half = self.band_fraction / 2.0[cite: 11]
        mask = np.abs(freqs - self.band_center) <= half[cite: 11]
        if not mask.any():
            mask[np.argmin(np.abs(freqs - self.band_center))] = True[cite: 11]
        s = np.fft.ifft(np.fft.ifftshift(spectrum * mask))[cite: 11]
        # Enerjiyi birim ortalama güce geri ölçekle
        return s / np.sqrt(np.mean(np.abs(s) ** 2))[cite: 11]

    def step(self, slot_idx=0, user_active=None, sensed_power_w=None, noise_power_w=1e-12, **_):
        self.slots_total += 1[cite: 11]

        # 1. Mevcut slotta meşru linki dinle
        if self._detect(user_active, sensed_power_w, noise_power_w):
            start = slot_idx + self.reaction_delay[cite: 11]
            self._pending[start] = self._pending.get(start, 0) + 1[cite: 11]

        # 2. Bu slot için zamanlanmış bir saldırı darbesi var mı?
        if self._pending.pop(slot_idx, 0):
            self._burst_remaining = max(self._burst_remaining, self.burst_len)[cite: 11]

        # 3. Sinyal bas ya da sessiz kal
        if self._burst_remaining > 0:
            self._burst_remaining -= 1[cite: 11]
            self.slots_active += 1[cite: 11]
            return self.power_w, self._band_limited_symbols()[cite: 11]
        return self._silent()[cite: 11]

    @property
    def duty_cycle(self):
        """Jammer'ın aktif olduğu slot oranı (Enerji verimliliği metriği)."""[cite: 11]
        return self.slots_active / max(self.slots_total, 1)[cite: 11]


# --- DOĞRULAMA VE TEST ---
if __name__ == "__main__":
    barrage = BarrageJammer(power_dbm=40.0, num_symbols=256, seed=0)[cite: 11]
    p, s = barrage.step(0)[cite: 11]
    print(f"Barrage : P_j = {watt_to_dbm(p):.1f} dBm | mean |s|^2 = {np.mean(np.abs(s) ** 2):.3f}")[cite: 11]

    reactive = ReactiveJammer(power_dbm=30.0, num_symbols=256, seed=1,
                              band_fraction=0.25, reaction_delay=1, burst_len=2)[cite: 11]
    user_pattern = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0][cite: 11]
    for t, active in enumerate(user_pattern):
        p, s = reactive.step(t, user_active=bool(active))[cite: 11]
        print(f"Slot {t:02d}: Legitimate Tx={active} | Jammer Output={watt_to_dbm(p):.1f} dBm | Active={p > 0}")
    print(f"Reactive Jammer Duty Cycle: {reactive.duty_cycle:.2%}")
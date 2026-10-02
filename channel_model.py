import numpy as np


class PathLossModel:
    """
    3D koordinatlara dayalı mesafe ve Path Loss (Yol Kaybı) hesaplama sınıfı.
    """

    def __init__(self, pl_exponent_direct=3.5, pl_exponent_ris=2.2, c_0_db=-30.0):
        # c_0_db: 1 metredeki referans yol kaybı (dB cinsinden)[cite: 1, 8]
        self.c_0 = 10 ** (c_0_db / 10.0)
        self.alpha_d = pl_exponent_direct  # Doğrudan yol (NLoS, yüksek kayıp)[cite: 1, 8]
        self.alpha_r = pl_exponent_ris  # RIS yolları (LoS baskın, düşük kayıp)[cite: 1, 8]

    def calculate_distance(self, pos1, pos2):
        """İki 3D nokta arasındaki Öklid mesafesini hesaplar."""[cite: 1]
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))

    def get_path_loss(self, distance, alpha):
        """Mesafe ve sönümleme katsayısına göre doğrusal kazancı döner."""[cite: 1]
        return np.sqrt(self.c_0 * (distance ** (-alpha)))


class WirelessChannel:
    """
    RIS Destekli MISO Sistemi için Gelişmiş Kanal Simülatörü.
    - Meşru link : BS (M anten) -> RIS (N eleman) -> User (1 anten)
    - Saldırı    : Jammer (1 anten) -> User ve Jammer -> RIS
    - Uzamsal dizi yanıtı (ULA steering vector) ile gerçekçi Rician LoS modelleme.
    """

    def __init__(
            self,
            num_antennas=4,  # M: Baz istasyonu anten sayısı[cite: 1]
            num_elements=16,  # N: RIS eleman sayısı[cite: 1]
            bs_pos=(0.0, 0.0, 10.0),  # BS Koordinatı (x, y, z)[cite: 1]
            ris_pos=(50.0, 10.0, 10.0),  # RIS Koordinatı[cite: 1]
            user_pos=(60.0, 0.0, 1.5),  # Kullanıcı Koordinatı[cite: 1]
            jammer_pos=(55.0, -15.0, 1.5),  # Jammer Koordinatı
            k_factor_bs_ris=10.0,  # Kule-Bina arası güçlü LoS (10 dB civarı)
            k_factor_ris_user=3.0,  # RIS-Kullanıcı arası LoS[cite: 1]
            k_factor_jammer_ris=2.0,  # Yer seviyesindeki Jammer - Yüksek RIS arası orta LoS
            pl_exponent_jammer_direct=None
    ):
        self.M = num_antennas
        self.N = num_elements
        self.bs_pos = np.array(bs_pos, dtype=float)
        self.ris_pos = np.array(ris_pos, dtype=float)
        self.user_pos = np.array(user_pos, dtype=float)
        self.jammer_pos = np.array(jammer_pos, dtype=float)

        # Rician K faktörleri
        self.K_bs_ris = k_factor_bs_ris
        self.K_ris_user = k_factor_ris_user
        self.K_j_ris = k_factor_jammer_ris

        self.path_loss_engine = PathLossModel()

        # Jammer -> User yol kaybı üssü (varsayılan NLoS doğrudan yol)
        self.alpha_jd = (self.path_loss_engine.alpha_d
                         if pl_exponent_jammer_direct is None
                         else pl_exponent_jammer_direct)

    def _generate_rayleigh(self, shape):
        """Rayleigh (Tamamen NLoS) küçük ölçekli sönümleme matrisi üretir (CN(0, 1))."""[cite: 1]
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        return (real + 1j * imag) / np.sqrt(2.0)

    def _steering_vector(self, num_elements, angle_rad):
        """
        Düzgün Doğrusal Dizi (ULA) için yarı dalga boyu aralıklı (d = lambda/2)
        yönlendirme/dizi yanıt vektörü (Array Response Vector).
        """
        indices = np.arange(num_elements)
        return np.exp(-1j * np.pi * indices * np.sin(angle_rad)).reshape(-1, 1)

    def _calculate_azimuth(self, tx_pos, rx_pos):
        """İki 3D nokta arasındaki x-y düzlemi geliş/ayrılış yatay açısını (azimuth) hesaplar."""
        delta = rx_pos - tx_pos
        return float(np.arctan2(delta[1], delta[0]))

    def _generate_rician_channel(self, shape, k_factor, h_los):
        """Fiziksel LoS dizi yanıtı ve NLoS dağınık bileşeni birleştiren Rician kanal üretici."""[cite: 1]
        h_nlos = self._generate_rayleigh(shape)[cite: 1]
        weight_los = np.sqrt(k_factor / (k_factor + 1.0))[cite: 1, 8]
        weight_nlos = np.sqrt(1.0 / (k_factor + 1.0))[cite: 1, 8]
        return weight_los * h_los + weight_nlos * h_nlos

    def get_channel_realization(self):
        """
        Anlık kanal matrislerini üretir.

        Dönen Matrisler:
        - h_d  : BS -> User doğrudan kanal                [M x 1][cite: 1, 6]
        - G    : BS -> RIS kanalı                         [N x M][cite: 1, 6]
        - h_r  : RIS -> User kanalı                       [N x 1][cite: 1, 6]
        - h_ju : Jammer -> User doğrudan girişim kanalı   [1 x 1]
        - G_j  : Jammer -> RIS çok yollu girişim kanalı   [N x 1]
        """
        pl = self.path_loss_engine

        # 1. 3D Mesafeleri Hesapla
        d_bs_user = pl.calculate_distance(self.bs_pos, self.user_pos)[cite: 1]
        d_bs_ris = pl.calculate_distance(self.bs_pos, self.ris_pos)[cite: 1]
        d_ris_user = pl.calculate_distance(self.ris_pos, self.user_pos)[cite: 1]
        d_j_user = pl.calculate_distance(self.jammer_pos, self.user_pos)
        d_j_ris = pl.calculate_distance(self.jammer_pos, self.ris_pos)

        # 2. Path Loss Katsayılarını Hesapla
        pl_d = pl.get_path_loss(d_bs_user, pl.alpha_d)[cite: 1]
        pl_G = pl.get_path_loss(d_bs_ris, pl.alpha_r)[cite: 1]
        pl_hr = pl.get_path_loss(d_ris_user, pl.alpha_r)[cite: 1]
        pl_ju = pl.get_path_loss(d_j_user, self.alpha_jd)
        pl_Gj = pl.get_path_loss(d_j_ris, pl.alpha_r)

        # 3. LoS Bileşenleri İçin Dizi Yanıtlarını (Steering Vectors) Hesapla
        # BS -> RIS: BS Ayrılış Açısı (AoD) ve RIS Geliş Açısı (AoA)
        angle_bs_to_ris = self._calculate_azimuth(self.bs_pos, self.ris_pos)
        angle_ris_from_bs = self._calculate_azimuth(self.ris_pos, self.bs_pos)
        a_bs = self._steering_vector(self.M, angle_bs_to_ris)
        a_ris_bs = self._steering_vector(self.N, angle_ris_from_bs)
        h_los_G = a_ris_bs @ a_bs.conj().T  # [N x M] LoS kanal matrisi

        # RIS -> User: RIS Ayrılış Açısı (AoD)
        angle_ris_to_user = self._calculate_azimuth(self.ris_pos, self.user_pos)
        h_los_hr = self._steering_vector(self.N, angle_ris_to_user)  # [N x 1]

        # Jammer -> RIS: Jammer'dan RIS'e Geliş Açısı (AoA)
        angle_ris_from_j = self._calculate_azimuth(self.ris_pos, self.jammer_pos)
        h_los_Gj = self._steering_vector(self.N, angle_ris_from_j)  # [N x 1]

        # 4. Küçük Ölçekli Sönümleme ve Path Loss Birleşimi
        # h_d: Doğrudan yol (NLoS -> Rayleigh) [M x 1]
        h_d = pl_d * self._generate_rayleigh((self.M, 1))[cite: 1]

        # G: BS -> RIS (Güçlü LoS -> Rician) [N x M]
        G = pl_G * self._generate_rician_channel((self.N, self.M), self.K_bs_ris, h_los_G)

        # h_r: RIS -> User (LoS -> Rician) [N x 1]
        h_r = pl_hr * self._generate_rician_channel((self.N, 1), self.K_ris_user, h_los_hr)

        # h_ju: Jammer -> User doğrudan yolu (NLoS -> Rayleigh) [1 x 1]
        h_ju = pl_ju * self._generate_rayleigh((1, 1))

        # G_j: Jammer -> RIS (LoS -> Rician) [N x 1]
        G_j = pl_Gj * self._generate_rician_channel((self.N, 1), self.K_j_ris, h_los_Gj)

        return h_d, G, h_r, h_ju, G_j


if __name__ == "__main__":
    M, N = 4, 32
    channel_sim = WirelessChannel(num_antennas=M, num_elements=N)[cite: 1]
    h_d, G, h_r, h_ju, G_j = channel_sim.get_channel_realization()

    print("--- Doğrulanmış Kanal Matrisleri ve Güç Seviyeleri ---")
    print(f"h_d  (BS -> User)     : Boyut {h_d.shape}  | Ortalama Güç: {np.mean(np.abs(h_d) ** 2):.2e}")[cite: 1]
    print(f"G    (BS -> RIS)      : Boyut {G.shape} | Ortalama Güç: {np.mean(np.abs(G) ** 2):.2e}")[cite: 1]
    print(f"h_r  (RIS -> User)    : Boyut {h_r.shape} | Ortalama Güç: {np.mean(np.abs(h_r) ** 2):.2e}")[cite: 1]
    print(f"h_ju (Jammer -> User) : Boyut {h_ju.shape}   | Ortalama Güç: {np.mean(np.abs(h_ju) ** 2):.2e}")
    print(f"G_j  (Jammer -> RIS)  : Boyut {G_j.shape} | Ortalama Güç: {np.mean(np.abs(G_j) ** 2):.2e}")
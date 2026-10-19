"""
Noise models for synthetic InSAR data.

This module provides noise models:
- Uncorrelated Gaussian noise
- Spatially correlated power-law noise (atmospheric turbulence)
- Orbital/baseline ramp errors
- Decorrelation (coherence-loss) phase noise: vegetation, snow, surface rupture
"""

import numpy as np
from typing import Tuple, Optional, Dict


def generate_random_noise(
    shape: Tuple[int, int],
    amplitude_m: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate simple random Gaussian noise.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    amplitude_m : float
        Standard deviation of noise in meters (default: 5 mm)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noise : np.ndarray
        Random noise field in meters
    """
    if seed is not None:
        np.random.seed(seed)

    return amplitude_m * np.random.randn(*shape)


def generate_correlated_noise(
    shape: Tuple[int, int],
    amplitude_m: float = 0.005,
    beta: float = 5 / 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate spatially correlated noise with a power-law spectrum.

    Models atmospheric turbulence in InSAR interferograms. The power spectral
    density follows P(f) ∝ f^(-β).  Kolmogorov atmospheric turbulence uses
    β = 5/3; β = 0 recovers uncorrelated white noise.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    amplitude_m : float
        Standard deviation of the output noise field in meters (default: 5 mm)
    beta : float
        Power-law spectral index (default: 5/3 ≈ Kolmogorov turbulence).
        Higher values produce longer-range spatial correlations.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noise : np.ndarray
        Spatially correlated, zero-mean noise field in meters
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq = np.sqrt(FX**2 + FY**2)
    freq[0, 0] = 1.0          # avoid divide-by-zero at DC

    power = freq ** (-beta / 2)
    power[0, 0] = 0.0          # zero mean: remove DC component

    white = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    field = np.real(np.fft.ifft2(power * white))

    std = field.std()
    if std > 0:
        field /= std

    return field * amplitude_m


def generate_orbital_ramp(
    shape: Tuple[int, int],
    ramp_east_m_per_km: float = 0.0,
    ramp_north_m_per_km: float = 0.0,
    offset_m: float = 0.0,
    pixel_size_km: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate orbital/baseline error ramp.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape
    ramp_east_m_per_km : float
        East-West gradient in meters per km
    ramp_north_m_per_km : float
        North-South gradient in meters per km
    offset_m : float
        Constant offset in meters
    pixel_size_km : float
        Pixel size in km
    seed : int, optional
        Random seed (for random ramp if gradients not specified)

    Returns
    -------
    ramp : np.ndarray
        Orbital ramp in meters
    """
    ny, nx = shape

    if ramp_east_m_per_km == 0 and ramp_north_m_per_km == 0 and seed is not None:
        np.random.seed(seed)
        ramp_east_m_per_km = np.random.uniform(-0.0005, 0.0005)
        ramp_north_m_per_km = np.random.uniform(-0.0005, 0.0005)

    x_km = np.arange(nx) * pixel_size_km - (nx - 1) * pixel_size_km / 2
    y_km = np.arange(ny) * pixel_size_km - (ny - 1) * pixel_size_km / 2
    X_km, Y_km = np.meshgrid(x_km, y_km)

    ramp = offset_m + ramp_east_m_per_km * X_km + ramp_north_m_per_km * Y_km

    return ramp


def _patch_weight(
    shape: Tuple[int, int],
    fraction: float,
    beta: float,
    edge_softness: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Build a smooth [0, 1] weight map covering roughly ``fraction`` of the grid.

    A spatially correlated power-law field is thresholded at its
    (1 - fraction) quantile and passed through a logistic function so patch
    boundaries are gradual rather than hard-edged.
    """
    if fraction <= 0.0:
        return np.zeros(shape)
    if fraction >= 1.0:
        return np.ones(shape)

    field = generate_correlated_noise(shape, amplitude_m=1.0, beta=beta, seed=seed)
    threshold = np.quantile(field, 1.0 - fraction)
    if edge_softness <= 0.0:
        return (field >= threshold).astype(float)
    z = np.clip((field - threshold) / edge_softness, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _rupture_weight(
    shape: Tuple[int, int],
    pixel_size_km: float,
    strike_deg: float,
    length_km: float,
    width_km: float,
    center_km: Tuple[float, float],
) -> np.ndarray:
    """
    Gaussian [0, 1] weight around a straight surface-rupture segment.

    Uses the distance from each pixel to the finite segment, so the
    decorrelated zone has rounded ends. ``width_km`` is the full width at
    which the weight falls to exp(-2) ≈ 0.14 (i.e. sigma = width / 4).
    """
    ny, nx = shape
    x_km = np.arange(nx) * pixel_size_km - (nx - 1) * pixel_size_km / 2
    y_km = np.arange(ny) * pixel_size_km - (ny - 1) * pixel_size_km / 2
    X_km, Y_km = np.meshgrid(x_km, y_km)

    # Strike is clockwise from North -> unit vector (east, north)
    s = np.deg2rad(strike_deg)
    ue, un = np.sin(s), np.cos(s)

    dx = X_km - center_km[0]
    dy = Y_km - center_km[1]
    along = dx * ue + dy * un
    across = -dx * un + dy * ue

    half_len = length_km / 2.0
    along_excess = np.clip(np.abs(along) - half_len, 0.0, None)
    dist = np.sqrt(along_excess**2 + across**2)

    sigma = max(width_km / 4.0, 1e-9)
    return np.exp(-0.5 * (dist / sigma) ** 2)


def generate_decorrelation_noise(
    shape: Tuple[int, int],
    wavelength_m: float = 0.05546,
    pixel_size_km: float = 0.5,
    n_looks: int = 4,
    background_coherence: float = 0.9,
    # --- Vegetation (temporal decorrelation) ---
    vegetation_fraction: float = 0.0,
    vegetation_mask: Optional[np.ndarray] = None,
    temporal_baseline_days: float = 12.0,
    vegetation_tau_days: float = 20.0,
    vegetation_coherence_long_term: float = 0.1,
    vegetation_patch_beta: float = 3.0,
    # --- Snow ---
    snow_fraction: float = 0.0,
    snow_mask: Optional[np.ndarray] = None,
    snow_coherence: float = 0.05,
    snow_patch_beta: float = 4.0,
    # --- Surface rupture ---
    rupture_length_km: float = 0.0,
    rupture_width_km: float = 2.0,
    rupture_strike_deg: float = 0.0,
    rupture_center_km: Tuple[float, float] = (0.0, 0.0),
    rupture_coherence: float = 0.0,
    # --- Misc ---
    edge_softness: float = 0.15,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate decorrelation phase noise from a synthetic coherence map.

    A coherence map gamma(x, y) is built from a uniform background value
    and up to three decorrelation sources, which combine multiplicatively
    (as independent decorrelation terms do, e.g. Zebker & Villasenor 1992):

        gamma = gamma_bg * f_veg * f_snow * f_rupture,
        f_k   = 1 - w_k * (1 - gamma_k),

    where w_k in [0, 1] is the spatial weight of source k.

    Phase noise is then drawn *exactly* (not from a Gaussian approximation)
    by simulating ``n_looks`` pairs of circular complex Gaussian SLC samples
    with correlation gamma and taking the phase of their multilooked
    interferogram. This reproduces the true heavy-tailed multilook phase
    PDF (Lee et al. 1994): near-zero noise at gamma -> 1 and uniformly
    distributed phase on [-pi, pi] at gamma = 0.

    Sources
    -------
    Vegetation
        Temporal decorrelation following an exponential model
        (Rocca 2007; Parizzi et al. 2009):
        gamma_veg = gamma_inf + (1 - gamma_inf) * exp(-dt / tau).
        Patches are generated from a correlated random field covering
        ``vegetation_fraction`` of the scene, or taken from ``vegetation_mask``.
    Snow
        Snow cover change between acquisitions decorrelates almost
        completely; modelled as large patches with ``snow_coherence``.
        Pass ``snow_mask`` (e.g. a DEM-derived elevation mask) to control
        exactly where snow is.
    Surface rupture
        A band of near-total coherence loss along a straight fault trace,
        representing surface breaks, damage zone and high strain gradients
        near the rupture.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Output array shape.
    wavelength_m : float
        Radar wavelength in meters, used to express the phase noise as an
        equivalent LOS displacement (default: Sentinel-1, 5.546 cm).
    pixel_size_km : float
        Pixel size in km (used for the rupture geometry).
    n_looks : int
        Number of independent looks in the multilooked interferogram.
        More looks -> less phase noise at a given coherence.
    background_coherence : float
        Coherence of undisturbed terrain, in [0, 1].
    vegetation_fraction : float
        Approximate fraction of the scene covered by vegetation, in [0, 1].
        Ignored if ``vegetation_mask`` is given.
    vegetation_mask : np.ndarray, optional
        User-supplied vegetation weight map of shape ``shape``, values in [0, 1].
    temporal_baseline_days : float
        Time between the two acquisitions in days.
    vegetation_tau_days : float
        Decorrelation time constant of vegetated surfaces in days.
    vegetation_coherence_long_term : float
        Long-term (asymptotic) coherence of vegetation, gamma_inf.
    vegetation_patch_beta : float
        Spectral index of the field used to place vegetation patches
        (higher -> larger, smoother patches).
    snow_fraction : float
        Approximate fraction of the scene with snow change, in [0, 1].
        Ignored if ``snow_mask`` is given.
    snow_mask : np.ndarray, optional
        User-supplied snow weight map of shape ``shape``, values in [0, 1].
    snow_coherence : float
        Coherence inside snow-covered areas.
    snow_patch_beta : float
        Spectral index of the field used to place snow patches.
    rupture_length_km : float
        Length of the surface rupture in km. 0 disables rupture decorrelation.
    rupture_width_km : float
        Full width of the decorrelated zone across the trace in km.
    rupture_strike_deg : float
        Strike of the rupture trace, clockwise from North.
    rupture_center_km : tuple (east_km, north_km)
        Rupture midpoint relative to the grid centre (matches the
        ``xcen_km``/``ycen_km`` convention of the generators).
    rupture_coherence : float
        Coherence on the rupture trace itself.
    edge_softness : float
        Softness of random patch boundaries in units of the underlying
        field's standard deviation. 0 gives hard-edged patches.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        - ``phase_noise_rad``: wrapped decorrelation phase noise in [-pi, pi]
        - ``noise_m``: the same noise as equivalent LOS displacement (m),
          using phase = -4*pi*d / wavelength. Note this is a wrapped quantity,
          bounded by +/- wavelength / 4.
        - ``coherence``: final coherence map in [0, 1]
        - ``vegetation_weight``, ``snow_weight``, ``rupture_weight``:
          per-source weight maps in [0, 1] (useful as ML labels)

    Examples
    --------
    >>> res = generate_decorrelation_noise(
    ...     (256, 256), vegetation_fraction=0.3, temporal_baseline_days=24,
    ...     rupture_length_km=30, rupture_strike_deg=45, seed=0)
    >>> phase_noisy = wrap_phase(phase_noisy + res["phase_noise_rad"])
    """
    ny, nx = shape
    if n_looks < 1:
        raise ValueError(f"n_looks={n_looks} must be >= 1.")
    for name, val in [
        ("background_coherence", background_coherence),
        ("vegetation_coherence_long_term", vegetation_coherence_long_term),
        ("snow_coherence", snow_coherence),
        ("rupture_coherence", rupture_coherence),
        ("vegetation_fraction", vegetation_fraction),
        ("snow_fraction", snow_fraction),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name}={val} is outside the valid range [0, 1].")
    if temporal_baseline_days < 0 or vegetation_tau_days <= 0:
        raise ValueError(
            "temporal_baseline_days must be >= 0 and vegetation_tau_days > 0."
        )

    def _check_mask(mask, name):
        mask = np.asarray(mask, dtype=float)
        if mask.shape != (ny, nx):
            raise ValueError(f"{name} shape {mask.shape} does not match {shape}.")
        return np.clip(mask, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    veg_seed, snow_seed = (int(v) for v in rng.integers(0, 2**31 - 1, size=2))

    # --- Source weight maps ---
    if vegetation_mask is not None:
        w_veg = _check_mask(vegetation_mask, "vegetation_mask")
    else:
        w_veg = _patch_weight(shape, vegetation_fraction, vegetation_patch_beta,
                              edge_softness, veg_seed)

    if snow_mask is not None:
        w_snow = _check_mask(snow_mask, "snow_mask")
    else:
        w_snow = _patch_weight(shape, snow_fraction, snow_patch_beta,
                               edge_softness, snow_seed)

    if rupture_length_km > 0.0:
        w_rup = _rupture_weight(shape, pixel_size_km, rupture_strike_deg,
                                rupture_length_km, rupture_width_km,
                                rupture_center_km)
    else:
        w_rup = np.zeros(shape)

    # --- Coherence map ---
    gamma_veg = vegetation_coherence_long_term + (
        1.0 - vegetation_coherence_long_term
    ) * np.exp(-temporal_baseline_days / vegetation_tau_days)

    coherence = (
        background_coherence
        * (1.0 - w_veg * (1.0 - gamma_veg))
        * (1.0 - w_snow * (1.0 - snow_coherence))
        * (1.0 - w_rup * (1.0 - rupture_coherence))
    )
    coherence = np.clip(coherence, 0.0, 1.0)

    # --- Exact multilook phase simulation ---
    # s1 ~ CN(0,1); s2 = gamma*s1 + sqrt(1-gamma^2)*n  =>  corr(s1, s2) = gamma
    scale = np.sqrt(1.0 - coherence**2)
    interferogram = np.zeros(shape, dtype=complex)
    for _ in range(int(n_looks)):
        s1 = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        n = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        s2 = coherence * s1 + scale * n
        interferogram += s1 * np.conj(s2)

    phase_noise_rad = np.angle(interferogram)
    noise_m = -phase_noise_rad * wavelength_m / (4.0 * np.pi)

    return {
        "phase_noise_rad": phase_noise_rad,
        "noise_m": noise_m,
        "coherence": coherence,
        "vegetation_weight": w_veg,
        "snow_weight": w_snow,
        "rupture_weight": w_rup,
    }

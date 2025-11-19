from __future__ import annotations

from typing import Literal
import numpy as np
import scipy.signal
import scipy.interpolate

CollapseMethod = Literal["ls", "fft", "gaussian", "savgol"]
InterpolationMethod = Literal["linear", "polar", "pchip", "lanczos"]

# -----------------------------------------------------------------------------
# Collapse / Mean Estimation
# -----------------------------------------------------------------------------

def estimate_mean_ls(
    omegas: np.ndarray,
    phi: np.ndarray,
    m: int = 4,
) -> float:
    """
    Estimate E[G] from the characteristic function φ(ω) via a local quadratic fit
    around ω = 0:
      φ(ω) ≈ a0 + a1 ω + a2 ω^2
    Then:
      φ'(0) = a1   and   E[G] = Im(φ'(0))
    """
    K = len(omegas)
    idx0 = int(np.argmin(np.abs(omegas)))  # index closest to 0

    start = max(0, idx0 - m)
    end = min(K, idx0 + m + 1)

    w = omegas[start:end]
    y = phi[start:end]

    if len(w) < 3:
        return 0.0

    A = np.stack([np.ones_like(w), w, w ** 2], axis=1)

    # Fit real and imaginary parts separately
    coef_real, *_ = np.linalg.lstsq(A, y.real, rcond=None)
    coef_imag, *_ = np.linalg.lstsq(A, y.imag, rcond=None)

    # φ'(0) = a1
    a1 = coef_real[1] + 1j * coef_imag[1]
    mean_est = a1.imag

    return float(mean_est)


def estimate_mean_fft(
    omegas: np.ndarray,
    phi: np.ndarray,
) -> float:
    """
    Estimate E[G] by inverting the CF via FFT to get the PDF/PMF,
    then computing the expected value in the spatial domain.
    
    Assumes `omegas` is a symmetric uniform grid centered at 0 (or close to it).
    """
    # Check uniformity roughly
    d_omega = omegas[1] - omegas[0]
    
    # We need to be careful with the ordering for standard FFT.
    # np.fft.ifft expects [0, 1, ..., N/2, -N/2, ..., -1] * dw
    # Our `omegas` are usually sorted [-W, ..., W].
    # We use ifftshift to swap halves before calling ifft.
    
    phi_shifted = np.fft.ifftshift(phi)
    
    # Inverse FFT to get spatial distribution
    # pdf (complex) -> real part is the density
    pdf = np.fft.ifft(phi_shifted)
    pdf_real = pdf.real
    
    # Normalize just in case (though if phi(0)=1 it should sum to 1/dx or similar depending on scaling)
    # With discrete DFT, sum(pdf) should be 1 if phi(0)=1.
    # Let's check normalization sum
    norm = np.sum(pdf_real)
    if abs(norm) > 1e-9:
        pdf_real /= norm
    
    # Construct corresponding x-axis (returns)
    K = len(omegas)
    # xs = np.fft.fftfreq(K, d=d_omega / (2 * np.pi))
    # Usually fftfreq gives [0, 1, ..., -1], need to match ifft output
    # The ifft result corresponds to time/space domain
    
    xs = np.fft.fftfreq(K, d=d_omega / (2 * np.pi))
    
    # Compute mean
    mean_est = np.sum(xs * pdf_real)
    
    return float(mean_est)


def estimate_mean_gaussian(
    omegas: np.ndarray,
    phi: np.ndarray,
    max_w: float = 2.0,
) -> float:
    """
    Estimate E[G] by fitting a Gaussian log-CF form:
       log φ(ω) ≈ i μ ω - 0.5 σ^2 ω^2
    We fit a line to the unwrapped phase arg(φ) ≈ μ ω.
    """
    # Select range [-max_w, max_w] where signal is strong
    mask = np.abs(omegas) <= max_w
    w_sub = omegas[mask]
    phi_sub = phi[mask]
    
    if len(w_sub) < 2:
        # Fallback to all
        w_sub = omegas
        phi_sub = phi
        
    # Unwrapped phase
    phase = np.unwrap(np.angle(phi_sub))
    
    # Fit line through origin: phase = mu * w
    # simple regression: mu = sum(w * phase) / sum(w^2)
    # Or lstsq with bias? Theoretically phase(0)=0.
    
    num = np.sum(w_sub * phase)
    den = np.sum(w_sub ** 2)
    
    if den < 1e-12:
        return 0.0
        
    mu = num / den
    return float(mu)


def estimate_mean_savgol(
    omegas: np.ndarray,
    phi: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> float:
    """
    Estimate E[G] = Im(φ'(0)) using a Savitzky-Golay filter to smooth
    and differentiate the imaginary part of φ.
    """
    # Ensure window_length is odd and <= len(omegas)
    if window_length > len(omegas):
        window_length = len(omegas)
        if window_length % 2 == 0:
            window_length -= 1
            
    if window_length < polyorder + 2:
        # Fallback to LS if not enough points
        return estimate_mean_ls(omegas, phi)
        
    d_omega = omegas[1] - omegas[0] # assume uniform roughly for scale
    
    # Calculate 1st derivative of imaginary part
    # deriv=1
    imag_deriv = scipy.signal.savgol_filter(
        phi.imag, 
        window_length=window_length, 
        polyorder=polyorder, 
        deriv=1, 
        delta=d_omega,
        mode='interp'
    )
    
    # Extract value at 0
    idx0 = int(np.argmin(np.abs(omegas)))
    mean_est = imag_deriv[idx0]
    
    return float(mean_est)


# -----------------------------------------------------------------------------
# Interpolation Methods
# -----------------------------------------------------------------------------

def interpolate_linear(
    target_omegas: np.ndarray,
    grid_omegas: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Standard linear interpolation in the complex plane (Cartesian).
    """
    # Real part
    real_interp = np.interp(
        target_omegas, 
        grid_omegas, 
        phi.real,
        left=phi.real[0], 
        right=phi.real[-1]
    )
    # Imag part
    imag_interp = np.interp(
        target_omegas, 
        grid_omegas, 
        phi.imag,
        left=phi.imag[0], 
        right=phi.imag[-1]
    )
    return real_interp + 1j * imag_interp


def interpolate_polar(
    target_omegas: np.ndarray,
    grid_omegas: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Interpolate Magnitude and Phase separately.
    Preserves unitarity |phi| better than Cartesian linear interp.
    """
    mag = np.abs(phi)
    phase = np.unwrap(np.angle(phi))
    
    mag_interp = np.interp(
        target_omegas, 
        grid_omegas, 
        mag,
        left=mag[0], 
        right=mag[-1]
    )
    
    phase_interp = np.interp(
        target_omegas, 
        grid_omegas, 
        phase,
        left=phase[0], 
        right=phase[-1]
    )
    
    return mag_interp * np.exp(1j * phase_interp)


def interpolate_pchip(
    target_omegas: np.ndarray,
    grid_omegas: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
    Monotonicity preserving - avoids overshoots common in cubic splines.
    Interpolates Real and Imag parts separately.
    """
    # PchipInterpolator expects sorted x. Our omegas are sorted.
    
    pchip_real = scipy.interpolate.PchipInterpolator(grid_omegas, phi.real)
    pchip_imag = scipy.interpolate.PchipInterpolator(grid_omegas, phi.imag)
    
    # Extrapolate=False by default (nan), but we can handle edges manually or let it extrapolate?
    # Pchip generally extrapolates monotonically. 
    # However, let's clamp for safety similar to np.interp if needed.
    # Actually Pchip extrapolates using the end polynomials.
    
    real_interp = pchip_real(target_omegas)
    imag_interp = pchip_imag(target_omegas)
    
    return real_interp + 1j * imag_interp


def interpolate_lanczos(
    target_omegas: np.ndarray,
    grid_omegas: np.ndarray,
    phi: np.ndarray,
    a: int = 3
) -> np.ndarray:
    """
    Lanczos resampling (windowed sinc interpolation).
    Best for uniform grids to preserve spectral properties.
    
    Kernel L(x) = sinc(x) * sinc(x/a) if -a < x < a, else 0.
    """
    # Assume uniform grid
    dx = grid_omegas[1] - grid_omegas[0]
    
    # For each target point x_new, we find source points x_i within window [x_new - a*dx, x_new + a*dx]
    # This can be vectorized but slow in pure python for large arrays.
    
    # Simplified implementation:
    # Map target omegas to fractional indices
    indices = (target_omegas - grid_omegas[0]) / dx
    
    K = len(grid_omegas)
    output = np.zeros_like(target_omegas, dtype=complex)
    
    # Vectorized Lanczos sum
    # Iterate over window [-a+1, a] roughly
    # Optimization: We only loop over the 'a' neighbors
    
    # To do this efficiently:
    # x is the array of target fractional indices
    # For each integer shift i in floor(x)-a+1 to floor(x)+a:
    #   weight = LanczosKernel(x - i)
    #   sum += weight * phi[i]
    
    # We need floor(indices)
    idx_floor = np.floor(indices).astype(int)
    
    for i in range(-a + 1, a + 1):
        # Current neighbor index
        neighbor_idx = idx_floor + i
        
        # Valid neighbors only
        mask = (neighbor_idx >= 0) & (neighbor_idx < K)
        
        if not np.any(mask):
            continue
            
        # Calculate distance d = indices - neighbor_idx (normalized by dx already)
        d = indices[mask] - neighbor_idx[mask]
        
        # Lanczos kernel
        # L(x) = sinc(x) * sinc(x/a)
        # numpy sinc is sin(pi x)/(pi x)
        weights = np.sinc(d) * np.sinc(d / a)
        
        output[mask] += weights * phi[neighbor_idx[mask]]
        
    return output

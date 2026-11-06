"""Map a PEC y=0 half grid back to the full co-located E/H detector aperture."""
import numpy as np


# Ex, Ez, Hy are odd; Ey, Hx, Hz are even. FDTDX co-locates all six at integer y.
PARITY = np.array([-1, 1, -1, 1, -1, 1])


def positive_count(start, count, middle):
    """Include the extra positive sample mapping to the outermost negative sample."""
    if count <= 0:
        raise ValueError("Detector width must be positive.")
    return 1 + max(abs(start - middle), abs(start + count - 1 - middle))


def full_plane(fields, yaxis, caxis, start, count, middle):
    """Restore negative-y samples, original aperture ordering, and the plane at y=0."""
    positive = np.moveaxis(fields, (caxis, yaxis), (0, 1)).copy()
    if positive.shape[:2] != (6, positive_count(start, count, middle)):
        raise ValueError("Half-domain detector has the wrong E/H vector or aperture extent.")
    # FDTDX 0.6.2 co-locates these components using the preceding Yee sample. Its
    # PEC padding returns zero for that exterior sample. The real full-domain
    # neighbor is equal to the positive-side sample, so restore its missing half.
    positive[[1, 3, 5], 0] *= 2
    offsets = np.arange(start, start + count) - middle
    signs = np.where(offsets[None, :] < 0, PARITY[:, None], 1)
    signs = signs.reshape((*signs.shape, *((1,) * (fields.ndim - 2))))
    full = positive[:, abs(offsets)] * signs
    return np.moveaxis(full.astype(fields.dtype), (0, 1), (caxis, yaxis))

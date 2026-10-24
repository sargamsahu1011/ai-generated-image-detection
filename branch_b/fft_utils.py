from PIL import Image
import numpy as np


def image_to_fft(image: Image.Image, size: int = 224) -> Image.Image:
    """Convert an image into a normalized FFT magnitude spectrum."""
    image = image.convert("RGB").resize((size, size))

    pixels = np.asarray(image, dtype=np.float32) / 255.0

    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))

    low, high = np.percentile(magnitude, [1, 99])
    magnitude = np.clip(
        (magnitude - low) / max(float(high - low), 1e-8),
        0,
        1,
    )

    fft_uint8 = (magnitude * 255).astype(np.uint8)
    fft_rgb = np.repeat(fft_uint8[:, :, None], 3, axis=2)

    return Image.fromarray(fft_rgb)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fftpack import fftshift, fft2
from PIL import Image

# Percorso dell'immagine .png da aprire
image_path = './test_fake_images/seed03834427.png'

# Apri l'immagine e convertila in scala di grigi
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Calcola la FFT2 dell'immagine e shifta lo spettro verso il centro
fft_result = fft2(image_array)
fft_shifted = fftshift(fft_result)

# Calcola il modulo e il logaritmo del modulo per una migliore visualizzazione
fft_magnitude = np.abs(fft_shifted)
fft_magnitude_log = np.log1p(fft_magnitude)

# Plotta lo spettro della FFT2
plt.figure(figsize=(10, 10))
plt.imshow(fft_magnitude_log, cmap='gray', norm=LogNorm())
plt.title('Spettro della FFT2 con spostamento al centro')
plt.colorbar(format='%g')
plt.show()
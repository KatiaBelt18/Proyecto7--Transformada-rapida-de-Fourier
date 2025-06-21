import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from FFT import *
from IFFT import *

# Programa principal 

# Leer archivo de audio 
sample_rate, data = wavfile.read("entrada1.wav")
print("Sample rate:", sample_rate)
print("Shape:", data.shape)

# Convertir a mono si es estéreo
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Ajustar longitud a potencia de 2
N = 2**int(np.floor(np.log2(len(data))))
data = data[:N]

# Convertir a complejos para FFT
data_complex = [complex(val, 0) for val in data]

# FFT
fft_data = fft(data_complex)

# Crear arreglo de frecuencias
freq = np.fft.fftfreq(N, d=1/sample_rate)

# Magnitud original
magnitude = np.abs(fft_data)

# Filtro pasabajas
cutoff = 1000  # Hz
fft_filtered = [val if abs(freq[i]) <= cutoff else 0 for i, val in enumerate(fft_data)]

# Magnitud filtrada
magnitude_filtered = np.abs(fft_filtered)

# Graficar espectros antes y después del filtro
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(freq[:N//2], magnitude[:N//2])
plt.title("Espectro original")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

plt.subplot(1,2,2)
plt.plot(freq[:N//2], magnitude_filtered[:N//2])
plt.title("Espectro filtrado")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

plt.tight_layout()
plt.show()

# IFFT normalizada para reconstruir señal
filtered_signal_complex = ifft_normalized(fft_filtered)

# Tomar solo la parte real y convertir a int16 para WAV
filtered_signal = np.array([int(np.real(x)) for x in filtered_signal_complex], dtype=np.int16)

# Guardar archivo de salida
wavfile.write("salida_filtrada.wav", sample_rate, filtered_signal)
print("Archivo guardado como salida_filtrada.wav")

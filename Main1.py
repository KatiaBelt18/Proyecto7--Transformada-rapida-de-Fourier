import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from FFT import *
from IFFT import *

# Programa principal 

# Leer archivo de audio 
archivo_entrada = input("Nombre del archivo .wav: ")
sample_rate, data = wavfile.read(archivo_entrada)
print("Sample rate:", sample_rate)

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
cutoff = 500  # Hz
fft_filtered = [val if abs(freq[i]) <= cutoff else 0 for i, val in enumerate(fft_data)]

# Magnitud filtrada
magnitude_filtered = np.abs(fft_filtered)

# IFFT normalizada para reconstruir señal
filtered_signal_complex = ifft_normalized(fft_filtered)

# Tomar solo la parte real y convertir a int16 para WAV
filtered_signal = np.array([int(np.real(x)) for x in filtered_signal_complex], dtype=np.int16)

# Graficar dominio del tiempo y frecuencia
plt.figure(figsize=(14,10))

# Señal original en el tiempo
plt.subplot(2, 2, 1)
t = np.arange(N) / sample_rate
plt.plot(t, data)
plt.title("Señal original (dominio del tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

# Señal filtrada en el tiempo
plt.subplot(2, 2, 2)
plt.plot(t, filtered_signal)
plt.title("Señal filtrada (dominio del tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

# Espectro original
plt.subplot(2, 2, 3)
plt.plot(freq[:N//2], magnitude[:N//2])
plt.title("Espectro original")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

# Espectro filtrado
plt.subplot(2, 2, 4)
plt.plot(freq[:N//2], magnitude_filtered[:N//2])
plt.title("Espectro filtrado")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

plt.tight_layout()
plt.show()

# Guardar archivo de salida
wavfile.write("audio_salida.wav", sample_rate, filtered_signal)
print("Archivo guardado como audio_salida.wav")

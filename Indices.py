import numpy as np
from scipy.io import wavfile
from scipy import signal

def compute_spectrogram(data, samplerate, windowLength=512, windowHop=256, windowType='hann'):
    """
    Compute a spectrogram of an audio signal.
    Return the spectrogram and the list of frequencies.

    Parameters:
    data -- Array of audio signal samples.
    samplerate -- Sampling rate of the audio signal.
    windowLength -- Length of the FFT window (in samples).
    windowHop -- Hop size of the FFT window (in samples).
    windowType -- Type of the window function to use.
    """
    W = signal.get_window(windowType, windowLength, fftbins=False)
    frames = [
        data[i:i + windowLength] * W
        for i in range(0, len(data) - windowLength + 1, windowHop)
    ]
    spectro = [abs(np.fft.rfft(frame)) ** 2 for frame in frames]
    spectro = np.transpose(spectro)
    frequencies = np.fft.rfftfreq(windowLength, 1 / samplerate)
    return spectro, frequencies

def compute_ACI(spectro, j_bin):
    """
    Compute the Acoustic Complexity Index (ACI) from the spectrogram.

    Parameters:
    spectro -- The spectrogram of the audio signal (2D array).
    j_bin -- Temporal size of the frame (in samples).

    Returns:
    A single global ACI value.
    """
    times = range(0, spectro.shape[1] - 10, j_bin)
    jspecs = [np.array(spectro[:, i:i + j_bin]) for i in times]
    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs]
    return sum(aci)

# Exemplo de uso
file_path = r"D:\Dados Chuvas TCC\DadosSelecionados\SMM08571-M1_20231214_074000_2-8_heavy_morning.wav"

# Carregar o áudio
samplerate, data = wavfile.read(file_path)

# Calcular o espectrograma
spectrogram, frequencies = compute_spectrogram(data, samplerate, windowLength=512, windowHop=256, windowType='hann')

# Calcular o ACI
j_bin = 10  # Escolha do tamanho do bin temporal
aci_global = compute_ACI(spectrogram, j_bin)

# Resultado
print(f"ACI Global: {aci_global}")

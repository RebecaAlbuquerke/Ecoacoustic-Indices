import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import random

def compute_spectrogram(data, samplerate, windowLength=512, windowHop=256, 
                        scale_audio=True, square=True, windowType='hann', 
                        centered=False, normalized=False):
    """
    Compute a spectrogram of an audio signal.
    Return a spectrogram matrix and the corresponding frequency bins.

    Parameters:
    data: numpy array of audio samples
    samplerate: sampling rate of the audio file
    windowLength: length of the FFT window (in samples)
    windowHop: hop size of the FFT window (in samples)
    scale_audio: if True, scale signal samples between -1 and 1
    square: if True, use the square of the magnitude for the spectrogram
    windowType: type of window to apply ('hann', 'hamming', etc.)
    centered: if True, center each FFT on the corresponding sliding window
    normalized: if True, normalize spectrogram values
    """
    if scale_audio:
        data = data / np.max(np.abs(data))  # Scale between -1 and 1

    W = signal.get_window(windowType, windowLength, fftbins=False)
    halfWindowLength = int(windowLength / 2)

    if centered:
        time_shift = int(windowLength / 2)
        times = range(time_shift, len(data) + 1 - time_shift, windowHop)  # Centered
        frames = [data[i - time_shift:i + time_shift] * W for i in times]  # Centered frames
    else:
        times = range(0, len(data) - windowLength + 1, windowHop)
        frames = [data[i:i + windowLength] * W for i in times]

    spectro = [
        abs(np.fft.rfft(frame, windowLength))[:halfWindowLength] ** (2 if square else 1)
        for frame in frames
    ]
    spectro = np.transpose(spectro)  # Transpose for a friendly format

    if normalized:
        spectro = spectro / np.max(spectro)  # Normalize to max value

    frequencies = [e * (samplerate / 2) / float(halfWindowLength) for e in range(halfWindowLength)]
    return spectro, frequencies

def compute_ACI(spectro, j_bin):
    """
    Compute the Acoustic Complexity Index (ACI) from the spectrogram of an audio signal.

    Reference: Pieretti N, Farina A, Morri FD (2011)

    Parameters:
    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in spectrogram columns)

    Returns:
    main_value: global ACI value
    temporal_values: ACI values per temporal bin
    """
    times = range(0, spectro.shape[1] - 10, j_bin)  # Temporal indices for sub-spectrograms
    jspecs = [np.array(spectro[:, i:i + j_bin]) for i in times]  # Sub-spectrograms of size j_bin

    aci = [
        sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1)))
        for jspec in jspecs
    ]  # ACI for each sub-spectrogram
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values

def process_audio_and_compute_ACI(input_folder, output_folder, windowLength=512, windowHop=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                output_spectro = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_spectrogram.npz")
                output_aci = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_ACI.npz")

                try:
                    # Read audio file
                    samplerate, data = wavfile.read(file_path)
                    
                    # Compute spectrogram
                    spectrogram, frequencies = compute_spectrogram(
                        data, samplerate, windowLength, windowHop
                    )
                    
                    # Select a random j_bin for ACI computation
                    j_bin = random.randint(5, 50)  # Random value between 5 and 50
                    main_aci, temporal_aci = compute_ACI(spectrogram, j_bin)

                    # Save spectrogram and ACI results
                    np.savez_compressed(output_spectro, spectrogram=spectrogram, frequencies=frequencies)
                    np.savez_compressed(output_aci, main_aci=main_aci, temporal_aci=temporal_aci, j_bin=j_bin)
                    
                    print(f"Processed and saved: {output_spectro} and {output_aci}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Diretórios de entrada e saída
input_directories = [
    r"D:\Dados Chuvas TCC\DadosSelecionados",
    # r"D:\Dados Chuvas TCC\Mamiraua\5 - SMM11209 (vb trilha)\Data",
    # r"D:\Dados Chuvas TCC\Mamiraua\7 - SMM11307 (ch trilha)\Data",
    # r"D:\Dados Chuvas TCC\Campus\1 - SMM00894 (plv)\Data"
]

output_base = r"D:\Dados Chuvas TCC\Processed"

# Processar cada pasta
for input_dir in input_directories:
    output_dir = os.path.join(output_base, os.path.basename(input_dir))
    process_audio_and_compute_ACI(input_dir, output_dir)

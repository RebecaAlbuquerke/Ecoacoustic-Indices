import os
from scipy.io import wavfile
import numpy as np
from scipy import signal

def compute_spectrogram(data, samplerate, windowLength=512, windowHop=256, 
                        scale_audio=True, square=True, windowType='hann', 
                        centered=False, normalized=False):
    """
    Compute a spectrogram of an audio signal.
    Return a list of list of values as the spectrogram, and a list of frequencies.

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

def process_audio_files(input_folder, output_folder, windowLength=512, windowHop=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_spectrogram.npz")
                
                try:
                    # Read audio file
                    samplerate, data = wavfile.read(file_path)
                    
                    # Compute spectrogram
                    spectrogram, frequencies = compute_spectrogram(
                        data, samplerate, windowLength, windowHop
                    )
                    
                    # Save as a compressed NumPy file
                    np.savez_compressed(output_path, spectrogram=spectrogram, frequencies=frequencies)
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Diretórios de entrada e saída
input_directories = [
    r"D:\Dados Chuvas TCC\DadosSelecionados",
    # r"D:\Dados Chuvas TCC\Mamiraua\5 - SMM11209 (vb trilha)\Data",
    # r"D:\Dados Chuvas TCC\Mamiraua\7 - SMM11307 (ch trilha)\Data",
    # r"D:\Dados Chuvas TCC\Campus\1 - SMM00894 (plv)\Data"
]

output_base = r"D:\Dados Chuvas TCC\Spectrograms"

# Processar cada pasta
for input_dir in input_directories:
    output_dir = os.path.join(output_base, os.path.basename(input_dir))
    process_audio_files(input_dir, output_dir)

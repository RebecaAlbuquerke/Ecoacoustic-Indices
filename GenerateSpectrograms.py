import os
import warnings
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import WavFileWarning

# Ignorar avisos de chunks não suportados
warnings.simplefilter('ignore', WavFileWarning)

def generate_spectrograms(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
                
                try:
                    samplerate, data = wavfile.read(file_path)

                    # Calcular o espectrograma
                    f, t, Sxx = spectrogram(data, fs=samplerate, window='hann', nperseg=512, noverlap=256)
                    
                    # Plotar e salvar o espectrograma
                    plt.figure(figsize=(10, 6))
                    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
                    plt.colorbar(label='Intensity (dB)')
                    plt.title(f"Spectrogram of {file}")
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.savefig(save_path)
                    plt.close()
                    
                    print(f"Saved spectrogram: {save_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Paths to your directories
directories = [
    r"D:\Dados Chuvas TCC\DadosSelecionados",
    # r"D:\Dados Chuvas TCC\Mamiraua\5 - SMM11209 (vb trilha)\Data",
    # r"D:\Dados Chuvas TCC\Mamiraua\7 - SMM11307 (ch trilha)\Data",
    # r"D:\Dados Chuvas TCC\Campus\1 - SMM00894 (plv)\Data"
]

output_base = r"D:\Dados Chuvas TCC\Spectrograms"

# Generate spectrograms for each directory
for input_dir in directories:
    output_dir = os.path.join(output_base, os.path.basename(input_dir))
    generate_spectrograms(input_dir, output_dir)

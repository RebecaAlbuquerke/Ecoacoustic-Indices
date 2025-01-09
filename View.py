import numpy as np
import matplotlib.pyplot as plt

def plot_spectrogram(file_path):
    """
    Carrega e plota o espectrograma salvo no arquivo .npz.
    """
    data = np.load(file_path)
    spectrogram = data['spectrogram']
    frequencies = data['frequencies']

    plt.figure(figsize=(12, 6))
    plt.imshow(
        10 * np.log10(spectrogram + 1e-10),  # Escala logarítmica para dB
        aspect='auto',
        origin='lower',
        extent=[0, spectrogram.shape[1], frequencies[0], frequencies[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel("Tempo (frames)")
    plt.ylabel("Frequência (Hz)")
    plt.title("Espectrograma")
    plt.show()

def plot_ACI(file_path):
    """
    Carrega e plota os valores de ACI global e temporal.
    """
    data = np.load(file_path)
    main_aci = data['main_aci']
    temporal_aci = data['temporal_aci']
    j_bin = data['j_bin']

    plt.figure(figsize=(12, 6))
    plt.plot(temporal_aci, label='ACI Temporal', marker='o')
    plt.axhline(main_aci, color='red', linestyle='--', label=f'ACI Global (Total: {main_aci:.2f})')
    plt.xlabel("Janela Temporal (bins)")
    plt.ylabel("ACI")
    plt.title(f"Acoustic Complexity Index (j_bin={j_bin})")
    plt.legend()
    plt.grid()
    plt.show()

# Exemplos de visualização
spectrogram_file = r"D:\Dados Chuvas TCC\Processed\DadosSelecionados\SMM11284_20240412_050000_0-0_no-rain_night_spectrogram.npz"
aci_file = r"D:\Dados Chuvas TCC\Processed\DadosSelecionados\SMM11284_20240412_050000_0-0_no-rain_night_ACI.npz"

# Plotar espectrograma
plot_spectrogram(spectrogram_file)

# Plotar valores de ACI
plot_ACI(aci_file)

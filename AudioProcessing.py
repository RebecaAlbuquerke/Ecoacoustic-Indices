import os
import librosa
import numpy as np
import pandas as pd

# Diretórios dos grupos
directories = {
    "campus": r"D:\Dados Chuvas TCC\Campus\1 - SMM00894 (plv)\Data",
    "mamiraua_regua": r"D:\Dados Chuvas TCC\Mamiraua\4 - SMM08571M1 (regua)\Data",
    "mamiraua_vb_trilha": r"D:\Dados Chuvas TCC\Mamiraua\5 - SMM11209 (vb trilha)\Data",
    "mamiraua_ch_trilha": r"D:\Dados Chuvas TCC\Mamiraua\7 - SMM11307 (ch trilha)\Data"
}

# Pasta para salvar os resultados
output_directory = "DataLibrosa"
os.makedirs(output_directory, exist_ok=True)

# Função para calcular o ACI
def calculate_aci(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        frame_size = int(sr * 5)  # Tamanho do frame de 5 segundos
        hop_size = frame_size
        aci_values = []

        for start in range(0, len(y), hop_size):
            end = start + frame_size
            frame = y[start:end]
            if len(frame) < frame_size:
                break
            spectrum = np.abs(np.fft.rfft(frame))
            aci = np.sum(np.abs(np.diff(spectrum))) / np.sum(spectrum)
            aci_values.append(aci)

        return np.mean(aci_values) if aci_values else None
    except Exception as e:
        print(f"Erro ao processar {audio_path}: {e}")
        return None

# Processar os arquivos de áudio em cada diretório
for group, directory in directories.items():
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Filtrar arquivos de áudio
                file_path = os.path.join(root, file)
                aci_value = calculate_aci(file_path)
                results.append({"Group": group, "File": file, "ACI": aci_value})

    # Salvar os resultados em um arquivo CSV
    output_file = os.path.join(output_directory, f"{group}_aci_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Resultados do grupo '{group}' salvos em {output_file}")

print("Processamento concluído!")

import os
import pandas as pd
import re

# Diretórios e arquivos de saída
datasets = {
    "campus": r"D:\Dados Chuvas TCC\Campus\1 - SMM00894 (plv)\Data",
    "mamiraua_regua": r"D:\Dados Chuvas TCC\Mamiraua\4 - SMM08571M1 (regua)\Data",
    "mamiraua_vb_trilha": r"D:\Dados Chuvas TCC\Mamiraua\5 - SMM11209 (vb trilha)\Data",
    "mamiraua_ch_trilha": r"D:\Dados Chuvas TCC\Mamiraua\7 - SMM11307 (ch trilha)\Data"
}
output_path = r"D:\Dados Chuvas TCC\Dataset"

# Cria o diretório de saída se não existir
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Função para processar os nomes de arquivos e extrair informações
def parse_filename(file_name):
    # Regex para capturar os componentes do nome do arquivo
    match = re.match(r"(\w+)_(\d{8}_\d{6})_(\d+-\d+)_(\w+)_(\w+)\.wav", file_name)
    if match:
        return {
            "Filename": match.group(1),
            "Timestamp": match.group(2),
            "Total Rain": match.group(3),
            "Rain Class": match.group(4),
            "Period": match.group(5)
        }
    return None

# Itera pelos diretórios e cria os datasets
for dataset_name, input_path in datasets.items():
    audio_files = []

    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        
        # Verifica se é um arquivo válido .wav
        if os.path.isfile(file_path) and file_name.lower().endswith('.wav'):
            file_data = parse_filename(file_name)
            if file_data:  # Inclui apenas arquivos válidos
                audio_files.append(file_data)

    # Cria o DataFrame e salva o dataset
    if audio_files:
        output_file = os.path.join(output_path, f"{dataset_name}_dataset.csv")
        audio_df = pd.DataFrame(audio_files)
        audio_df.to_csv(output_file, index=False)
        print(f"Dataset {dataset_name} salvo com sucesso em: {output_file}")
    else:
        print(f"Nenhum arquivo válido encontrado no dataset {dataset_name}.")


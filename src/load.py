
import polars as pl
import numpy as np

def load_part_data(part_number):
    dfs = []
    for device in range(1, 6):
        filename = f"data/part{part_number}/part{part_number}dev{device}.csv"
        try:
            df = pl.read_csv(filename, has_header=False)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filename}")
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")
    if dfs:
        return pl.concat(dfs).to_numpy()
    else:
        return np.empty((0,))

def load_data():
    full_data = [load_part_data(i) for i in range(14)]
    return np.concatenate([d for d in full_data if d.size > 0], axis=0)

'''

def load_part_data(part_number):

    data = []

    for device in range(1, 6):
        filename = f"data/part{part_number}/part{part_number}dev{device}.csv"
        
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append([float(x) for x in row])
        except FileNotFoundError:
            print(f"Arquivo não encontrado.")
            continue
    
    return np.array(data)

def load_data():
    full_data = [load_part_data(i) for i in range(14)]
    return np.concatenate(full_data, axis=0)

'''
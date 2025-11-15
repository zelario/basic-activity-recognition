
import csv
import numpy as np

# --- Exercise 1: Load Part Data ---

def load_part_data(part_number):

    data = []

    for device in range(1, 6):
        filename = f"data/part{part_number}/part{part_number}dev{device}.csv"
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append([float(x) for x in row] + [part_number])
        except FileNotFoundError:
            print(f"Arquivo não encontrado.")
            continue
    return np.array(data)

# --- Exercise 2: Load Full Data ---

def load_data():

    try:
        data = np.load("data/data.npy", allow_pickle=True)
        return data
    except FileNotFoundError:
        full_data = [load_part_data(i) for i in range(14)]
        data = np.concatenate(full_data, axis=0)
        np.save("data/data.npy", data)
        return data

import csv
import numpy as np

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

def load_data():
    full_data = [load_part_data(i) for i in range(14)]
    return np.concatenate(full_data, axis=0)
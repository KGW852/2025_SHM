# csv_utils.py
import os
import csv

def read_csv(file_path, skip_header=False):
    data = []

    if not os.path.exists(file_path):
        return data

    # Read lines from file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Skip the header
    if skip_header and len(lines) > 0:
        lines = lines[1:]

    # Parse each line
    for line in lines:
        values = line.strip().split(',')
        data.append(values)

    return data

def save_csv(save_data, save_path):
    with open(save_path, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(save_data)
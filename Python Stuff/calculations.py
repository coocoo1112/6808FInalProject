import csv


csv_name = "experiment.csv"

with open(csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row.keys())
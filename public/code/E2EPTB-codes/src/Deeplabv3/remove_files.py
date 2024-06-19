import os

set_of_files = set(os.listdir("data/masks"))
for filename in os.listdir("data/ultrasounds"):
    if filename not in set_of_files:
        print(filename)
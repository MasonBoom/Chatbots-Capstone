from itertools import zip_longest
import re

data_path = "dialogs.txt" #path_name.txt

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

new_lines = []
#print(lines[3])
for line in lines:
    line = line.split("\t")
    for part in line:
        new_lines.append(part)

# group lines by response pair
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
pairs = list(grouper(new_lines, 2))
#print(pairs)
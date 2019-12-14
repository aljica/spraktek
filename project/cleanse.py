lines = []
with open("SMSSpamCollection.txt", "r") as f:
    for line in f:
        line = line[4:]
        lines.append(line.strip())

with open("out.txt", "w") as f:
    for line in lines:
        f.write(line)
        f.write("\n")

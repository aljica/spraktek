# Cleanses SMSSpamCollection and picks out only the text messages.
# Each row in output file "sms.txt" contains a text message.

lines = []
with open("SMSSpamCollection.txt", "r") as f:
    for line in f:
        line = line[4:]
        lines.append(line.strip())

with open("sms.txt", "w") as f:
    for line in lines:
        f.write(line)
        f.write("\n")

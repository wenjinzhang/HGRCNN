import queue

data = []

for i in range(18):
    data.append(i)

print(data)

del data[0]
data.append(18)
print(data)


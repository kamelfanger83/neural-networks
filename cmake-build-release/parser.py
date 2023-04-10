f = open("t10k-labels.idx1-ubyte", "rb")

read = f.read()

o = open("labelst.txt", "w")

for writer in range(8,len(read)):
    o.write(str(read[writer]) + "\n")

o.close()
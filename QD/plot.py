import matplotlib.pyplot as plt

f = open("qd1.out", "r")
rows = []
for line in f:
	rows.append(line.split())

cols = zip(*rows)

plt.plot(cols[0], cols[1], label="ekin")
plt.plot(cols[0], cols[1], label="epot")
plt.plot(cols[0], cols[1], label="etot")
plt.xlabel("time")
plt.legend()
plt.show()
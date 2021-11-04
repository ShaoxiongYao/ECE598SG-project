import matplotlib.pyplot as plt

with open('temp_file', 'r') as f:
    lines = f.readlines()

loss_lst = []
for idx, line in enumerate(lines):
    if idx%2 == 1:
        loss = float(line.split()[4])
        loss_lst.append(loss)

plt.plot(loss_lst)
plt.title("Validation loss vs. training epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("val_loss.png")
# plt.show()

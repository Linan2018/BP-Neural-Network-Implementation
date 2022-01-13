import os
import pickle

import matplotlib.pyplot as plt

pkl_dir_name = "pkl_files"
fig_dir_name = "figs"

# 就是画图，这部分写的有点啰嗦。。。。。。

pkl_files = os.listdir(pkl_dir_name)
pkl_files_0 = [f for f in pkl_files if f.startswith('0')]
pkl_files_1 = [f for f in pkl_files if f.startswith('1')]
pkl_files_2 = [f for f in pkl_files if f.startswith('2')]
pkl_files_3 = [f for f in pkl_files if f.startswith('3')]

n = 999
x = range(n)
train_loss_all, train_acc_all, test_loss_all, test_acc_all = [], [], [], []
lr = []

# lr
for pkl_file in pkl_files_0:
    lr.append(eval(pkl_file.split('-')[1][:-4])["lr"])
    with open(pkl_dir_name + '/' + pkl_file, 'rb') as f:
        train_loss, train_acc, test_loss, test_acc = pickle.load(f)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(test_loss)
        test_acc_all.append(test_acc)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

for i in range(len(pkl_files_0)):
    train_loss, train_acc, test_loss, test_acc = train_loss_all[i], train_acc_all[i], test_loss_all[i], test_acc_all[i]
    ax1.plot(x, train_loss, label="lr=" + str(lr[i]))
    ax1.set_xlabel('iterations')
    ax1.set_title('Train loss curve')
    ax1.legend()

    ax2.plot(x, test_loss, label="lr=" + str(lr[i]))
    ax2.set_xlabel('iterations')
    ax2.set_title('Test loss curve')
    ax2.legend()

    ax3.plot(x, train_acc, label="lr=" + str(lr[i]))
    ax3.set_xlabel('iterations')
    ax3.set_title('Train accuracy curve')
    ax3.legend()

    ax4.plot(x, test_acc, label="lr=" + str(lr[i]))
    ax4.set_xlabel('iterations')
    ax4.set_title('Test accuracy curve')
    ax4.legend()

fig.suptitle('Comparison of learning rates', fontsize=16)
plt.savefig(fig_dir_name + "/" + "lr.svg")
plt.show()

# node
node = []
train_loss_all, train_acc_all, test_loss_all, test_acc_all = [], [], [], []
for pkl_file in pkl_files_1:
    # print(eval(pkl_file.split('-')[1][:-4])["hidden_nodes"])
    node.append(eval(pkl_file.split('-')[1][:-4])["hidden_nodes"])
    with open(pkl_dir_name + '/' + pkl_file, 'rb') as f:
        train_loss, train_acc, test_loss, test_acc = pickle.load(f)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(test_loss)
        test_acc_all.append(test_acc)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
# print(node)
for i in range(len(pkl_files_1)):
    train_loss, train_acc, test_loss, test_acc = train_loss_all[i], train_acc_all[i], test_loss_all[i], test_acc_all[i]
    ax1.plot(x, train_loss, label="n=" + str(node[i]))
    ax1.set_xlabel('iterations')
    ax1.set_title('Train loss curve')
    ax1.legend()

    ax2.plot(x, test_loss, label="n=" + str(node[i]))
    ax2.set_xlabel('iterations')
    ax2.set_title('Test loss curve')
    ax2.legend()

    ax3.plot(x, train_acc, label="n=" + str(node[i]))
    ax3.set_xlabel('iterations')
    ax3.set_title('Train accuracy curve')
    ax3.legend()

    ax4.plot(x, test_acc, label="n=" + str(node[i]))
    ax4.set_xlabel('iterations')
    ax4.set_title('Test accuracy curve')
    ax4.legend()

fig.suptitle('Comparison of hidden nodes', fontsize=16)
plt.savefig(fig_dir_name + "/" + "node.svg")
plt.show()

gamma = []
train_loss_all, train_acc_all, test_loss_all, test_acc_all = [], [], [], []
for pkl_file in pkl_files_2:
    gamma.append(eval(pkl_file.split('-')[1][:-4])["gamma"])
    with open(pkl_dir_name + '/' + pkl_file, 'rb') as f:
        train_loss, train_acc, test_loss, test_acc = pickle.load(f)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(test_loss)
        test_acc_all.append(test_acc)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

for i in range(len(pkl_files_2)):
    train_loss, train_acc, test_loss, test_acc = train_loss_all[i], train_acc_all[i], test_loss_all[i], test_acc_all[i]
    ax1.plot(x, train_loss, label="gamma=" + str(gamma[i]))
    ax1.set_xlabel('iterations')
    ax1.set_title('Train loss curve')
    ax1.legend()

    ax2.plot(x, test_loss, label="gamma=" + str(gamma[i]))
    ax2.set_xlabel('iterations')
    ax2.set_title('Test loss curve')
    ax2.legend()

    ax3.plot(x, train_acc, label="gamma=" + str(gamma[i]))
    ax3.set_xlabel('iterations')
    ax3.set_title('Train accuracy curve')
    ax3.legend()

    ax4.plot(x, test_acc, label="gamma=" + str(gamma[i]))
    ax4.set_xlabel('iterations')
    ax4.set_title('Test accuracy curve')
    ax4.legend()

fig.suptitle('Comparison of gamma', fontsize=16)
plt.savefig(fig_dir_name + "/" + "gamma.svg")
plt.show()

bs = []
train_loss_all, train_acc_all, test_loss_all, test_acc_all = [], [], [], []
for pkl_file in pkl_files_3:
    bs.append(eval(pkl_file.split('-')[1][:-4])["batch_size"])
    with open(pkl_dir_name + '/' + pkl_file, 'rb') as f:
        train_loss, train_acc, test_loss, test_acc = pickle.load(f)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(test_loss)
        test_acc_all.append(test_acc)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

for i in range(len(pkl_files_3)):
    train_loss, train_acc, test_loss, test_acc = train_loss_all[i], train_acc_all[i], test_loss_all[i], test_acc_all[i]
    ax1.plot(x, train_loss, label="batch size=" + str(bs[i]))
    ax1.set_xlabel('iterations')
    ax1.set_title('Train loss curve')
    ax1.legend()

    ax2.plot(x, test_loss, label="batch size=" + str(bs[i]))
    ax2.set_xlabel('iterations')
    ax2.set_title('Test loss curve')
    ax2.legend()

    ax3.plot(x, train_acc, label="batch size=" + str(bs[i]))
    ax3.set_xlabel('iterations')
    ax3.set_title('Train accuracy curve')
    ax3.legend()

    ax4.plot(x, test_acc, label="batch size=" + str(bs[i]))
    ax4.set_xlabel('iterations')
    ax4.set_title('Test accuracy curve')
    ax4.legend()

fig.suptitle('Comparison of batch size', fontsize=16)
plt.savefig(fig_dir_name + "/" + "batch_size.svg")
plt.show()

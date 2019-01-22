import re
from matplotlib import pyplot as plt

with open('log.txt', 'r', encoding='utf-8') as f:
    reconstruction_errors1 = list()
    reconstruction_errors2 = list()
    reconstruction_errors3 = list()
    index = 0
    for line in f:
        # print(line)
        matchGroup = re.match('Epoch: (\d+) reconstruction error: (.+)', line)
        if matchGroup:
            if index == 0:
                reconstruction_errors1.append(float(matchGroup.group(2)))
            elif index == 1:
                reconstruction_errors2.append(float(matchGroup.group(2)))
            elif index == 2:
                reconstruction_errors3.append(float(matchGroup.group(2)))
            print(matchGroup.group(1))
            if int(matchGroup.group(1)) == 99:
                index += 1

    plt.figure()
    plt.plot(range(len(reconstruction_errors1)), reconstruction_errors1, c='b', label='RBM 1')
    plt.plot(range(len(reconstruction_errors2)), reconstruction_errors2, c='r', label='RBM 2')
    plt.plot(range(len(reconstruction_errors3)), reconstruction_errors3, c='g', label='RBM 3')
    plt.xlabel('epochs')
    plt.ylabel('reconstruction_errors')
    plt.legend()
    plt.show()

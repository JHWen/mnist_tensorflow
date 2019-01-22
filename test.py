import re
from matplotlib import pyplot as plt

with open('log2.txt', 'r', encoding='utf-8') as f:
    mean_errors = list()
    for line in f:
        # print(line)
        matchGroup = re.match('Accuracy rating for epoch (\d+): (.+)', line)
        if matchGroup:
            print(matchGroup.group(1))
            accuracy = float(matchGroup.group(2)) * 100.0
            print(float('%.2f' % accuracy))
            mean_errors.append(float('%.2f' % accuracy))

    plt.figure()
    plt.plot(range(len(mean_errors)), mean_errors, c='b')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.show()

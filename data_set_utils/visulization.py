import matplotlib.pyplot as plt
import os
import csv
import numpy as np
# names = ['1:1', '1:2', '1:3','1:4','1:5','1:6','1:7','1:8','1:9' ]
# values = [82, 69,62,53,46,40,38,35,31]
#
# plt.plot(names, values)
# plt.suptitle('F1-score')
# plt.show()

def plot_pretrain_stat(dir_id):
    save_dir = save_dir = 'saves/' + dir_id
    config_file = open(save_dir + '/configs.txt', 'r')
    print(config_file.read())
    config_file.close()
    stat_on_weak = []
    stat = []
    with open(save_dir + '/logs/pretrain_log.csv') as log_csv:
        for i, item in enumerate(csv.reader(log_csv)):
            if item[1].strip() == 'On QR_weak_dev':
                stat_on_weak.append(item[4])
            elif item[1].strip() == 'On QR_dev':
                stat.append(item[4])
        plt.subplot('121')
        plt.plot(np.linspace(1,len(stat_on_weak),num=len(stat)),stat)
        plt.subplot('122')
        plt.plot(np.linspace(1,len(stat_on_weak),num=len(stat)),stat_on_weak)
        plt.show()


if __name__ == '__main__':
    plot_pretrain_stat('1551688424')
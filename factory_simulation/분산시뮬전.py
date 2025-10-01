# -*- coding: cp949 -*-
import numpy as np
import matplotlib.pyplot as plt
from draw import plot_process_data, plot_combined_process_data
from worker import Group_A, Group_B, Group_C, process

# Initialize global variables
global_time, global_interval, process_time = 0, 0.1, 3000


# A 공정 선언(총 4명이 작업, 따라서 프로그래밍 동안에는 a4는 사용 안함)
A_list = []
names = ['a1', 'a2', 'ace', 'a3', 'a4']
A_means = [38.696, 38.958, 31.723, 38.83, 31.866]
A_std = [4.457, 1.974, 19.018, 19.018, 4.242]
for name, mean, std in zip(names, A_means, A_std):
    A_list.append(Group_A(name, mean, std, global_interval))

# B 공정 선언
names = ['b2', 'b3', 'b4', 'b5', 'b6']
B_means = [29, 38, 39.803, 48.13, 54]
B_std = [6.2, 6.2, 6.485, 5.83, 6.2]
b = Group_B(global_interval, B_means, B_std)

# C 공정 선언
C_mean = 1.6
C_std = 0.5
c1 = Group_C('c1', global_interval, C_mean, C_std)
C_list = [c1]

# Run the process
results = process(process_time, global_interval, A_list, b, C_list)

# Plot the results using the returned data
#plot_combined_process_data(*results)

# for individual plots
plot_process_data(*results)
# -*- coding: cp949 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from worker import Group_A, Group_B, Group_C, process

def run_simulation_with_varying_std(global_interval, process_time, std_percent_range, num_repeats):
    A_means = [38.696, 38.958, 31.723, 38.83, 31.866]
    B_means = [29, 38, 39.803, 48.13, 54]
    C_mean = 1.6
    
    A_std_base = [mean * 0.01 for mean in A_means]
    B_std_base = [mean * 0.01 for mean in B_means]
    C_std_base = 0.5
    
    std_percentages = np.linspace(1, 100, std_percent_range)
    after_c_box_means = np.zeros((std_percent_range, std_percent_range))
    after_c_box_vars = np.zeros((std_percent_range, std_percent_range))

    for i, std_a_percent in enumerate(std_percentages):
        for j, std_b_percent in enumerate(std_percentages):
            after_c_box_samples = []
            for _ in range(num_repeats):
                A_list = [Group_A(name, mean, std * std_a_percent, global_interval) for name, mean, std in zip(['a1', 'a2', 'ace', 'a3', 'a4'], A_means, A_std_base)]
                b = Group_B(global_interval, B_means, [std * std_b_percent for std in B_std_base])
                c1 = Group_C('c1', global_interval, C_mean, C_std_base)
                C_list = [c1]
                
                results = process(process_time, global_interval, A_list, b, C_list)
                after_c_box_samples.append(results[5][-1])  # 각 시뮬레이션 결과를 저장

                print(f"A의 분산: 평균의 {std_a_percent}%, B의 분산: 평균의 {std_b_percent}%의 {_}번째 실행")

           

            after_c_box_means[i, j] = np.mean(after_c_box_samples)  # 평균을 저장
            after_c_box_vars[i, j] = np.var(after_c_box_samples)  # 분산을 저장
    
    return std_percentages, after_c_box_means, after_c_box_vars

# Initialize global variables
global_interval, process_time = 0.1, 3000

# Run the simulation with varying standard deviations
std_percent_range = 20  # Range from 1% to 100%, 20 steps
num_repeats = 100 # Number of repetitions for each combination
std_percentages, after_c_box_means, after_c_box_vars = run_simulation_with_varying_std(global_interval, process_time, std_percent_range, num_repeats)

# Plot the results in a 3D scatter plot
X, Y = np.meshgrid(std_percentages, std_percentages)
Z = after_c_box_means
C = after_c_box_vars

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X, Y, Z, c=C, cmap='viridis')

ax.set_xlabel('A Standard Deviation Percentage')

ax.set_ylabel('B Standard Deviation Percentage')
ax.set_zlabel('Mean after_c_box Value')
ax.set_title('Effect of A and B Standard Deviations on Mean after_c_box with Variance Coloring')
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)


plt.show()

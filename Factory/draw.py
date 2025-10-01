# draw.py

import matplotlib.pyplot as plt

def plot_combined_process_data(time_points, after_a_values, through_b_values, after_b_values, through_c_values, after_c_box_values, sum_b_values, sum_a_values):
    plt.figure(figsize=(14, 8))
    plt.plot(time_points, after_a_values, label='after_a')
    plt.plot(time_points, through_b_values, label='through_b')
    plt.plot(time_points, after_b_values, label='after_b')
    plt.plot(time_points, through_c_values, label='through_c')
    plt.plot(time_points, after_c_box_values, label='after_c_box')
    plt.plot(time_points, sum_b_values, label='sum_b')
    plt.plot(time_points, sum_a_values, label='sum_a')
    plt.xlabel('Time', fontsize=3)
    plt.ylabel('Values', fontsize=3)
    plt.title('Process Output Over Time', fontsize=5)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

def plot_process_data(time_points, after_a_values, through_b_values, after_b_values, through_c_values, after_c_box_values, sum_b_values, sum_a_values):
    plt.figure(figsize=(5, 6))

    #b공정에서 작업 부하량
    plt.subplot(4, 1, 1)
    plt.plot(time_points, through_b_values, label='through_b', color='g')
    plt.xlabel('Time', fontsize=3)
    plt.ylabel('through_b', fontsize=3)
    plt.title('through_b Over Time', fontsize=5)
    plt.xticks(fontsize=2)
    plt.yticks(fontsize=2)
    plt.grid(True)


    plt.subplot(4, 1, 2)
    plt.plot(time_points, sum_a_values, label='sum_a', color='k')
    plt.xlabel('Time', fontsize=3)
    plt.ylabel('sum_a', fontsize=3)
    plt.title('sum_a Over Time', fontsize=5)
    plt.xticks(fontsize=2)
    plt.yticks(fontsize=2)
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time_points, sum_b_values, label='sum_b', color='y')
    plt.xlabel('Time', fontsize=3)
    plt.ylabel('sum_b', fontsize=3)
    plt.title('sum_b Over Time', fontsize=5)
    plt.xticks(fontsize=2)
    plt.yticks(fontsize=2)
    plt.grid(True)


    plt.subplot(4, 1, 4)
    plt.plot(time_points, after_c_box_values, label='after_c_box', color='m')
    plt.xlabel('Time', fontsize=3)
    plt.ylabel('after_c_box', fontsize=3)
    plt.title('after_c_box Over Time', fontsize=5)
    plt.xticks(fontsize=2)
    plt.yticks(fontsize=2)
    plt.grid(True)

    

    plt.tight_layout()
    plt.show()

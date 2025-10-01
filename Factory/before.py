# -*- coding: cp949 -*-

import numpy as np

######## A ########
a1 = [32.21, 38.10, 40.85, 36.68, 45.64]
a2 = [39.04, 36.89, 42.11, 37.79]
ace = [30.12, 31.67, 36.59, 32.36, 31.91, 30.52, 36.36, 34.82, 29.72, 28.94, 25.65, 32.01]
a3 = [33.98, 35.01, 36.59, 33.33, 34,98, 32.15, 38.83, 26.61, 31.03, 27.6]
a4 = [41.26, 27.45, 33.93, 33.89, 26.81, 27.65, 31.74, 30.92, 33.14]

A_list = [a1, a2, ace, a3, a4]
names = ['A1', 'A2', 'Ace', 'A3', 'A4']


######## B ########
b_4 = [41, 33, 37.01,  43.52, 51.42, 32.87]
b_5 = [42.30, 53.96]

B_list = [b_4, b_5]
names = ['b_4', 'b_5']

for B, name in zip(B_list, names):
    print(name)
    print(f'평균: {np.mean(B)}')
    print(f'표준편차: {np.std(B)}')


from scipy.stats import norm

# Given means and standard deviations
means = {'b_2': 35.41, 'b_3': 29.23}
std_devs = {'b_2': 6.2, 'b_3': 6.2}

# Values to find probability for
values = {'b_2': 29, 'b_3': 38}

# Calculate probability for each value
probabilities = {}
for key, value in values.items():
    z_score = (value - means[key]) / std_devs[key]
    probability = norm.cdf(z_score)
    probabilities[key] = probability

# Output probabilities
for key, value in probabilities.items():
    print(f"Probability of observing {key} given the mean and standard deviation: {value:.4f}")

# -*- coding: cp949 -*-

import numpy as np
global after_a, after_b, after_c_box, after_c_plt
global_interval = 1
after_a = 0
after_b = 0
after_c_box = 0
after_c_plt = 0

class Group_A:
    def __init__(self, name, mean, std):
        self.name = name
        self.mean = mean
        self.std = std

        #일 안하고 있는 상태(False)
        self.status = False
        self.work_t = 0

    def work(self):
        global after_a
        #일을 하고 있지 않다면, 작업 시작
        if self.status == False:
            self.work_t = np.random.normal(self.mean, self.std)
            self.status = True

        #작업중
        self.work_t -= global_interval

        #작업이 완료된 경우, 일 하고 있지 않는 상태로 변경    
        if self.work_t <= 0:
            self.status = False
            after_a += 1
            

# Group A
A_list = []
names = ['a1', 'a2', 'ace', 'a3', 'a4']
A_means = [38.696, 38.958, 31.723, 38.83, 31.866]
A_std = [4.457, 1.974, 19.018, 19.018, 4.242]
for name, mean, std in zip(names, A_means, A_std):
    A_list.append(Group_A(name, mean, std))

# B_work
class B_work:
    def __init__(self):
        # 작업시간별 평균,표준편차
        self.b2 = [29,6.2]
        self.b3 = [38,6.2]
        self.b4 = [39.803, 6.485]
        self.b5 = [48.13, 5.83]
        self.b6 = [54, 6.2]

        #일 안하고 있는 상태(False)
        self.status = 0
        self.work_t = 0

    def work(self, after_a):
        global after_b
        if self.status == 0:

            if after_a == 2:
                self.work_t = np.random.normal(*self.b2)
                self.status = 2

            elif after_a == 3:
                self.work_t = np.random.normal(*self.b3)
                self.status = 3

            elif after_a == 4:
                self.work_t = np.random.normal(*self.b4)
                self.status = 4

            elif after_a == 5:
                self.work_t = np.random.normal(*self.b5)
                self.status = 5

            elif after_a >= 6:
                self.work_t = np.random.normal(*self.b6)
                self.status = 6

        #작업중
        self.work_t -= global_interval

        #작업이 완료된 경우, 일 하고 있지 않는 상태로 변경    
        if self.work_t <= 0:
            after_b += self.status
            after_a -= self.status
            self.status = 0
            self.work_t = 0

        return after_a, after_b
b = B_work()

class Group_C:
    def __init__(self, name):
        self.name = name
        self.mean = 4.2
        self.std = 2.1

        #일 안하고 있는 상태(False)
        self.status = False
        self.work_t = 0
        self.through_c = 0

    def work(self, after_b):
        global after_c_box
        #일을 하고 있지 않다면, 작업 시작
        if self.status == False and after_b >= 1:
            self.work_t = np.random.normal(self.mean, self.std)
            after_b -= 1
            self.through_c += 1
            self.status = True

        #작업중
        self.work_t -= global_interval

        #작업이 완료된 경우, 일 하고 있지 않는 상태로 변경    
        if self.status == True and self.work_t <= 0:
            self.status = False
            after_c_box += 1
            self.through_c -= 1

        return after_b

global_time = 0
after_c_plt = after_c_box//50
c1 = Group_C('c1')
c2 = Group_C('c2')
C_list = [c1, c2]
while global_time <= 300:
    global_time += global_interval
    for i in range(len(A_list)-1):
        A_list[i].work()
    print(f'after_a: {after_a} 개')

    after_a, after_b = b.work(after_a)
    print(f'after_b: {after_b} 개')
    for i in range(len(C_list)):
        after_b = C_list[i].work(after_b)

    print(f'after_c_box: {after_c_box} 개')
    print(f'{global_time} 초 이 지났습니다')





# -*- coding: cp949 -*-
import numpy as np


class Group_A:
    def __init__(self, name, mean, std, global_interval):
        self.name = name
        self.mean = mean
        self.std = std
        self.global_interval = global_interval
        #�� ���ϰ� �ִ� ����(False), work_t: �۾��� �Ϸ���� �ɸ��� �ð�
        self.status = False
        self.work_t = 0
        self.through_a = 0

        self.complete = 0

    def work(self):
        #���� �ϰ� ���� �ʴٸ�, �۾� ����
        if self.status == False:
            self.complete = 0
            self.work_t = np.random.normal(self.mean, self.std)
            self.status = True
            self.through_a += 1

        #�۾���
        self.work_t -= self.global_interval

        #�۾��� �Ϸ�� ���, �� �ϰ� ���� �ʴ� ���·� ����    
        if self.work_t <= 0:
            self.status = False
            self.complete += self.through_a
            self.through_a -= 1

            
            

class Group_B:
    def __init__(self, global_interval, mean, std):
        # �۾��ð��� ���,ǥ������
        self.b2 = [mean[0], std[0]]
        self.b3 = [mean[1], std[1]]
        self.b4 = [mean[2], std[2]]
        self.b5 = [mean[3], std[3]]
        self.b6 = [mean[4], std[4]]

        #�� ���ϰ� �ִ� ����(False)
        self.status = False
        self.work_t = 0
        self.through_b = 0
        self.complete = 0
        self.global_interval = global_interval

    def work(self, after_a):
        if self.status == False:
            self.complete = 0
            if after_a == 2:
                self.work_t = np.random.normal(*self.b2)
                self.through_b = 2

            elif after_a == 3:
                self.work_t = np.random.normal(*self.b3)
                self.through_b = 3

            elif after_a == 4:
                self.work_t = np.random.normal(*self.b4)
                self.through_b = 4

            elif after_a == 5:
                self.work_t = np.random.normal(*self.b5)
                self.through_b = 5

            elif after_a >= 6:
                self.work_t = np.random.normal(*self.b6)
                self.through_b = 6

            after_a -=self.through_b
            self.status = True


        #�۾���
        self.work_t -= self.global_interval

        #�۾��� �Ϸ�� ���, �� �ϰ� ���� �ʴ� ���·� ����    
        if self.work_t <= 0:
            self.status = False

            self.complete += self.through_b
            self.through_b = 0

        return after_a

class Group_C:
    def __init__(self, name, global_interval, mean, std):
        self.name = name
        self.mean = mean
        self.std = std
        self.global_interval = global_interval

        #�� ���ϰ� �ִ� ����(False)
        self.status = False
        self.work_t = 0
        self.through_c = 0
        self.complete = 0
        

    def work(self, after_b):
        #���� �ϰ� ���� �ʴٸ�, �۾� ����
        if self.status == False:
            self.complete = 0

            if after_b >= 1:
                #self.work_t: �ش� �۾��� �Ϸ��ϴµ� �ɸ��� �ð�
                self.work_t = np.random.normal(self.mean, self.std)
                self.through_c = 1

                after_b -= self.through_c
                self.status = True

        #�۾���
        if self.status == True:
            self.work_t -= self.global_interval

        #�۾��� �Ϸ�� ���, �� �ϰ� ���� �ʴ� ���·� ����    
        if self.status == True and self.work_t <= 0:
            self.complete = 1

            self.status = False
            self.through_c = 0

        return after_b

def process(process_time, global_interval, A_list, b, C_list):
    global_time = 0
    after_a = 0
    through_b, after_b = 0, 0
    through_c, after_c_box = 0, 0
    
    sum_a, sum_b = 0, 0
    
    # Lists to store values for plotting
    time_points = []
    after_a_values = []
    through_b_values = []
    after_b_values = []
    through_c_values = []
    after_c_box_values = []
    sum_b_values = []
    sum_a_values = []

    while global_time <= process_time:
        global_time += global_interval

        # Track time
        time_points.append(global_time)

        # A ����
        for i in range(len(A_list) - 1):
            A_list[i].work()
            after_a += A_list[i].complete
            sum_a += A_list[i].complete

        # B ����
        after_a = b.work(after_a)
        through_b += b.through_b
        after_b += b.complete
        sum_b += b.complete

        # C ����
        for i in range(len(C_list)):
            after_b = C_list[i].work(after_b)
            after_c_box += C_list[i].complete
            through_c += C_list[i].through_c

        # Store values for plotting
        after_a_values.append(after_a)
        through_b_values.append(through_b)
        after_b_values.append(after_b)
        through_c_values.append(through_c)
        after_c_box_values.append(after_c_box)
        sum_b_values.append(sum_b)
        sum_a_values.append(sum_a)
        through_b, through_c = 0, 0

    #print(f'�� �۾��Ϸ�� �ڽ� ��: {after_c_box/50}')
    print(f'�� �۾��Ϸ�� �ڽ� ��: {after_c_box}')

    return (time_points, after_a_values, through_b_values, after_b_values, through_c_values, after_c_box_values, sum_b_values, sum_a_values)
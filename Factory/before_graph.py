# -*- coding: cp949 -*-
import numpy as np
import matplotlib.pyplot as plt
from worker import Group_A, Group_B, Group_C

#A,B,C �������� �����Ǹ�, �۾� �Ϸ� ������ ���⹰�� ������ ��
global global_interval, after_a, through_b, after_b, through_c, after_c_box
"""
after_a: A ���� �� ���� ���⹰��
through_b: B �������� �۾����� �ڽ��� ��
after_b: B ���� �� ���� ���⹰��
through_c: C�������� �۾� ���� �ڽ��� ��
after_c_box: ���� �ڽ� ��

sum_b,sum_c: �� �۾��� 
"""
global_time, global_interval = 0, 0.1
after_a = 0
through_b, after_b = 0, 0
through_c, after_c_box, after_c_plt = 0, 0, 0

sum_a,sum_b = 0, 0

#A ���� ����(�� 4���� �۾�, ���� ���α׷��� ���ȿ��� a4�� ��� ����)
A_list = []
names = ['a1', 'a2', 'ace', 'a3', 'a4']
A_means = [38.696, 38.958, 31.723, 38.83, 31.866]
A_std = [4.457, 1.974, 19.018, 19.018, 4.242]
for name, mean, std in zip(names, A_means, A_std):
    A_list.append(Group_A(name, mean, std, global_interval))

#B ���� ����
b = Group_B(global_interval)

#C ���� ����
c1 = Group_C('c1', global_interval)
c2 = Group_C('c2', global_interval)
C_list = [c1, c2]

# ���� �尡��
while global_time <= 3000:
    global_time += global_interval

    # A����
    for i in range(len(A_list)-1):
        A_list[i].work()
        after_a += A_list[i].complete
        sum_a += A_list[i].complete

    # B����
    after_a = b.work(after_a)
    through_b +=  b.through_b
    after_b += b.complete
    sum_b += b.complete

    # C����
    for i in range(len(C_list)):
        after_b =  C_list[i].work(after_b)
        after_c_box +=  C_list[i].complete
        through_c +=  C_list[i].through_c


    through_b, through_c = 0,0

print(f'�� �۾��Ϸ�� �ȷ�Ʈ: {after_c_box/50}')
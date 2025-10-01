# -*- coding: cp949 -*-
import numpy as np
import matplotlib.pyplot as plt
from worker import Group_A, Group_B, Group_C

#A,B,C 공정으로 구성되며, 작업 완료 이후의 산출물만 보고자 함
global global_interval, after_a, through_b, after_b, through_c, after_c_box
"""
after_a: A 공정 후 나온 산출물들
through_b: B 공정에서 작업중인 박스의 수
after_b: B 공정 후 나온 산출물들
through_c: C공정에서 작업 중인 박스의 수
after_c_box: 최종 박스 수

sum_b,sum_c: 총 작업량 
"""
global_time, global_interval = 0, 0.1
after_a = 0
through_b, after_b = 0, 0
through_c, after_c_box, after_c_plt = 0, 0, 0

sum_a,sum_b = 0, 0

#A 공정 선언(총 4명이 작업, 따라서 프로그래밍 동안에는 a4는 사용 안함)
A_list = []
names = ['a1', 'a2', 'ace', 'a3', 'a4']
A_means = [38.696, 38.958, 31.723, 38.83, 31.866]
A_std = [4.457, 1.974, 19.018, 19.018, 4.242]
for name, mean, std in zip(names, A_means, A_std):
    A_list.append(Group_A(name, mean, std, global_interval))

#B 공정 선언
b = Group_B(global_interval)

#C 공정 선언
c1 = Group_C('c1', global_interval)
c2 = Group_C('c2', global_interval)
C_list = [c1, c2]

# 공장 드가자
while global_time <= 3000:
    global_time += global_interval

    # A공정
    for i in range(len(A_list)-1):
        A_list[i].work()
        after_a += A_list[i].complete
        sum_a += A_list[i].complete

    # B공정
    after_a = b.work(after_a)
    through_b +=  b.through_b
    after_b += b.complete
    sum_b += b.complete

    # C공정
    for i in range(len(C_list)):
        after_b =  C_list[i].work(after_b)
        after_c_box +=  C_list[i].complete
        through_c +=  C_list[i].through_c


    through_b, through_c = 0,0

print(f'총 작업완료된 팔레트: {after_c_box/50}')
# -*- coding: cp949 -*-

import random 

 
#푸레시 오렌지 1 리터 67,395   33521.61693

#푸레시 오렌지/망고 1 리터 42,162  21002.40304

#푸레시 오렌지/C-파워 1 리터 11,361  6522.732603
 
#푸레시 오렌지 PET 118,084   68838.57632
 
#푸레시 오렌지/망고 PET 51,040  30674.10194

#푸레시 오렌지/C-파워 PET 17,741  11856.06787

day = 180

#수요 정보 목표 진부화율
demand = 67395
sigma = 33521.61693

#설정
utong = 0.785
interval = 10
safety_inventory = 2
#
parameter_a = 0
if interval ==7:
    parameter_a = 10
elif interval ==10:
    parameter_a = 5

#추가 계산
op_amount = (day/7)/(day/interval)
op_amounts = op_amount + safety_inventory/parameter_a

 
#
parameter_a = 0
if interval ==7:
    parameter_a = 10
elif interval ==10:
    parameter_a = 5

# 시뮬레이션 정보
time = 1
sum_zinbu = 0


#고객서비스 라인 품목 기준계산
sum_service_level = 0


for _ in range(time):

    # 생산 List(주문 가능 양, 유통기한, 사용 중 = 0, 사용완료 = 1)
    factory = []
    #총생산량, 진부화된 양, 주문 못맞춘 양
    s_factory = 0
    trash_factory = 0
    false_factory = 0
    
    order_q = 0
    to_make = 0

    service_num = 0 

    for day in range(140):   
    
        #10일 지날 때마다 새로운 생산 추가
        if day%interval == 0:
            to_make = 1

        if to_make == 1 and day%7 <=4:
            s = random.normalvariate(op_amounts,op_amounts/3)* demand
            factory.append([s, 1, 0])
            s_factory += s

            to_make = 0
    
        #7일 지날때마다 주문량 추가
        if day%7 == 0:
            #주문량
            order_q += random.normalvariate(demand, sigma)
    
        #사용가능한지 모든 생산 보기
        #print(f'day: {day}, 주문량: {order_q}')
        for num in range(len(factory)):

            service = False

            #사용 가능인 경우
            #print(f'{num}번째 생산 확인 중: {factory[num]}')
            if factory[num][2] == 0 and order_q>0:
                
                #유통기한이 남은 경우
                if factory[num][1] > utong:
            
                    #주문량만큼 생산량에서 제거
                    factory[num][0] = factory[num][0] - order_q
            
                    #근데 주문량이 생산량을 초과한 경우, 다음으로 이전
                    if factory[num][0] <0:                     
                        order_q = factory[num][0]*(-1)
                        factory[num][0] = 0
                        factory[num][2] = 1
                    else:               
                        order_q = 0
                        service_num+=1
                        continue
                        
            
                                 
                #폐기
                else:
                    trash_factory+=factory[num][0]
                    factory[num][2] = 1
            
            #사용 불가인 경우~남은 생산량x or 유통기한 지남
            else:
                pass    
    

        for k in range(len(factory)):
            factory[k][1] -= 1/140
    
    service_level = service_num
    print(f'전체 생산량: {s_factory}, 폐기된 양: {trash_factory}, 진부화율: {trash_factory/s_factory}')
    sum_zinbu += trash_factory/s_factory

print(f'average_zinbu: {sum_zinbu/time}')
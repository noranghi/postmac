import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import copy

#기지국 특성 설정
RU = 15 #기지국 15개 설정
sp = 3 #service provider 3명, 각 관리하는 기지국 5개씩 설정
region = 50 #지역 각 100m
gamma = 2
Ptx = 50
noise = 0.5
color = ['red', 'orange', 'yellow', 'green', 'blue', 'black']

#interference graph 생성
#거리 설정
#전송전력 설정
#간섭 관계 행렬 생성
#그래프 생성

random.seed(100)
lx = []
ly = []

D = np.zeros((RU, RU))
Gain = np.zeros((RU, RU))

for i in range(RU):
    x = random.randint(1, region)
    y = random.randint(1, region)
    lx.append(x)
    ly.append(y)

#거리 계산
for i in range(RU):
    for j in range(i, RU):
        D_ij = math.sqrt(math.pow(lx[i] - lx[j], 2) + math.pow(ly[i] - ly[j], 2))
        D[i][j] = D_ij
        D[j][i] = D_ij

#gain 계산
for i in range(RU):
    for j in range(i, RU):
        if D[i][j] == 0:
            Gain[i][j] = 0
        else:

            G_ij = math.pow(D[i][j], -gamma)
            Gain[i][j] = G_ij
            Gain[j][i] = G_ij

Prx = Gain * Ptx

#간섭 값 확인


def show_location(RU, sp, lx, ly):
    #위치 그래프
    num = int(RU//sp)
    for j in range(1, sp+1): #1, 2, 3,
        plt.scatter(lx[num * (j-1):num*j], ly[num * (j-1):num*j], label='sp {}'.format(j), color = color[j-1] , s = 200)
    plt.title('RUs Location of each service providers')
    plt.legend(loc = 'upper right')
    plt.show()

show_location(RU, sp, lx, ly)

#만약 전체 총 전력이 0.5이 넘으면 무조건 연결
def check_interference(RU, Prx):
    I = np.zeros((RU, RU))
    for i in range(RU):
        cnt = 0
        except_point = []
        for j in range(RU):
            if j == i:
                I[i][j] == 0
            else:
                if Prx[i][j] >= 1:
                    I[i][j] = 1
                else:
                    cnt += Prx[i][j]
                    except_point.append(j)
        if cnt >= 1:
            for k in except_point:
                I[i][k] = 2
        else:
            for k in except_point:
                I[i][k] = 0
    return I
#interference relationship graph
I_map= check_interference(RU, Prx)
print(I_map)

#make result map
def make_graph(RU, sp, I_map, lx, ly, colored):
    I_graph = nx.Graph()

    for i in range(RU):
        I_graph.add_node(i)
    for i in range(RU):
        for j in range(i, RU):
            if I_map[i][j] == 1:
                I_graph.add_edge(i, j, weight = 1)

            elif I_map[i][j] == 2:
                I_graph.add_edge(i, j, weight = 2)

    pos = {}
    for i in range(RU):
        pos.update({i:[lx[i],ly[i]]})

    must = [(u, v) for (u, v, d) in I_graph.edges(data=True) if d['weight'] == 1]
    can = [(u, v) for (u, v, d) in I_graph.edges(data=True) if d['weight'] == 2]

    color = []
    num = RU // sp
    for i in range(sp):
        for j in range(num*(i-1), num*i):
            color.append(colored[i])



    nx.draw_networkx_nodes(I_graph, pos, node_color = color)
    nx.draw_networkx_labels(I_graph, pos)
    nx.draw_networkx_edges(I_graph ,pos, edgelist= must)
    nx.draw_networkx_edges(I_graph, pos, edgelist= can, edge_color = 'b', style = 'dotted')

    #legend_labels = ['sp_1', 'sp_2', 'sp_3'] # Assuming sp values and corresponding labels
    plt.title('Interference graph')
    #plt.legend(legend_labels, loc='upper right', title='service provider')

    # Show the plot
    plt.show()

make_graph(RU, sp, I_map, lx, ly, color)



random.seed(100)
# service provider가 제공할 가격
contract_price = []
for i in range(sp):
    contract_price.append(int(random.uniform(10, 20)) * 1000)
print(contract_price)

expect = []
for i in range(RU):
    expect.append(random.randint(20, 30))

# 순서 정하기
# 1. 가장 많이 가격을 제시한 sp의 추가 score가 20, 그 다음 10, 0 이러한 식으로 가산점이 붙어 할당을 시작한다.
# 2. RU들의 점수가 높은 순서대로 할당을 시작한다.
def order_score(sp, contract_price, RU, expect):
    add_score = []
    for i in range(sp):
        add_score.append(math.log2(int((contract_price[i] / sum(contract_price)) * 1000)))
    score = []
    num = int(RU // sp)

    for i in range(sp):  # 0, 1, 2
        for j in range(num * i, num * (i + 1)):
            score.append(expect[j] + add_score[i])
    same_score = copy.copy(score)

    RU_order = []

    for i in sorted(score, reverse=True):
        order = score.index(i)
        RU_order.append(order)
        score[order] = 0
    return RU_order, same_score


order_score(sp, contract_price, RU, expect)[1]


# 색 할당 하기 Welsh-Powell 전체적으로 필요할 자원 량 알기
# 확보 자원량 대비 얼마나 제공할 수 있는지
def welshPowell(sp, I_map, RU, order, expect, Prx, contract_price):
    # 그래프 노드 순서 = order
    # SAS_access : SAS가 할당해줄 수 있는 최대 개수
    colored = {i: [] for i in range(RU)}  # 각 노드들이 가지는 색상
    colors = 0  # 전체적으로 필요한 색깔
    used = []
    color_num = 0
    I = {i: 0 for i in range(RU)}  # 간섭 허용률
    # expect : 각 필요한 주파수 자원량
    # Imap : 간섭 관계 확인
    # Prx : 1넘는지 안넘는지
    SAS_resource = sum(contract_price) // 3000
    print(SAS_resource)

    for i in range(max(expect)):  # 가장 할당이 필요한 개수가 많은 친구
        color_dic = {}
        used_color = []
        if color_num <= SAS_resource:
            for node in order:
                # loc = nodes.index(node)
                if len(colored[node]) != expect[node]:  # 색깔은 온전히 다 못받았을 때
                    if len(color_dic.keys()) == 0:
                        color_dic[node] = color_num
                        used_color.append(color_num)
                        # print(color_dic)
                    else:

                        zero_conflict = []
                        one_conflict = []
                        two_conflict = []
                        node_check = list(color_dic.keys())
                        interference_Prx = []
                        for i in node_check:
                            # 0로 연결된 친구
                            if I_map[node][i] == 0:
                                zero_conflict.append(i)
                            # 1로 연결된 친구
                            elif I_map[node][i] == 1:
                                one_conflict.append(i)
                                if color_dic[i] in used_color:
                                    used_color.remove(color_dic[i])
                            # 2로 연결된 친구
                            elif I_map[node][i] == 2:
                                two_conflict.append(i)
                                interference_Prx.append(Prx[node][i])

                        if len(zero_conflict) != 0:
                            if len(used_color) == 0:
                                color_num += 1
                                used_color.append(color_num)
                                color_dic[node] = color_num
                                if color_num == SAS_resource:
                                    break;
                            else:
                                for i in zero_conflict:
                                    if color_dic[i] in used_color:
                                        color_dic[node] = color_dic[i]
                                        break;


                        elif len(two_conflict) != 0:
                            if len(used_color) == 0:
                                color_num += 1
                                used_color.append(color_num)
                                color_dic[node] = color_num
                                if color_num == SAS_resource:
                                    break;
                            else:
                                for j in two_conflict:
                                    if color_dic[j] in used_color:
                                        color_dic[node] = color_dic[i]

                        if node not in color_dic.keys():
                            color_num += 1
                            used_color.append(color_num)
                            color_dic[node] = color_num
                            if color_num == SAS_resource:
                                break;

                        check = 'can'
                        own_interference = I[node]
                        for i in color_dic.keys():
                            if color_dic[node] == color_dic[i]:
                                if I_map[node][i] == 2:
                                    own_interference += Prx[node][i]
                                    if own_interference > 1 or I[i] + Prx[node][i] > 1:
                                        check = 'notcan'

                        if check == 'notcan':
                            color_dic[node] += 1
                            color_num += 1
                            if color_num == SAS_resource:
                                break;
                        else:
                            for i in color_dic.keys():
                                if i != node and color_dic[node] == color_dic[i]:
                                    if I_map[node][i] == 2:
                                        I[node] += Prx[node][i]
                                        I[i] += Prx[node][i]

        color_num += 1

        for key, value in color_dic.items():
            colored[key] += [value]
        if color_num > SAS_resource:
            break;
    """if sum(contract_price)//3000 >= color_num:
        SAS_resource = color_num
    else:
        SAS_resource = sum(contract_price)//3000"""
    sp_needs = []  # 각 원소
    how_many = []  # 몇개
    num_list = int(RU // sp)

    for i in range(sp):
        let_needs = []
        for j in range(num_list * i, num_list * (i + 1)):
            for k in colored[j]:
                let_needs.append(k)
        sp_needs.append(list(set(let_needs)))
        how_many.append(len(let_needs))
    # print(len(sp_needs[0]))

    how_can = []  # 각 원소
    outage = []  # outage
    for i in range(sp):
        let_check = []
        for j in range(num_list * i, num_list * (i + 1)):
            for k in colored[j]:
                if k <= SAS_resource:
                    let_check.append(k)
        how_can.append(list(set(let_check)))
    # print(len(how_can[0]))
    out_RU = []
    for i in range(RU):
        outage_RU = (expect[i] - len(colored[i])) / expect[i]
        out_RU.append(outage_RU)
    for j in range(sp):
        outage.append(sum(out_RU[num_list * j:(num_list) * (j + 1)]) / num_list)

    return I, color_num, colored, SAS_resource, outage


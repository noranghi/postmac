import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

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
        for j in range(i+1, RU):
            if Prx[i][j] >= 1:
                I[i][j] = 1
                I[j][i] = 1
            else:
                cnt += Prx[i][j]
                except_point.append(j)
        if cnt >= 1:
            for k in except_point:
                I[i][k] = 2
                I[k][i] = 2
        else:
            for k in except_point:
                I[i][k] = 0
                I[k][i] = 0
    return I
#interference relationship graph
I_map= check_interference(RU, Prx)

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

#색 할당 하기 Welsh-Powell 전체적으로 필요할 자원 량 알기
def welshPowell(I_map, RU, expect, Prx):
    nodes = list(range(RU)) #그래프 노드 개수
    colored = {i : [] for i in range(RU)} #각 노드들이 가지는 색상
    I = {i : 0 for i in range(RU)} #간섭 허용률
    colors = 0 #전체적으로 필요한 색깔
    used= []
    color_num = 0
    #expect : 각 필요한 주파수 자원량
    #Imap : 간섭 관계 확인
    #Prx : 1넘는지 안넘는지
    for i in range(max(expect)): #가장 할당이 필요한 개수가 많은 친구
        color_dic = {}
        used_color = []
        for node in nodes:
            if len(colored[node]) != expect[node]: #색깔은 온전히 다 못받았을 때
                if len(color_dic.keys()) == 0:
                    color_dic[node] = color_num
                    used_color.append(color_num)
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
                        else:
                            for i in zero_conflict:
                                if color_dic[i].values() in used_color:
                                    color_dic[node] = color_dic[i].values()
                                    break;

                    elif len(two_conflict) != 0:
                        if len(used_color) == 0:
                            color_num += 1
                            used_color.append(color_num)
                            color_dic[node] = color_num
                        else:
                            for j in two_conflict:
                                if color_dic[j].values() in used_color:
                                    color_dic[node] = color_dic[i].values()


                    if node not in color_dic.keys():
                        color_num += 1
                        used_color.append(color_num)
                        color_dic[node] = color_num
        print(color_dic)
        print(used_color)



welshPowell(I_map, RU, [1, 2, 3, 1, 1, 1, 1, 1, 1, 1,1,1, 1, 1, 1], Prx)






























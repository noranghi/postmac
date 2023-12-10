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

    legend_labels = ['sp_1', 'sp_2', 'sp_3'] # Assuming sp values and corresponding labels


    # Draw the legends for the nodes
    plt.title('Interference graph')
    plt.legend(legend_labels, loc='upper right', title='service provider')

    # Show the plot
    plt.show()

make_graph(RU, sp, I_map, lx, ly, color)








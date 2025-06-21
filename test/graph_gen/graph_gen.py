"""
DAG graph generator for HEFT/PEFT scheduling simulations
Generates random task graphs with configurable parameters
"""

import numpy as np
import random as rndm
import copy
import csv


class graph_node:
    def __init__(self, task_id, level, out_count):
        self.outgoing_edge_count = out_count
        self.outgoing_edge_node = []
        self.outgoing_edge_weight = []
        self.level = level
        self.task_id = task_id
        self.resource_exe_time = []


def split_number(number, parts):
    if parts < 1:
        raise ValueError("Parts must be at least 1")
    if parts == 1:
        return [number]
    
    count_per_part = []
    while len(count_per_part) == 0:
        for i in range(parts - 1):
            tmp1 = (number - sum(count_per_part)) / (parts - i)
            while True:
                tmp2 = int(rndm.normalvariate(tmp1, tmp1/2 ))
                if tmp2 > 0:
                    count_per_part.append(tmp2)
                    break
        count_per_part.append(number - sum(count_per_part))
        
        if min(count_per_part) <= 0:
            count_per_part = []
            
    return count_per_part

def generate_graph(SEED = 2894,resource_count = 10,graph_height= 3,vertex_count = 1000,mean_outdeg = 6,comm_2_comp = 1,HF = 0.5,edge_weight_range = [1, 10],bw_range = [1, 1000]):
    # Initialize default parameters
    sd_outdeg = 80

    # Validate parameters
    if (HF < 0) or (HF > 1):
        print("0 <= (Heterogenity Factor(HF)) < 1")
        exit()

    vertex_count = vertex_count
    graph_height = graph_height

    np.random.seed(SEED)
    rndm.seed(SEED)


    nodes_list = []
    if vertex_count < graph_height:
        print("Number of nodes are smaller than graph height")
        exit()

    # Initialize resource communication bandwidth matrix
    resource_com_bw = np.zeros((resource_count, resource_count))
    cluster_size = resource_count 
    intra_cluster_bw = (100, 1000)      
    
    
    for i in range(resource_count):
        for j in range(i + 1):
            if i == j:
                continue

            bw = rndm.randint(intra_cluster_bw[0], intra_cluster_bw[1])
            resource_com_bw[i][j] = bw
            resource_com_bw[j][i] = bw

    node_count_per_level = [1]
    node_count_per_level.extend(split_number(vertex_count - 2, graph_height - 2))
    node_count_per_level.extend([1])

    level_nodes_list = []
    count = 0

    # Create nodes and assign to levels
    for level in range(len(node_count_per_level)):
        tmp1 = []
        elem = node_count_per_level[level]
        out_edge_count_to_next_level = []
        
        if level != len(node_count_per_level) - 1:
            if elem > node_count_per_level[level + 1]:
                out_edge_count_to_next_level = split_number(elem, elem)
            else:
                out_edge_count_to_next_level = split_number(node_count_per_level[level + 1], elem)
        else:
            out_edge_count_to_next_level = list(np.zeros((elem)))
            
        for i in range(elem):
            tmp1.extend([graph_node(count, level, int(out_edge_count_to_next_level[i]))])
            count = count + 1
        
        for elem1 in tmp1:
            nodes_list.extend([elem1])
        level_nodes_list.append(tmp1)

    # Connect nodes between adjacent levels
    tmp = 0
    while tmp != graph_height:
        l1 = []
        l0 = []

        for elem in nodes_list:
            if elem.level == tmp + 1:
                l1.extend([elem.task_id])
            if elem.level == tmp:
                l0.extend([elem.task_id])
        
        l1_tmp = copy.deepcopy(l1)
        for elem in l0:
            for elem1 in range(nodes_list[elem].outgoing_edge_count):
                tmp1 = rndm.choice(l1_tmp)
                nodes_list[elem].outgoing_edge_node.extend([tmp1])
                nodes_list[elem].outgoing_edge_weight.extend(
                    [rndm.randint(edge_weight_range[0], edge_weight_range[1])])
                l1_tmp.remove(tmp1)
                if len(l1_tmp) == 0:
                    l1_tmp = copy.deepcopy(l1)
        tmp = tmp + 1

    # Add cross-level edges
    for elem in nodes_list:
        if ((elem.level == (graph_height - 1)) or 
            (elem.level == (graph_height - 2)) or 
            (elem.level == 0)):
            continue
        
        current_node_level = elem.level
        remaining_nodes = 0
        l1 = []
        
        for elem1 in nodes_list:
            if elem1.level > elem.level:
                l1.extend([elem1.task_id])
        
        for elem1 in elem.outgoing_edge_node:
            l1.remove(elem1)
        
        remaining_nodes = len(l1)
        tmp1 = 0
        
        while (tmp1 <= 0) or (tmp1 > remaining_nodes):
            tmp1 = int(np.random.normal(mean_outdeg, sd_outdeg))
        
        if (elem.outgoing_edge_count >= tmp1):
            continue
        
        new_nodes_to_connect = []
        for i in range(tmp1 - elem.outgoing_edge_count):
            tmp2 = rndm.choice(l1)
            l1.remove(tmp2)
            new_nodes_to_connect.extend([tmp2])
        
        elem.outgoing_edge_count = elem.outgoing_edge_count + len(new_nodes_to_connect)
        for elem1 in new_nodes_to_connect:
            elem.outgoing_edge_node.extend([elem1])
            elem.outgoing_edge_weight.extend([rndm.randint(1, 100)])

    # Extract link bandwidths
    link_bw = []
    for i in range(len(resource_com_bw)):
        for j in range(len(resource_com_bw[i])):
            if (i == j):
                break
            else:
                link_bw.extend([resource_com_bw[i][j]])

    # Assign computation times
    for elem in nodes_list:
        max_weight = -1
        if elem.outgoing_edge_count == 0:
            average_com_time = float(rndm.randint(edge_weight_range[0], edge_weight_range[1])) / float(
                rndm.randint(bw_range[0], bw_range[1]))
        else:
            for tmp in elem.outgoing_edge_weight:
                if tmp > max_weight:
                    max_weight = tmp
            
            com_time = [float(max_weight) / float(bw) for bw in link_bw]
            average_com_time = sum(com_time) / float(len(com_time))
        
        mean_comp_time = float(average_com_time) / float(comm_2_comp)
        resource_count = len(resource_com_bw)
        exe_time = []
        
        if HF == 0:
            exe_time = [mean_comp_time for et in range(resource_count)]
        else:
            lb = mean_comp_time - (HF * mean_comp_time)
            ub = mean_comp_time + (HF * mean_comp_time)
            exe_time = np.random.uniform(lb, ub, resource_count)
        
        elem.resource_exe_time = exe_time

    # Build connectivity matrix
    connect_matrix = np.zeros((len(nodes_list), len(nodes_list)))
    for elem in nodes_list:
        for elem1 in range(len(elem.outgoing_edge_node)):
            connect_matrix[elem.task_id][elem.outgoing_edge_node[elem1]] = elem.outgoing_edge_weight[elem1]

    # Write output files
    # Resource bandwidth
    resource_count = len(resource_com_bw)
    tmp = ["P_" + str(i) for i in range(resource_count)]
    tmp.insert(0, "P")
    write_data = [tmp]

    for i in range(resource_count):
        tmp = ["P_" + str(i)]
        for j in range(resource_count):
            tmp.extend([resource_com_bw[i][j]])
        write_data.append(tmp)

    with open("resource_BW.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(write_data)

    # Task connectivity
    task_count = len(connect_matrix)
    tmp = ["T_" + str(i) for i in range(task_count)]
    tmp.insert(0, "T")
    write_data = [tmp]

    for i in range(task_count):
        tmp = ["T_" + str(i)]
        for j in range(task_count):
            tmp.extend([connect_matrix[i][j]])
        write_data.append(tmp)

    with open("task_connectivity.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(write_data)

    # Execution times
    resource_count = len(resource_com_bw)
    tmp = ["P_" + str(i) for i in range(resource_count)]
    tmp.insert(0, "TP")
    write_data = [tmp]

    task_count = len(connect_matrix)
    for i in range(len(nodes_list)):
        tmp = ["T_" + str(i)]
        for j in range(resource_count):
            tmp.extend([nodes_list[i].resource_exe_time[j]])
        write_data.append(tmp)

    with open("task_exe_time.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(write_data)
if __name__ == "__main__":
    with open("graph.config","r") as f:
        config = f.readlines()
    config = [(x.strip()).split() for x in config]
    #print(config)
    for x in config:
        #print(x)
        if len(x) == 0:
            continue
        if    x[0] == "RC": resource_count = int(x[1])
        elif  x[0] == "GH": graph_height = int(x[1])
        elif  x[0] == "TC": vertex_count = int(x[1])
        elif  x[0] == "AOD": mean_outdeg = int(x[1])
        elif  x[0] == "CCR": comm_2_comp = float(x[1])
        elif  x[0] == "HF": HF = float(x[1])
        elif  x[0] == "CDR": edge_weight_range = [float(x[1]), float(x[2])]
        elif  x[0] == "LBW": bw_range = [float(x[1]), float(x[2])]
        elif  x[0] == "SEED": SEED = int(x[1])
        
    generate_graph(
        SEED=SEED,
        resource_count=resource_count,
        graph_height=graph_height,
        vertex_count=vertex_count,
        mean_outdeg=mean_outdeg,
        comm_2_comp=comm_2_comp,
        HF=HF,
        edge_weight_range=edge_weight_range,
        bw_range=bw_range
    )
	

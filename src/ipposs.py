from collections import deque, namedtuple
from math import inf
from src.gantt import showGanttChart
from types import SimpleNamespace
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
logger = logging.getLogger('ipposs')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')
def schedule_dag(dag, computation_matrix, communication_matrix, proc_schedules=None, time_offset=0, relabel_nodes=True, **kwargs):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs
    给定一个DAG和一组指定PE带宽和（任务，PE）执行时间的矩阵，计算HEFT调度
    将该DAG添加到该组PE上
    """
    if proc_schedules == None:
        proc_schedules = {}
    optimistic_cost_table = {}
    improved_predict_cost_matrix = {}
    _self = {
        'computation_matrix': computation_matrix,
        'communication_matrix': communication_matrix,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'numExistingJobs': 0,
        'time_offset': time_offset,
        'root_node': None,
        'optimistic_cost_table':optimistic_cost_table,
        'improved_predict_cost_matrix':improved_predict_cost_matrix
    }
    _self = SimpleNamespace(**_self)
    
    num_processors = len(_self.computation_matrix[0]) if _self.computation_matrix is not None else 0 
    for task in dag.nodes():
        _self.optimistic_cost_table[task] = [0] * num_processors  
        _self.improved_predict_cost_matrix[task] = [0] * num_processors
    for proc in proc_schedules:
        _self.numExistingJobs = _self.numExistingJobs + len(proc_schedules[proc])

    if relabel_nodes:
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+_self.numExistingJobs), list(dag.nodes()))))
    else:
        #Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

    for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
        _self.task_schedules[i] = None
    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []
    

    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event


    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    _self.root_node = root_node

    logger.debug(""); logger.debug("====================== Performing Rank-U Computation ======================\n"); logger.debug("")
    _compute_ranku(_self, dag, **kwargs)
    
    for node in dag.nodes():  
        dag.nodes[node]['ranku'] = sum(_self.improved_predict_cost_matrix[node]) / len(_self.communication_matrix)

    logger.debug(""); logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); logger.debug("")
    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)
    if sorted_nodes[0] != root_node:
        logger.debug("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
        idx = sorted_nodes.index(root_node)
        sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
    task_queue = []
    entry_task = sorted_nodes[0]  
    heapq.heappush(task_queue, (-dag.nodes[entry_task]['ranku'], entry_task))    
    scheduled_tasks = set()
    while task_queue:
        _, node = heapq.heappop(task_queue)
        if _self.task_schedules[node] is not None:
            continue
        scheduled_tasks.add(node)

        minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
        
        for proc in range(len(communication_matrix)):
            taskschedule = _compute_eft(_self, dag, node, proc)
            if (taskschedule.end+_self.optimistic_cost_table[node][taskschedule.proc] < minTaskSchedule.end+_self.optimistic_cost_table[node][minTaskSchedule.proc]):
                minTaskSchedule = taskschedule
        
        _self.task_schedules[node] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: (schedule_event.end, schedule_event.start))
        for successor in dag.successors(node):
            all_predecessors_scheduled = all(pred in scheduled_tasks for pred in dag.predecessors(successor))
            if all_predecessors_scheduled:
                heapq.heappush(task_queue, (-dag.nodes[successor]['ranku'], successor))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('\n')
            for proc, jobs in _self.proc_schedules.items():
                logger.debug(f"Processor {proc} has the following jobs:")
                logger.debug(f"\t{jobs}")
            logger.debug('\n')
        for proc in range(len(_self.proc_schedules)):
            for job in range(len(_self.proc_schedules[proc])-1):
                first_job = _self.proc_schedules[proc][job]
                second_job = _self.proc_schedules[proc][job+1]
                assert first_job.end <= second_job.start, \
                f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}" 
    return _self.proc_schedules, _self.task_schedules
def _compute_ranku(_self, dag, **kwargs):
    """
    Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
    """
    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    terminal_node = terminal_node[0]

    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
    np.fill_diagonal(diagonal_mask, 0)
    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask])
    for edge in dag.edges():
        logger.debug(f"Assigning {edge}'s average weight based on average communication cost. {float(dag.get_edge_data(*edge)['weight'])} => {float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost}")
        nx.set_edge_attributes(dag, { edge: float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost }, 'avgweight')

    # Utilize a masked array so that np.mean, etc, calculations ignore the entries that are inf
    comp_matrix_masked = np.ma.masked_where(_self.computation_matrix == inf, _self.computation_matrix)

    nx.set_node_attributes(dag, { terminal_node: np.mean(comp_matrix_masked[terminal_node-_self.numExistingJobs]) }, "ranku")
    visit_queue = deque(dag.predecessors(terminal_node))
    _self.improved_predict_cost_matrix[terminal_node] = _self.computation_matrix[terminal_node]

    while visit_queue:
        node = visit_queue.pop()
        while _node_can_be_processed(_self, dag, node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        logger.debug(f"Assigning ranku for node: {node}")
        max_successor_ranku = -1
        for succnode in dag.successors(node):
            logger.debug(f"\tLooking at successor node: {succnode}")
            logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {dag[node][succnode]['avgweight']}, and the ranku for node {node} is {dag.nodes()[succnode]['ranku']}")
            val = float(dag[node][succnode]['avgweight']) + dag.nodes()[succnode]['ranku']
            if val > max_successor_ranku:
                max_successor_ranku = val
        assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"
        nx.set_node_attributes(dag, { node: np.mean(comp_matrix_masked[node-_self.numExistingJobs]) + max_successor_ranku }, "ranku")

        for proc in range(len(_self.communication_matrix)):
            for succnode in dag.successors(node):
                optimistic_cost_ = inf
                for preproc in range(len(_self.communication_matrix)):
                    if preproc != proc:
                        val = _self.optimistic_cost_table[succnode][preproc]+dag[node][succnode]['avgweight']+_self.computation_matrix[succnode][preproc]
                    else: 
                        val = _self.optimistic_cost_table[succnode][proc]+_self.computation_matrix[succnode][preproc]
                    if val < optimistic_cost_ :
                        optimistic_cost_ = val
                if optimistic_cost_ > _self.optimistic_cost_table[node][proc]:
                    _self.optimistic_cost_table[node][proc] = optimistic_cost_

        for proc in range(len(_self.communication_matrix)):
            _self.improved_predict_cost_matrix[node][proc] = _self.computation_matrix[node][proc]
            max_optimistic_cost = 0
            for succnode in dag.successors(node):
                optimistic_cost_ = inf
                for preproc in range(len(_self.communication_matrix)):
                    if preproc != proc:
                        val = _self.improved_predict_cost_matrix[succnode][preproc]+dag[node][succnode]['avgweight']+_self.computation_matrix[succnode][preproc] 
                    else: 
                        val = _self.improved_predict_cost_matrix[succnode][proc]+_self.computation_matrix[succnode][preproc]
                    if val < optimistic_cost_ :
                        optimistic_cost_ = val
                if optimistic_cost_ >max_optimistic_cost:
                    _self.improved_predict_cost_matrix[node][proc] = optimistic_cost_+_self.computation_matrix[node][proc]
                    max_optimistic_cost = optimistic_cost_
                   
        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])
    
    logger.debug("")
    for node in dag.nodes():
        logger.debug(f"Node: {node}, Rank U: {dag.nodes()[node]['ranku']}")
def _node_can_be_processed(_self, dag, node):
    """
    Validates that a node is able to be processed in Rank U calculations. Namely, that all of its successors have their Rank U values properly assigned
    Otherwise, errors can occur in processing DAGs of the form
    A
    |\
    | B
    |/
    C
    Where C enqueues A and B, A is popped off, and it is unable to be processed because B's Rank U has not been computed
    """
    for succnode in dag.successors(node):
        if 'ranku' not in dag.nodes()[succnode]:
            logger.debug(f"Attempted to compute the Rank U for node {node} but found that it has an unprocessed successor {dag.nodes()[succnode]}. Will try with the next node in the queue")
            return False
    return True
def _compute_eft(_self, dag, node, proc):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    logger.debug(f"Computing EFT for node {node} on processor {proc}")
    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
        if _self.communication_matrix[predjob.proc, proc] == 0:
            ready_time_t = predjob.end
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc] 
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    computation_time = _self.computation_matrix[node-_self.numExistingJobs, proc]
    job_list = _self.proc_schedules[proc]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time > 0:
                logger.debug(f"Found an insertion slot before the first job {prev_job} on processor {proc}")
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start+computation_time, proc)
                break
        if idx == len(job_list)-1:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx+1]
        #Start of next job - computation time == latest we can start in this window
        #Max(ready_time, previous job's end) == earliest we can start in this window
        #If there's space in there, schedule in it
        logger.debug(f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start - max(ready_time, prev_job.end)}")
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            logger.debug(f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end}, {next_job.start}]")
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
    logger.debug(f"\tFor node {node} on processor {proc}, the EFT is {min_schedule}")

    # 返回一个 ScheduleEvent 对象，该对象表示当前任务在指定处理器上的调度信息，包括开始时间、结束时间和处理器编号
    return min_schedule    
def readCsvToNumpyMatrix(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = list(filter(lambda arr: arr != [''], contentsList))
        
        matrix = np.array(contentsList)
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix

def readDagMatrix(dag_file, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix = readCsvToNumpyMatrix(dag_file)
    if matrix.shape[0]!=matrix.shape[1]:
        raise ValueError("Input matrix is not square")
    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    if show_dag:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file", 
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Topcuoglu 2002 if none given.", 
                        type=str, default="../test/canonicalgraph_task_connectivity.csv")
    parser.add_argument("-p", "--pe_connectivity_file", 
                        help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Topcuoglu 2002 if none given. If communication startup costs (L) are needed, a \"Startup\" row can be used as the last CSV row", 
                        type=str, default="../test/canonicalgraph_resource_BW.csv")
    parser.add_argument("-t", "--task_execution_file", 
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Topcuoglu 2002 if none given.", 
                        type=str, default="../test/canonicalgraph_task_exe_time.csv")
    parser.add_argument("-l", "--loglevel", 
                        help="The log level to be used in this module. Default: INFO", 
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--showDAG", 
                        help="Switch used to enable display of the incoming task DAG", 
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt", 
                        help="Switch used to enable display of the final scheduled Gantt chart", 
                        dest="showGantt", action="store_true")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()
    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    communication_matrix = readCsvToNumpyMatrix(args.pe_connectivity_file)
    computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)
    dag = readDagMatrix(args.dag_file, args.showDAG) 
    processor_schedules, task_schedules = schedule_dag(dag, communication_matrix=communication_matrix,  computation_matrix=computation_matrix)
    for proc, jobs in processor_schedules.items():
        logger.info(f"Processor {proc} has the following jobs:")
        logger.info(f"\t{jobs}")
    if args.showGantt:
        showGanttChart(processor_schedules)
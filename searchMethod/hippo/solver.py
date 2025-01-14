import itertools
import numpy as np
import pickle as pkl
import time
from pyomo.environ import *


def generate_example(num_tasks=5, num_configs=4, seed=None):
    """
    Generate a complex example with random parameters for the integer programming problem.

    :param num_tasks: Number of tasks to process.
    :param num_configs: Number of configurations available per task.
    :param seed: Random seed for reproducibility.
    :return: A tuple of (accuracy_matrix, resource_matrix, cpu_time_constraint, memory_constraint).
    """
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility

    # 随机生成准确性矩阵
    accuracy_matrix = np.random.uniform(
        low=0.6, high=1.0, size=(num_tasks, num_configs))

    # 随机生成资源矩阵，其中CPU时间和内存使用量
    memory_matrix = np.random.randint(
        low=15, high=50, size=(num_tasks, num_configs))

    # 设置资源约束，确保至少有一组解。这里我们采用简化的方法，将约束设置为资源矩阵中的最小值和最大值的平均
    memory_constraint = np.mean(memory_matrix) * num_tasks

    return accuracy_matrix, memory_matrix, memory_constraint


def solve_integer_programming_with_nonlinear_constraints(accuracy_matrix, resource_matrix,
                                                         resource_constraint_value, max_pool_length):
    """
    Solve the integer programming problem with nonlinear resource constraints using Pyomo.

    :param accuracy_matrix: A matrix of shape (N, M) containing accuracy scores.
    :param resource_matrix: A matrix of shape (N, M) containing resource consumptions.
    :param resource_constraint: A function that takes a list of resource consumptions and returns a boolean
                                 indicating if the constraints are satisfied.
    :return: A list of selected configurations for each data point.
    """
    num_data, num_configs = accuracy_matrix.shape
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(num_data), range(num_configs), within=Binary)

    # Objective function
    def accuracy_rule(model):
        return sum(accuracy_matrix[i][j] * model.x[i, j] for i in range(num_data) for j in range(num_configs))
    model.total_accuracy = Objective(rule=accuracy_rule, sense=maximize)

    # Constraint: Only one configuration can be selected per data point
    def config_selection_rule(model, i):
        return sum(model.x[i, j] for j in range(num_configs)) == 1
    model.config_selection = Constraint(
        range(num_data), rule=config_selection_rule)

    # Nonlinear resource constraints
    # This is a placeholder. Actual implementation depends on the specifics of the resource_constraints function
    # and may require reformulating the function or using Pyomo's support for external functions.
    def resource_constraint_rule(model):
        resources = [sum(resource_matrix[v][c] * model.x[v, c]
                         for c in range(num_configs)) for v in range(num_data)]
        resources = sum(resources) / num_data
        real_resource = rfunc(resources, num_data / max_pool_length)
        return real_resource <= resource_constraint_value
    model.resource_constraint = Constraint(rule=resource_constraint_rule)

    # Solve the problem
    # Using IPOPT solver as an example, suitable for nonlinear problems
    solver = SolverFactory('glpk')
    solver.solve(model)

    # Extract the solution
    solution = [None] * num_data
    for i in range(num_data):
        for j in range(num_configs):
            if value(model.x[i, j]) >= 0.5:  # Assuming binary decision variable
                solution[i] = j
                break

    return solution


def solve_traverse(accuracy_matrix, resource_matrix,
                   resource_constraint_value, max_pool_length,
                   gpu_constrain=None):
    start_t = time.time()
    time_flag=False
    if gpu_constrain is not None:
        ClusetrParetoSet, indexs, gpu_resource_info, gpu_memory_bound = gpu_constrain

    cluster_number, config_number = accuracy_matrix.shape
    video_ids, config_ids = list(
        range(cluster_number)), list(range(config_number))
    combinations = list(itertools.product(config_ids, repeat=len(video_ids))) 
    constrainted_combinations = []
    for combination in combinations:
        resources = sum([resource_matrix[i][combination[i]]
                        for i in range(len(combination))])
        combination_resource = rfunc(resources / cluster_number,
                                     cluster_number / max_pool_length)

        if gpu_constrain is not None:
            gpu_memory = []
            for cluster_label, config_indice in enumerate(combination):
                if config_indice >= len(ClusetrParetoSet[cluster_label]):
                    gpu_memory.append(1000000)
                    break
                config_action = ClusetrParetoSet[cluster_label][config_indice][2]
                config_action_str = "_".join(
                    [str(config_action[index]) for index in indexs])
                gpu_memory.append(gpu_resource_info[config_action_str])
            if sum(gpu_memory) > gpu_memory_bound:
                continue
        if combination_resource <= resource_constraint_value:
            constrainted_combinations.append((combination,combination_resource))

    # if len(constrainted_combinations) == 0:
    #     raise ValueError("No solution found.")
    
    
    if len(constrainted_combinations) == 0:
        print("time_over")
        time_flag=True
        for combination in combinations:
            resources = sum([resource_matrix[i][combination[i]]
                            for i in range(len(combination))])
            combination_resource = rfunc(resources / cluster_number,
                                        cluster_number / max_pool_length)

            if gpu_constrain is not None:
                gpu_memory = []
                for cluster_label, config_indice in enumerate(combination):
                    if config_indice >= len(ClusetrParetoSet[cluster_label]):
                        gpu_memory.append(1000000)
                        break
                    config_action = ClusetrParetoSet[cluster_label][config_indice][2]
                    config_action_str = "_".join(
                        [str(config_action[index]) for index in indexs])
                    gpu_memory.append(gpu_resource_info[config_action_str])
                if sum(gpu_memory) > gpu_memory_bound:
                    continue

            #if combination_resource <= resource_constraint_value:
                constrainted_combinations.append((combination,combination_resource))

    constrainted_combination_with_accuracies = []
    for combination,resource in constrainted_combinations:
        accuracy = 0.0
        for i in range(len(combination)):
            accuracy += accuracy_matrix[i][combination[i]]
        if accuracy < 40:
            continue
        constrainted_combination_with_accuracies.append(
            (combination, accuracy,resource))

    # desired_rank = 1
    # sorted_combinations = sorted(
    # constrainted_combination_with_accuracies, key=lambda x: x[1], reverse=True)
    if time_flag:
        optimal_config_indices = max(
            constrainted_combination_with_accuracies, key=lambda x: x[2])[0]
    else:
        optimal_config_indices = max(
            constrainted_combination_with_accuracies, key=lambda x: x[1])[0]
    # print(sorted_combinations[desired_rank-1][1])
    # if sorted_combinations[desired_rank-1][1] <= 65:
    #     raise ValueError("No solution found.")
    print("ILP time:", time.time() - start_t)
    return optimal_config_indices

# def solve_traverse_test(accuracy_matrix, resource_matrix,
#                    resource_constraint_value, max_pool_length, desired_rank=1,
#                    gpu_constrain=None):
#     if gpu_constrain is not None:
#         ClusetrParetoSet, indexs, gpu_resource_info, gpu_memory_bound = gpu_constrain

#     cluster_number, config_number = accuracy_matrix.shape
#     video_ids, config_ids = list(
#         range(cluster_number)), list(range(config_number))
#     combinations = list(itertools.product(config_ids, repeat=len(video_ids)))
#     constrainted_combinations = []
#     for combination in combinations:
#         resources = sum([resource_matrix[i][combination[i]]
#                         for i in range(len(combination))])
#         combination_resource = rfunc(resources / cluster_number,
#                                      cluster_number / max_pool_length)

#         if gpu_constrain is not None:
#             gpu_memory = []
#             for cluster_label, config_indice in enumerate(combination):
#                 if config_indice >= len(ClusetrParetoSet[cluster_label]):
#                     gpu_memory.append(1000000)
#                     break
#                 config_action = ClusetrParetoSet[cluster_label][config_indice][2]
#                 config_action_str = "_".join(
#                     [str(config_action[index]) for index in indexs])
#                 gpu_memory.append(gpu_resource_info[config_action_str])
#             if sum(gpu_memory) > gpu_memory_bound:
#                 continue

#         if combination_resource <= resource_constraint_value:
#             constrainted_combinations.append(combination)

#     if len(constrainted_combinations) == 0:
#         raise ValueError("No solution found.")
#     constrainted_combination_with_accuracies = []
#     for combination in constrainted_combinations:
#         accuracy = 0.0
#         for i in range(len(combination)):
#             accuracy += accuracy_matrix[i][combination[i]]
#         constrainted_combination_with_accuracies.append(
#             (combination, accuracy))
        
    
#     sorted_combinations = sorted(
#     constrainted_combination_with_accuracies, key=lambda x: x[1], reverse=True)
#     optimal_config_indices = sorted_combinations[desired_rank-1][0]

    
#     return optimal_config_indices


def rfunc(resources, num_data):
    return resources * 3.32253366 + num_data * 15.23385008 + -14.499901547036576

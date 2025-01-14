import random
import matplotlib.pyplot as plt

def crowding_distance_assignment(front, index_acc=1, index_time=2):
    if len(front) == 0:
        return []
    
    distances = [0 for _ in front]
    
    for index in [index_acc, index_time]:
        sorted_indices = sorted(range(len(front)), key=lambda x: front[x][index])
        
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        for i in range(1, len(front) - 1):
            distances[sorted_indices[i]] += (front[sorted_indices[i+1]][index] - front[sorted_indices[i-1]][index])
    
    return distances

def select_k_solutions(pareto_solutions, K):
    if K >= len(pareto_solutions):
        return pareto_solutions

    distances = crowding_distance_assignment(pareto_solutions)
    
    selected_indices = sorted(range(len(distances)), key=lambda x: distances[x], reverse=True)[:K]
    
    return [pareto_solutions[i] for i in selected_indices]


def generate_pareto_like_samples(n=100):
    # 生成准确率的基本值
    accuracies = sorted([random.uniform(85, 100) for _ in range(n)], reverse=True)
    
    # 根据准确率生成时间值（更高的准确率对应更长的时间）
    times = [a * 0.15 + random.uniform(0, 5) for a in accuracies]

    return [["config" + str(i), accuracies[i], times[i]] for i in range(n)]

pareto_solutions = generate_pareto_like_samples()

K = 10  
selected_solutions = select_k_solutions(pareto_solutions, K)

# 可视化
plt.figure(figsize=(10, 6))
# 全部解
plt.scatter([x[1] for x in pareto_solutions], [x[2] for x in pareto_solutions], marker='o', label='All Pareto Solutions', color='blue')
# 选中的解
plt.scatter([x[1] for x in selected_solutions], [x[2] for x in selected_solutions], marker='x', label='Selected Solutions', color='red')

plt.xlabel('Accuracy')
plt.ylabel('Time')
plt.legend()
plt.title('Pareto Solutions Visualization')
plt.grid(True)
plt.savefig('pareto.png', dpi=300, bbox_inches='tight')
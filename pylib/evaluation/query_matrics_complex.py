import math
import numpy as np
from collections import Counter

from scipy.optimize import linear_sum_assignment


class QueryMatrics(object):
    def __init__(self, object_type, end_frame, frame_rate):
        self.object_type = object_type
        self.end_frame = end_frame
        self.frame_rate = frame_rate

    def preprocess(self, input_label, input_pred, constrain=None):
        constrain = 0 if constrain is None else constrain
        self.pred_result = {i: 0 for i in range(1, self.end_frame)}
        self.gt_result = {i: 0 for i in range(1, self.end_frame)}
        self.pred_id = []
        self.gt_id = []
        for key in input_label.keys():
            if input_label[key]["class_id"] == self.object_type:
                if input_label[key]["frame_bound"]["start_frame_id"] < self.end_frame:
                    if abs(input_label[key]["frame_bound"]["end_frame_id"] - input_label[key]["frame_bound"]["start_frame_id"]) >= constrain:
                        self.gt_id.append(key)
                        for i in range(input_label[key]["frame_bound"]["start_frame_id"],
                                       min(input_label[key]["frame_bound"]["end_frame_id"] + 1, self.end_frame)):
                            self.gt_result[i] += 1

        for key in input_pred.keys():
            if input_pred[key]["class_id"] == self.object_type:
                if input_pred[key]["frame_bound"]["start_frame_id"] < self.end_frame:
                    if abs(input_pred[key]["frame_bound"]["end_frame_id"] - input_pred[key]["frame_bound"]["start_frame_id"]) >= constrain:
                        self.pred_id.append(key)
                        for i in range(input_pred[key]["frame_bound"]["start_frame_id"],
                                       min(input_pred[key]["frame_bound"]["end_frame_id"] + 1, self.end_frame)):
                            self.pred_result[i] += 1

    def selection_query_1(self):
        TP, FP, TN, FN = 0, 0, 0, 0

        for frame_id in range(1, self.end_frame):
            if self.gt_result[frame_id] >= 1 and self.pred_result[frame_id] >= 1:
                TP += 1
            elif self.gt_result[frame_id] >= 1 and self.pred_result[frame_id] == 0:
                FN += 1
            elif self.gt_result[frame_id] == 0 and self.pred_result[frame_id] >= 1:
                FP += 1
            else:
                TN += 1
        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = round((TP)/(TP+FP), 4)

        if TP == 0 and FN == 0:
            recall = 0
        else:
            recall = round((TP)/(TP+FN), 4)

        if TP == 0 and FP == 0 and FN == 0:
            accuracy = 0
        else:
            accuracy = round((TP+TN)/(TP+TN+FP+FN), 4)

        if TP == 0:
            F1 = 0
        else:
            F1 = round(2*(precision*recall)/(precision+recall), 4)

        return recall, precision, accuracy, F1

    def aggregation_query_1(self):
        """
        Query: Count the number of cars per frame
        """
        MAE = 0
        ACC = 0
        for i in range(1, self.end_frame):
            MAE += abs(self.pred_result[i]-self.gt_result[i])
            if self.gt_result[i] != 0:
                ACC += 1 - \
                    (abs(self.pred_result[i] -
                     self.gt_result[i])/self.gt_result[i])
            else:
                if self.pred_result[i] != 0:
                    ACC += 0
                else:
                    ACC += 1
        MAE = MAE/self.end_frame
        ACC = ACC/self.end_frame

        return MAE, ACC

    def aggregation_query_2(self, gaptime=10):
        """
        Query: Count the number of cars per frame gap
        gap is xxx (s)
        """
        gap = gaptime * self.frame_rate
        n = len(self.pred_result)
        pred_segment_sums = [
            sum([self.pred_result[j] for j in range(i, i+gap)]) for i in range(1, n-gap+2)]
        gt_segment_sums = [
            sum([self.gt_result[j] for j in range(i, i+gap)]) for i in range(1, n-gap+2)]
        MAE, ACC = 0, 0
        for pred_segment_sum, gt_segment_sum in zip(pred_segment_sums, gt_segment_sums):
            MAE += abs(pred_segment_sum-gt_segment_sum)
            if gt_segment_sum != 0:
                ACC += 1 - (abs(pred_segment_sum-gt_segment_sum) /
                            gt_segment_sum)
            else:
                if pred_segment_sum != 0:
                    ACC += 0
                else:
                    ACC += 1
        MAE = MAE/len(pred_segment_sums)
        ACC = ACC/len(gt_segment_sums)

        return MAE, ACC

    def aggregation_query_3(self, object_count, frame_num=10, gap=300):
        """
        Query: Count the number of cars per frame gap
        gap is xxx (s)
        """

        selected_frame = 0
        hit_count = 0
        last_select = -100

        for i in range(1, self.end_frame):
            if self.pred_result[i] >= object_count and i - last_select > gap:
                selected_frame += 1
                for nei_id in self.gt_result[max(0, i-5):min(i+5, len(self.pred_result))]:
                    if self.pred_result[nei_id] >= object_count:
                        hit_count += 1
                        break
                last_select = i
            if selected_frame == frame_num:
                break

        return hit_count/frame_num

    def aggregation_query_4(self):
        """
        Query: Count the total number of cars
        """
        return len(set(self.gt_id)), len(set(self.pred_id))

    def segment_overlap(self, idx1, idx2, gap):
        """Calculate the overlap length between two segments starting at idx1 and idx2."""
        overlap_start = max(idx1, idx2)
        overlap_end = min(idx1 + gap, idx2 + gap)
        return max(0, overlap_end - overlap_start)

    def construct_cost_matrix(self, pred_indices, gt_indices, gap):
        """Construct the cost matrix for the Hungarian algorithm."""
        cost_matrix = []
        for pred_idx in pred_indices:
            cost_row = [
                gap - self.segment_overlap(pred_idx, gt_idx, gap) for gt_idx in gt_indices]
            cost_matrix.append(cost_row)
        return cost_matrix

    def top_k_query_1(self, k=5, gap=64):
        """
        Query the topk segments with the most targets using the Hungarian algorithm
        """
        gt_topk_indices = self.top_k_segments(self.gt_result, gap=gap, k=k)
        pred_topk_indices = self.top_k_segments(self.pred_result, gap=gap, k=k)

        cost_matrix = self.construct_cost_matrix(
            pred_topk_indices, gt_topk_indices, gap)

        # Use the Hungarian algorithm to find the optimal assignment
        _, optimal_gt_indices = linear_sum_assignment(cost_matrix)

        # Calculate total overlap using the optimal assignment
        total_overlap = sum(self.segment_overlap(pred_idx, gt_topk_indices[gt_idx], gap)
                            for pred_idx, gt_idx in zip(pred_topk_indices, optimal_gt_indices))

        overlap_rate = total_overlap / (k * gap)

        return overlap_rate

    def top_k_segments(self, data_list, gap=50, k=10):
        """
        Query the topk segments with the most targets, ensuring no overlap between segments.
        """
        n = len(data_list)
        segment_sums = [sum([data_list[j] for j in range(i, i+gap)])
                        for i in range(1, n-gap+2)]

        top_k_indices = []
        indices_sorted_by_sum = sorted(
            range(len(segment_sums)), key=lambda i: segment_sums[i], reverse=True)

        while len(top_k_indices) < k and indices_sorted_by_sum:
            idx = indices_sorted_by_sum.pop(0)
            top_k_indices.append(idx)

            # Remove overlapping segments
            overlap_range = set(range(max(0, idx - gap + 1), idx + gap))
            indices_sorted_by_sum = [
                i for i in indices_sorted_by_sum if i not in overlap_range]

        return top_k_indices

    def _sort_max_intervals(self, input, gap):

        selected_id = []
        max_set = []

        for id in input:
            if len(selected_id) == 0:
                selected_id.append([id, id])
            else:
                match_flag = False
                for clus in selected_id:
                    if id > clus[0] and id < clus[-1]:
                        clus.append(id)
                        clus.sort()
                        match_flag = True
                        break
                    elif (id > clus[-1]) and (id < clus[-1]+int(0.25*gap)):
                        clus.append(id)
                        match_flag = True
                        break
                    elif (id < clus[0]) and (id > clus[0]-int(0.25*gap)):
                        clus.insert(0, id)
                        match_flag = True
                        break
                if not match_flag:
                    selected_id.append([id, id])

        for selected in selected_id:
            tmp = [self.gt_result[i]
                   for i in range(selected[0], selected[-1]+1)]
            max_set.append(max(tmp))
        return max_set

    def cardinality_limited_query_1(self):
        object_num_list = list(set(list(self.gt_result.values())))
        max_object_num = max(object_num_list)

        if max_object_num > 1:
            half_object_num = int(max_object_num/2)
            cardinality_object_nums = []
            for gt_object_num in object_num_list:
                cardinality_object_nums.append(
                    [gt_object_num, abs(gt_object_num-half_object_num)])
            cardinality_object_num = min(
                cardinality_object_nums, key=lambda x: x[1])[0]
        else:
            return None
        meet_the_condition_gt_frame_id = []
        meet_the_condition_pred_frame_id = []
        for frame_id in range(1, self.end_frame):
            if self.gt_result[frame_id] == cardinality_object_num:
                meet_the_condition_gt_frame_id.append(frame_id)
            if self.pred_result[frame_id] == cardinality_object_num:
                meet_the_condition_pred_frame_id.append(frame_id)

        if len(set(meet_the_condition_pred_frame_id)) == 0:
            precision = 0
        else:
            precision = len(set(meet_the_condition_gt_frame_id) & set(
                meet_the_condition_pred_frame_id)) / len(set(meet_the_condition_pred_frame_id))

        if len(set(meet_the_condition_gt_frame_id)) == 0:
            return None
        else:
            recall = len(set(meet_the_condition_gt_frame_id) & set(
                meet_the_condition_pred_frame_id)) / len(set(meet_the_condition_gt_frame_id))

        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def cardinality_limited_query_2(self, input_label, input_pred):
        if self.object_type == 2:
            car_num, bus_num = 10, 1
            constraint_object_type = 5
            bus_pred_result = {i: 0 for i in range(1, self.end_frame)}
            bus_gt_result = {i: 0 for i in range(1, self.end_frame)}
            for key in input_label.keys():
                if input_label[key]["class_id"] == constraint_object_type:
                    if input_label[key]["frame_bound"]["start_frame_id"] < self.end_frame:
                        for i in range(input_label[key]["frame_bound"]["start_frame_id"],
                                    min(input_label[key]["frame_bound"]["end_frame_id"] + 1, self.end_frame)):
                            bus_gt_result[i] += 1
            for key in input_pred.keys():
                if input_pred[key]["class_id"] == constraint_object_type:
                    if input_pred[key]["frame_bound"]["start_frame_id"] < self.end_frame:
                        for i in range(input_pred[key]["frame_bound"]["start_frame_id"],
                                    min(input_pred[key]["frame_bound"]["end_frame_id"] + 1, self.end_frame)):
                            bus_pred_result[i] += 1
            
            meet_the_condition_gt_frame_id = []
            meet_the_condition_pred_frame_id = []
            for frame_id in range(1, self.end_frame):
                if self.gt_result[frame_id] == car_num and bus_gt_result[frame_id] == bus_num:
                    meet_the_condition_gt_frame_id.append(frame_id)
                if self.pred_result[frame_id] == car_num and bus_pred_result[frame_id] == bus_num:
                    meet_the_condition_pred_frame_id.append(frame_id)
                    
            if len(set(meet_the_condition_pred_frame_id)) == 0:
                precision = 0
            else:
                precision = len(set(meet_the_condition_gt_frame_id) & set(
                    meet_the_condition_pred_frame_id)) / len(set(meet_the_condition_pred_frame_id))

            if len(set(meet_the_condition_gt_frame_id)) == 0:
                return None
            else:
                recall = len(set(meet_the_condition_gt_frame_id) & set(
                    meet_the_condition_pred_frame_id)) / len(set(meet_the_condition_gt_frame_id))

            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return f1
        else:
            return 1.0

    @staticmethod
    def calculate_angle_change(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        angle_change = math.degrees(angle2 - angle1)
        return angle_change

    def classify_trajectory(self, trajectory):
        angle_changes = []
        for i in range(1, len(trajectory) - 1):
            angle_change = self.calculate_angle_change(np.array(trajectory[i-1]), 
                                                np.array(trajectory[i]), 
                                                np.array(trajectory[i+1]))
            angle_changes.append(angle_change)
        
        # 简单的分类逻辑
        total_angle_change = sum(angle_changes)
        if abs(total_angle_change) < 20:  # 阈值可以根据实际情况调整
            return 0
        elif total_angle_change > 0:
            return 1
        else:
            return 2

    def cardinality_limited_query_3(self, input_label, input_pred):
        left_turn_num = 0
        right_turn_num = 0
        straight_num = 0
        
        pred_left_turn_num = 0
        pred_right_turn_num = 0
        pred_straight_num = 0
        
        for key in input_label.keys():
            if input_label[key]["class_id"] == self.object_type:
                trajectory = input_label[key]["position_list"]
                trajectory_class = self.classify_trajectory(trajectory)
                if trajectory_class == 0:
                    straight_num += 1
                elif trajectory_class == 1:
                    left_turn_num += 1
                else:
                    right_turn_num += 1
        
        for key in input_pred.keys():
            if input_pred[key]["class_id"] == self.object_type:
                trajectory = input_pred[key]["position_list"]
                trajectory_class = self.classify_trajectory(trajectory)
                if trajectory_class == 0:
                    pred_straight_num += 1
                elif trajectory_class == 1:
                    pred_left_turn_num += 1
                else:
                    pred_right_turn_num += 1
        
        acc = 0
        acc += 1 - abs(pred_left_turn_num - left_turn_num) / left_turn_num
        acc += 1 - abs(pred_right_turn_num - right_turn_num) / right_turn_num
        acc += 1 - abs(pred_straight_num - straight_num) / straight_num
        acc = acc / 3
        return acc

    def matrics(self, gt_tuple, pred_tuple, constrain=None):
        self.preprocess(gt_tuple, pred_tuple)
        recall, precision, accuracy, F1 = self.selection_query_1()
        MAE, ACC = self.aggregation_query_1()
        gt_vehicle, pred_vehicle = self.aggregation_query_4()
        acc_topk = self.top_k_query_1()
        cardinality_f1 = self.cardinality_limited_query_1()
        cardinality2_f1 = self.cardinality_limited_query_2(gt_tuple, pred_tuple)
        cardinality3_acc = self.cardinality_limited_query_3(gt_tuple, pred_tuple)

        self.preprocess(gt_tuple, pred_tuple, constrain=constrain)
        recall_constrain, precision_constrain, accuracy_constrain, F1_constrain = self.selection_query_1()
        MAE_constrain, ACC_constrain = self.aggregation_query_1()
        gt_vehicle_constrain, pred_vehicle_constrain = self.aggregation_query_4()
        acc_topk_constrain = self.top_k_query_1()
        cardinality_f1_constrain = self.cardinality_limited_query_1()

        return recall, precision, accuracy, F1, MAE, ACC, gt_vehicle, pred_vehicle, acc_topk, cardinality_f1, cardinality2_f1, cardinality3_acc, \
            recall_constrain, precision_constrain, accuracy_constrain, F1_constrain, MAE_constrain, \
            ACC_constrain, gt_vehicle_constrain, pred_vehicle_constrain, acc_topk_constrain, \
            cardinality_f1_constrain
"""
This file implements a class which can evaluate the recall and false positive
"""
import numpy as np


class DetectionResultRecord(object):
    def __init__(self, score_threshold_stride):
        assert score_threshold_stride > 0
        self.score_threshold_stride = score_threshold_stride

        self.calcification_num_dataset_level = 0

        recall_num_dataset_level_list = list()
        FP_num_dataset_level_list = list()

        score_threshold = 0
        while score_threshold <= 1:
            recall_num_dataset_level_list.append(0)
            FP_num_dataset_level_list.append(0)

            score_threshold += self.score_threshold_stride

        self.recall_num_dataset_level_np = np.array(recall_num_dataset_level_list)
        self.FP_num_dataset_level_np = np.array(FP_num_dataset_level_list)

        return

    def __add__(self, other):
        assert self.score_threshold_stride == other.score_threshold_stride

        self.calcification_num_dataset_level += other.calcification_num_dataset_level
        self.recall_num_dataset_level_np += other.recall_num_dataset_level_np
        self.FP_num_dataset_level_np += other.FP_num_dataset_level_np

        return self

    def update_calcification_num(self, calcification_num_dataset_level):
        self.calcification_num_dataset_level += calcification_num_dataset_level

        return

    def update_recall_num(self, threshold_idx, recall_num):
        self.recall_num_dataset_level_np[threshold_idx] += recall_num

        return

    def update_FP_num(self, threshold_idx, FP_num):
        self.FP_num_dataset_level_np[threshold_idx] += FP_num

        return

    def get_detected_num(self):
        detected_num = (self.recall_num_dataset_level_np + self.FP_num_dataset_level_np)[0]

        return detected_num

    def print(self, logger=None):
        if logger is None:
            print('The annotated calcification number is: {}'.format(self.calcification_num_dataset_level))
            print('The number of the detected positive calcifications is: {}'.format(self.get_detected_num()))
            score_threshold = 0
            threshold_idx = 0
            while score_threshold <= 1:
                print('score_threshold = {}, recall_number = {}, FP_number = {}'.format(score_threshold,
                                                                                        self.recall_num_dataset_level_np[
                                                                                            threshold_idx],
                                                                                        self.FP_num_dataset_level_np[
                                                                                            threshold_idx]))

                score_threshold += self.score_threshold_stride
                threshold_idx += 1
        else:
            logger.write_and_print(
                'The annotated calcification number is: {}'.format(self.calcification_num_dataset_level))
            logger.write_and_print(
                'The number of the detected positive calcifications is: {}'.format(self.get_detected_num()))
            score_threshold = 0
            threshold_idx = 0
            while score_threshold <= 1:
                logger.write_and_print(
                    'score_threshold = {}, recall_number = {}, FP_number = {}'.format(score_threshold,
                                                                                      self.recall_num_dataset_level_np[
                                                                                          threshold_idx],
                                                                                      self.FP_num_dataset_level_np[
                                                                                          threshold_idx]))

                score_threshold += self.score_threshold_stride
                threshold_idx += 1

        return


class MetricsRadiographLevelDetection(object):
    def __init__(self, distance_threshold, score_threshold_stride):
        """
        :param distance_threshold: the threshold for discriminating recall amd FP
        """

        assert distance_threshold > 0
        self.distance_threshold = distance_threshold

        assert score_threshold_stride > 0
        self.score_threshold_stride = score_threshold_stride

        self.detection_result_record_dataset_level = DetectionResultRecord(self.score_threshold_stride)

        return

    def metric_all_score_thresholds(self, pred_coord_list, pred_score_list, label_coord_list):
        """
        evaluate at batch-level
        :param preds: residues
        :param labels: pixel-level label without dilated
        :return: the number of recalled calcification and FP
        """

        assert len(pred_score_list) == len(pred_coord_list)

        detection_result_record_radiograph_level = DetectionResultRecord(self.score_threshold_stride)
        detection_result_record_radiograph_level.update_calcification_num(len(label_coord_list))

        score_threshold = 0
        threshold_idx = 0
        while score_threshold <= 1:
            processed_pred_coord_list = self.process_pred_coord_list(pred_coord_list, pred_score_list, score_threshold)

            detection_result_record_radiograph_level = self.metric_a_specific_score_threshold(processed_pred_coord_list,
                                                                                              label_coord_list,
                                                                                              threshold_idx,
                                                                                              detection_result_record_radiograph_level)

            score_threshold += self.score_threshold_stride
            threshold_idx += 1

        self.detection_result_record_dataset_level = self.detection_result_record_dataset_level + \
                                                     detection_result_record_radiograph_level

        return detection_result_record_radiograph_level

    def process_pred_coord_list(self, pred_coord_list, pred_score_list, score_threshold):
        assert len(pred_score_list) == len(pred_coord_list)

        processed_pred_coord_list = list()

        for pred_idx in range(len(pred_coord_list)):
            pred_coord = pred_coord_list[pred_idx]
            pred_score = pred_score_list[pred_idx]
            if pred_score >= score_threshold:
                processed_pred_coord_list.append(pred_coord)

        assert len(pred_score_list) >= len(processed_pred_coord_list)

        return processed_pred_coord_list

    def metric_a_specific_score_threshold(self, pred_coord_list, label_coord_list, threshold_idx,
                                          detection_result_record_radiograph_level):
        pred_num = len(pred_coord_list)
        label_num = len(label_coord_list)

        recall_num = 0
        FP_num = 0

        # for the negative patch case
        if label_num == 0:
            FP_num = pred_num

        # for the positive patch case with failing to detect anything
        elif pred_num == 0:
            recall_num = 0

        # for the positive patch case with something being detected
        else:
            # calculate recall
            for label_idx in range(label_num):
                for pred_idx in range(pred_num):
                    if np.linalg.norm(
                            label_coord_list[label_idx] - pred_coord_list[pred_idx]) <= self.distance_threshold:
                        recall_num += 1
                        break

            # calculate FP
            for pred_idx in range(pred_num):
                for label_idx in range(label_num):
                    if np.linalg.norm(
                            label_coord_list[label_idx] - pred_coord_list[pred_idx]) <= self.distance_threshold:
                        break
                    if label_idx == label_num - 1:
                        FP_num += 1

        detection_result_record_radiograph_level.update_recall_num(threshold_idx, recall_num)
        detection_result_record_radiograph_level.update_FP_num(threshold_idx, FP_num)

        return detection_result_record_radiograph_level

import os.path
import cv2,warnings,sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sort import *
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from collections import deque
warnings.filterwarnings('ignore')
"""
A class to recognize suspicious stand-up persons using SORT and statistical analysis

we should collect some frames detection results from the video time by time, and feed into this class. 
the results  
main process:
1,  object detection and object tracking
2,  save the initial detection results and tracking results in dataframe
3,  combine different id for one object and fill in the missing id for one object
4,  find the corresponding head box for each action box
5,  visualization each frame
"""

def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union of two sets of bounding boxes - `boxes_true` and `boxes_detection`. Both sets of
    boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true: 2d `np.ndarray` representing ground-truth boxes. `shape = (N, 4)` where N is number of true objects.
        boxes_detection: 2d `np.ndarray` representing detection boxes. `shape = (M, 4)` where M is number of detected objects.

    Returns:
        iou: 2d `np.ndarray` representing pairwise IoU of boxes from `boxes_true` and `boxes_detection`. `shape = (N, M)` where N is number of true objects and M is number of detected objects.

    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)
class JudgeSuspiciousStandUp:
    def __init__(self,track_frame_num,detect_confidence_threshold=0.35):
        """
        track_frame_num: the number of frames we need to track and do statistics
        detect_confidence_threshold :confidence threshold score for object detection
        """
        self.track_frame_num = track_frame_num
        self.frame_count=0
        self.input_df=pd.DataFrame(columns=['stamp','imgid','actid','x1','y1','x2','y2','cid','score','face_idx'])
        self.time_stamp_list=[]
        # each element is (time_stamp,detection data array)
        self.detect_array_queue = deque(maxlen=self.track_frame_num)
        # output dict, saving the judge result for each time stamp
        self.output_dict = {}
        # confidence score threshold
        self.detect_score_confidence=detect_confidence_threshold
        # record the group number algorithm handle
        self.group_num = 1
    @staticmethod
    def box_iou(box1, box2):
        # box1
        x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
        s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
        # box2
        x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
        s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

        # intersection box
        xmin = np.maximum(x1min, x2min)
        ymin = np.maximum(y1min, y2min)
        xmax = np.minimum(x1max, x2max)
        ymax = np.minimum(y1max, y2max)
        inter_h = np.maximum(ymax - ymin + 1., 0.)
        inter_w = np.maximum(xmax - xmin + 1., 0.)
        intersection = inter_h * inter_w
        # union box
        union = s1 + s2 - intersection
        iou = intersection / union
        return iou

    @staticmethod
    def box_intersection_area(box_a, box_b):
        # Calculate the intersection area between box_a and box_b
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        # Calculate the coordinates of the intersection box
        x1_int = max(x1_a, x1_b)
        y1_int = max(y1_a, y1_b)
        x2_int = min(x2_a, x2_b)
        y2_int = min(y2_a, y2_b)

        # Calculate the area of box_a and box_b
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        # If the intersection is valid (not negative width or height)
        if x1_int < x2_int and y1_int < y2_int:
            # Calculate intersection area
            intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
            # Calculate the union area
            union_area = area_a + area_b - intersection_area
            return intersection_area / union_area
        else:
            # No intersection
            return 0


    def find_max_intersection_box(self,cur_box, b_list):
        max_intersection = 0
        max_intersection_idx = -1
        all_area = []
        for idx, box_b in enumerate(b_list):
            intersection_area = self.box_intersection_area(cur_box, box_b)
            all_area.append(intersection_area)
        # find the qualified head box
        actx1, acty1, actx2, acty2 = cur_box
        actcy = (acty1 + acty2) / 2
        for idx, area in enumerate(all_area):
            hx1, hy1, hx2, hy2 = b_list[idx]
            hcy = (hy1 + hy2) / 2
            if area > max_intersection and hcy < actcy:
                max_intersection = area
                max_intersection_idx = idx
        if max_intersection < 0.15:
            return None
        else:
            return b_list[max_intersection_idx]


    def find_related_tracker(self,cur_box, previous_1_track_df):
        """
        match current frame untracked detection box and previous frame tracked detection box
        cur_box: untracked detection box coordinates
        previous_1_track_df: previous frame tracked detection box dataframe
        """
        previous_1_track_df['iou'] = previous_1_track_df.apply(self.box_iou, axis=1, box2=cur_box)
        previous_match_tracker_index = previous_1_track_df['iou'].idxmax()
        previous_bottom_y = previous_1_track_df.loc[previous_match_tracker_index, 'by2']
        if previous_1_track_df['iou'].max() < 0.25 and (cur_box[3] - previous_bottom_y) > 20:
            return None
        return previous_match_tracker_index

    @staticmethod
    def find_corresponding_head(head_detections, cur_box):
        """
        finding the corresponding head box of current action box
        """
        head_boxes = head_detections[:, 0:4]
        cur_box_array = np.tile(cur_box, (head_boxes.shape[0], 1))
        # find the corresponding head box through iou
        corresponding_id = np.argmax(box_iou_batch(cur_box_array, head_boxes))
        corresponding_head = head_detections[corresponding_id]
        if np.max(box_iou_batch(cur_box_array, head_boxes)) < 0.1:
            return None
        return corresponding_head

    @staticmethod
    def suspicious_stand_up_judge(id_df):

        # judge everyone stand up or not
        # if not:0, if suspicious to stand up:1
        Q1 = id_df['top_y'].quantile(0.25)
        Q3 = id_df['top_y'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        id_df_new = id_df[id_df['top_y'] <= upper_bound]
        id_df_new.loc[:, 'stand_candidate'] = 0
        diff_y = id_df_new['top_y'].max() - id_df_new['top_y'].min()
        if diff_y < 30 or len(id_df) < 8:
            return id_df_new
        kmeans = KMeans(n_clusters=2, random_state=42).fit(np.array(id_df_new['top_y']).reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        if centers[0] > centers[1]:
            labels = 1 - labels
            centers = centers[::-1]
        gap = centers[1] - centers[0]
        t0_series = id_df_new['stamp'][labels == 0]
        t0_series_diff = np.diff(t0_series)
        t0_act_judge1 = id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'cid'] == 2
        t0_act_judge2 = id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'cid'] == 3
        t0_act_judge3 = id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'cid'] == 11
        t0_act_judge4 = id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'cid'] == 0
        t0_act_judge_num = t0_act_judge1.sum() + t0_act_judge2.sum() + t0_act_judge3.sum()
        t0_act_judge_num2 = t0_act_judge4.sum()
        if t0_act_judge_num2 > 4:
            id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'stand_candidate'] = 1
        # only tolerate the time gap of 2 frames (at most 8s)
        elif gap > 30 and t0_act_judge_num < 3 and (t0_series_diff < 13).all():
            id_df_new.loc[id_df_new['stamp'].isin(t0_series), 'stand_candidate'] = 1

        return id_df_new

    def track(self,df,time_stamp_list):
        """
        the object tracking main process with SORT tracker
        df: object detection results with bounding box, confidence score, time stamp in multiple frames
        time_stamp_list: time stamp list need to track  in multiple frames
        """
        # initialize a new SORT tracker
        tracker = Sort(max_age=35, min_hits=2, iou_threshold=0.25)
        person_series_df = pd.DataFrame(columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid'])
        head_series_df = pd.DataFrame(columns=['stamp', 'hx1', 'hy1', 'hx2', 'hy2', 'id'])
        for time_idx, timestamp in enumerate(time_stamp_list):
            sub_df = df[df['stamp'] == timestamp]
            detections = np.empty((0, 6))
            # draw the detected results on the image
            for row in sub_df.iterrows():
                cid = row[1]['cid']
                imgid = int(row[1]['imgid'])
                score = row[1]['score']
                if imgid % 2 != 0 or score < self.detect_score_confidence:
                    continue
                    # Get bounding box coordinates
                x1, y1, x2, y2 = int(row[1]['x1']), int(row[1]['y1']), int(row[1]['x2']), int(row[1]['y2'])
                currentArray = np.array([x1, y1, x2, y2, score, 0])
                if cid == 9:
                    hid = -1000
                    sample_df_head = pd.DataFrame(data=np.column_stack([timestamp, x1, y1, x2, y2, hid]),
                                                  columns=['stamp', 'hx1', 'hy1', 'hx2', 'hy2', 'id'])
                    head_series_df = pd.concat([head_series_df, sample_df_head], axis=0, ignore_index=True)
                else:
                    detections = np.vstack((detections, currentArray))

            resultTracker, unmatched_dets = tracker.update(detections)
            # 2,  save the initial detection results and tracking results in dataframe
            for result in resultTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cur_box = np.array([x1, y1, x2, y2])
                sub_df['area'] = sub_df.iloc[:, 3:7].apply(self.box_intersection_area, axis=1, box_b=cur_box)
                cid = sub_df.loc[sub_df['area'].idxmax(), 'cid']
                # temporaly mask as unknown
                save_x, save_y = -100, -100
                sample_df = pd.DataFrame(data=np.column_stack([id, save_x, save_y, timestamp, x1, y1, x2, y2, cid]),
                                         columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid'])
                person_series_df = pd.concat([person_series_df, sample_df], axis=0, ignore_index=True)

            # 3,deal with missing tracking and associate them by hand
            """                                                                                                                                                                                     
            1. find unmatched detected box                                                                                                                                                          
            2, calculate IOU with trackers' boxes in previous frame ,find matched tracker                                                                                                           
            3, find tracker id                                                                                                                                                                      
            """
            total_track_person = len(resultTracker)
            current_detect_person = len(detections)
            # some detection associate fails
            if current_detect_person > total_track_person:
                # finding the failure associated detection
                detect_array = detections[:, 0:4]
                track_array = resultTracker[:, 0:4].astype(int)
                iou_matrix = box_iou_batch(detect_array, track_array)
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                all_row_indices = np.arange(iou_matrix.shape[0])  # Array of all row indices
                matched_rows = set(row_ind)  # Rows that have been matched
                unmatched_rows = list(set(all_row_indices) - matched_rows)  # Unmatched rows
                previous_1_timestamp = time_stamp_list[time_idx - 1]
                cur_id_df = person_series_df[person_series_df['stamp'] == timestamp].loc[:, 'id']
                previous_1_track_df = person_series_df.loc[person_series_df['stamp'] == previous_1_timestamp, :].iloc[:,4::]
                previous_1_track_df_id = person_series_df.loc[person_series_df['stamp'] == previous_1_timestamp, :]
                for unmatch_row_id in unmatched_rows:
                    cur_box_ = detect_array[unmatch_row_id]
                    previous_match_tracker_index = self.find_related_tracker(cur_box_, previous_1_track_df)
                    if previous_match_tracker_index is not None:
                        previous_match_tracker_id = previous_1_track_df_id.loc[previous_match_tracker_index].id
                        exists_in_cur = previous_match_tracker_id in cur_id_df.values
                        # make sure not deplicate, and record it
                        if not exists_in_cur:
                            x1, y1, x2, y2 = cur_box_
                            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cid = sub_df.loc[(sub_df['x1'] == x1) & (sub_df['x2'] == x2) & (sub_df['y1'] == y1) & (
                                    sub_df['y2'] == y2), 'cid']
                            sample_df = pd.DataFrame(data=np.column_stack(
                                [previous_match_tracker_id, -100, -100, timestamp, x1, y1, x2, y2, cid]),
                                columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid'])
                            person_series_df = pd.concat([person_series_df, sample_df], axis=0, ignore_index=True)

        # 3, combine different id for one object
        average_values = person_series_df.groupby('id')[['bx1', 'by1', 'bx2', 'by2']].mean()
        l2_distances = cdist(average_values, average_values, metric='euclidean')
        max_distance = np.max(l2_distances)
        l2_distances = l2_distances / max_distance
        upper_distance = np.triu(l2_distances, k=1)
        # finding the close bounding boxes series pairs
        indices = np.where((upper_distance < 0.05) & (upper_distance != 0))
        # Extract values based on the indices
        values = upper_distance[indices]
        # Map the indices back to the original ids
        id0_list = average_values.index[indices[0]]
        id1_list = average_values.index[indices[1]]
        id_counts = person_series_df.value_counts('id')
        for id0, id1 in zip(id0_list, id1_list):
            id0_count = id_counts[id0]
            id1_count = id_counts[id1]
            if id0_count < 10 or id1_count < 10:
                if id0_count <= id1_count:
                    keep_id = id1
                    delete_id = id0
                else:
                    keep_id = id0
                    delete_id = id1
                # check per frame, make sure no duplication
                #  in frame t, keep id not exist then we can change the delete_id to keep_id
                delete_id_stamp = person_series_df.loc[person_series_df['id'] == delete_id, 'stamp'].unique()
                keep_id_stamp = person_series_df.loc[person_series_df['id'] == keep_id, 'stamp'].unique()
                intersection_stamp = np.intersect1d(delete_id_stamp, keep_id_stamp)
                change_flag = True
                if len(intersection_stamp) != 0 or len(delete_id_stamp) == 0 or len(keep_id_stamp) == 0:
                    change_flag = False

                if change_flag:
                    person_series_df.loc[person_series_df['id'] == delete_id, 'id'] = keep_id

        return person_series_df,head_series_df


    def JudgeAndVisualize(self,person_series_df,head_series_df,video_path,source_file):

        stamp_list = person_series_df['stamp'].unique()
        for stamp in stamp_list:
            sub_stamp_df = person_series_df[person_series_df['stamp'] == stamp]
            sub_head_df = head_series_df[head_series_df['stamp'] == stamp]
            head_detections = np.array(sub_head_df.iloc[:, 1::])
            if video_path is not None:
                cap = cv2.VideoCapture(video_path)
                # Set the video position to the desired timestamp in milliseconds
                cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)  # Convert seconds to milliseconds
                # Read the frame
                ret, cur_frame = cap.read()
                if not os.path.exists(source_file + '/' +'{}'.format(self.group_num)):
                    os.mkdir(source_file + '/' +'{}'.format(self.group_num))

            for index, track_obj in sub_stamp_df.iterrows():
                x1, y1, x2, y2, id = track_obj['bx1'], track_obj['by1'], track_obj['bx2'], track_obj['by2'], track_obj['id']
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cur_box = np.array([x1, y1, x2, y2])
                # corresponding_head=find_corresponding_head(head_detections,cur_box)
                corresponding_head = self.find_max_intersection_box(cur_box, head_detections[:, 0:4])
                if corresponding_head is not None:
                    hx1, hy1, hx2, hy2 = int(corresponding_head[0]), int(corresponding_head[1]), int(corresponding_head[2]), int(corresponding_head[3])
                    head_series_df.loc[(head_series_df['hx1'] == hx1) & (head_series_df['hx2'] == hx2) & (
                                head_series_df['hy1'] == hy1) & (head_series_df['hy2'] == hy2) & (
                                                   head_series_df['stamp'] == stamp), 'id'] = id
                    save_x, save_y = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                    if video_path is not None:
                        cv2.rectangle(cur_frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2, 1)

                    # save_x, save_y = hx1,hy1
                else:
                    # save_x, save_y = (x1 + x2) / 2, (y1 + y2) / 2
                    save_x, save_y = x1, y1
                if video_path is not None:
                    cv2.putText(cur_frame, f'ID:{int(id)}', (max(0, int(x1)), max(35, int(y1))),
                            2, 1, (0, 0, 0), 2)
                    cv2.rectangle(cur_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, 1)

                person_series_df.loc[
                    (person_series_df['id'] == id) & (person_series_df['stamp'] == stamp), 'top_x'] = save_x
                person_series_df.loc[
                    (person_series_df['id'] == id) & (person_series_df['stamp'] == stamp), 'top_y'] = save_y
            if video_path is not None :
                cv2.imwrite(source_file + '/' +'{}'.format(self.group_num) +'/'+'frame_at_{}.jpg'.format(stamp), cur_frame)

        print(person_series_df)
        id_list = person_series_df['id'].unique()

        if source_file is not None and (not os.path.exists(source_file + '/' +'{}'.format(self.group_num) +'/'+'id_plot')):
            os.mkdir(source_file + '/' +'{}'.format(self.group_num) +'/'+'id_plot')

        person_series_df.loc[:, 'stand_candidate'] = 0
        for i, id in enumerate(id_list):
            # if id == 42.0:
            #     print("id:", id)
            id_df = person_series_df[person_series_df['id'] == id].copy()
            id_head_df = head_series_df[head_series_df['id'] == id].copy()
            id_df_new = self.suspicious_stand_up_judge(id_df)


            # Highlight standing candidates
            stand_points = id_df_new[id_df_new['stand_candidate'] == 1]
            if len(stand_points) != 0:
                match_columns = ['stamp', 'bx1', 'by1', 'bx2', 'by2']
                merged_df = pd.merge(person_series_df,
                                     id_df_new[['stand_candidate'] + match_columns],
                                     on=match_columns,
                                     how='left',
                                     suffixes=('', '_new'))
                person_series_df['stand_candidate'] = merged_df['stand_candidate_new'].combine_first(
                    person_series_df['stand_candidate'])

            # Visualization
            if video_path is not None and source_file is not None:
                plt.figure(i)
                plt.scatter(id_df['top_x'], id_df['top_y'])
                plt.scatter(stand_points['top_x'], stand_points['top_y'], c='red')

                for x, y, cid, stamp in zip(id_df['top_x'], id_df['top_y'], id_df['cid'], id_df['stamp']):
                    text = '{}:{}'.format(int(stamp), int(cid))
                    plt.text(x, y, text, fontsize=6, ha='center', va='bottom', alpha=0.7)
                plt.xlabel('X position')
                plt.ylabel('Y Position')
                plt.savefig(f'{source_file}/{self.group_num}/id_plot/id_{id}.png', bbox_inches='tight')
                plt.close()
    def GetFormatOutput(self,person_series_df):
        """
        get the final recognition output result from person_series_df
        output array will be a 0-1 array, 0: not stand up 1: stand up suspiciously
        """
        stamp_list = person_series_df['stamp'].unique()
        assert len(stamp_list) == len(self.detect_array_queue)
        for stamp,input_df in self.detect_array_queue:
            #non_head_array = input_df.loc[(input_df['cid'] != 9) & (input_df['score'] >= self.detect_score_confidence), ['x1', 'y1', 'x2', 'y2']]
            #non_head_array =np.array(non_head_array)
            input_array= np.array(input_df.loc[:,['x1', 'y1', 'x2', 'y2']])
            #innitial the output judge result
            output_judge_array=-1*np.ones(len(input_array))
            sub_stamp_df = person_series_df[person_series_df['stamp'] == stamp]
            sub_stamp_array=np.array(sub_stamp_df.loc[:,['bx1','by1','bx2','by2']])
            #match the handled df result box to the input array box
            iou_matrix = box_iou_batch(input_array, sub_stamp_array)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            # Array of all row indices
            all_row_indices = np.arange(iou_matrix.shape[0])
            # Rows that have been matched
            matched_rows = set(row_ind)
            # Unmatched rows
            unmatched_rows = list(set(all_row_indices) - matched_rows)
            #find the input order of each array
            judge_stand_up=sub_stamp_df.iloc[col_ind]['stand_candidate']
            output_judge_array[row_ind] = judge_stand_up
            self.output_dict[stamp] =list(output_judge_array)
    def main(self,input_array,current_stamp,video_path=None,source_file=None):
        """
        The main process of stand-up recognition algorithm. When the input frame data array reach self.track_frame_num,
        it will output the result
        video_path: the source video, for visualization; can set to None to avoid it
        source_path: the saving path, for visualization; can set to None to avoid it
        current_stamp: the time stamp of current frame
        input_array: Nx10 array, 10='stamp','imgid','actid','x1','y1','x2','y2','cid','score','face_idx',
          the input data of specific time stamp
        """
        self.frame_count+=1
        self.time_stamp_list.append(current_stamp)
        input_df=pd.DataFrame(input_array,columns=['stamp','imgid','actid','x1','y1','x2','y2','cid','score','face_idx'])
        self.detect_array_queue.append((current_stamp,input_df))
        # initialize the output result in this time stamp
        self.output_dict[current_stamp]=-1*np.ones(len(input_array))
        self.input_df=pd.concat([self.input_df,input_df],axis=0)
        if self.frame_count == self.track_frame_num:
            person_series_df, head_series_df=self.track(self.input_df,self.time_stamp_list)
            self.JudgeAndVisualize(person_series_df, head_series_df,video_path,source_file)
            self.GetFormatOutput(person_series_df)
            # clear all the containers (self.time_stamp_list,self.detect_array_queue,self.input_df),
            # and reset the frame_count
            self.detect_array_queue.clear()
            self.frame_count=0
            self.input_df=self.input_df[0:0]
            self.time_stamp_list.clear()
            self.group_num+=1
            self.output_dict.clear()

        return self.output_dict

if __name__ == '__main__':
    source_file = './source5'
    video_path = os.path.join(source_file, 'students-full.mp4')
    # load raw saved data from db and visualize it
    # Connect to the SQLite database file
    conn = sqlite3.connect(source_file + '/' + '726_.db')
    # Write SQL query to select all data from t_action2
    query = "SELECT * FROM t_actions2"
    # Use pandas read_sql_query() to execute the query and load data into a DataFrame
    df = pd.read_sql_query(query, conn)
    # Display the DataFrame
    print(df)
    # Close the connection
    conn.close()
    # Get the value counts of the 'stamp' column
    stamp_counts = df['stamp'].value_counts()
    # Filter stamps that appear more than 10 times
    stamps_to_keep = stamp_counts[stamp_counts > 10].index
    # Filter rows in the original DataFrame where the 'stamp' value is in stamps_to_keep
    df = df[df['stamp'].isin(stamps_to_keep)]

    video_path = os.path.join(source_file, 'students-full.mp4')
    time_stamp_list = set(df['stamp'])

    time_stamp_list = sorted(time_stamp_list)
    start_index=int(np.where(np.array(time_stamp_list)==122.0)[0])
    end_index=start_index+45
    time_stamp_list_=time_stamp_list[start_index:end_index]
    judgeclass=JudgeSuspiciousStandUp(track_frame_num=15)
    #judgeclass.main(video_path, source_file, df, time_stamp_list_)
    for i in time_stamp_list_:
        input_array = np.array(df.loc[df['stamp']==i])
        output_dict=judgeclass.main(input_array,i,video_path,source_file)
    print('finish')
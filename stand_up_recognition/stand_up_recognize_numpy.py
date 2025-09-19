import os.path
import time
from scipy.stats import zscore
import cv2,warnings,sqlite3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sort import *
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans,KMeans
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
    # (M,4) --> (4,M) --> (M,)
    area_true = box_area(boxes_true.T)
    # (N,4) --> (4,N) --> (N,)
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
        self.input_array_list=np.empty((0, 9))
        self.time_stamp_list=[]
        # each element is (time_stamp,detection data array)
        self.detect_array_queue = deque(maxlen=self.track_frame_num)
        # output dict, saving the judge result for each time stamp
        self.output_dict = {}
        # confidence score threshold
        self.detect_score_confidence=detect_confidence_threshold
        # record the group number algorithm handle
        self.group_num = 1



    def find_max_intersection_box(self, cur_box, b_list):
        """
        find the box has max union with cur_box whose central Y is upper than cur_box
        """
        if len(b_list) == 0:
            return None

        b_array = np.array(b_list)  # shape (N, 4)
        ious = box_iou_batch(np.array([cur_box]), b_array)[0]  # shape (N,)

        # 計算中心 y 座標
        act_cy = (cur_box[1] + cur_box[3]) / 2
        b_cy = (b_array[:, 1] + b_array[:, 3]) / 2

        # 過濾在 cur_box 上方的 box
        mask = b_cy < act_cy
        valid_ious = ious * mask  # mask 為 False 的會乘成 0

        max_idx = np.argmax(valid_ious)
        if valid_ious[max_idx] >= 0.15:
            return b_list[max_idx]
        else:
            return None


    def find_related_tracker(self,cur_box, previous_1_track_array):
        """
        match current frame untracked detection box and previous frame tracked detection box
        cur_box: untracked detection box coordinates
        previous_1_track_df: previous frame tracked detection box dataframe
        """
        if len(previous_1_track_array) == 0:
            return None

        ious = box_iou_batch(np.array([cur_box]), previous_1_track_array[:,0:4])[0]  # shape = (N,)
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        previous_bottom_y = previous_1_track_array[max_iou_idx, 3]
        cur_bottom_y = cur_box[3]

        if max_iou < 0.25 and (cur_bottom_y - previous_bottom_y) > 20:
            return None
        return max_iou_idx

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
        # return id_df_new (N,): if not:0, if suspicious to stand up:1
        Q1 = np.quantile(id_df[:,2],0.25)
        Q3 = np.quantile(id_df[:,2],0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        id_df_new = id_df[id_df[:,2] <= upper_bound,:]


        diff_y = np.max(id_df_new[:,2]) - np.min(id_df_new[:,2])
        if diff_y < 30 or len(id_df) < 8:
            return id_df_new

        kmeans =KMeans(n_clusters=2, random_state=42,n_init=1).fit(id_df_new[:,2].astype(np.float32).reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        if centers[0] > centers[1]:
            labels = 1 - labels
            centers = centers[::-1]
        gap = centers[1] - centers[0]
        t0_series = id_df_new[:,3][labels == 0]
        t0_series_diff = np.diff(t0_series)
        mask = np.isin(id_df_new[:, 3], t0_series)
        t0_act_judge1 = id_df_new[mask, -2] == 2
        t0_act_judge2 = id_df_new[mask, -2] == 3
        t0_act_judge3 = id_df_new[mask, -2] == 11
        t0_act_judge4 = id_df_new[mask, -2] == 0
        t0_act_judge_num = np.sum(t0_act_judge1)  + np.sum(t0_act_judge3)
        t0_act_judge_num2 = np.sum(t0_act_judge4)
        if t0_act_judge_num2 > 4:
            id_df_new[mask, -1]=1
        # only tolerate the time gap of 2 frames (at most 8s)
        elif gap > 30  and np.all(t0_series_diff < 13):
            id_df_new[mask, -1]=1

        return id_df_new

    def track(self,input_array_all,time_stamp_list):
        """
        the object tracking main process with SORT tracker
        input_array_all: object detection results with bounding box, confidence score, time stamp in multiple frames
        shape=(N,9), where 9='stamp','imgid','actid','x1','y1','x2','y2','cid','score'
        time_stamp_list: time stamp list need to track  in multiple frames
        """
        # initialize a new SORT tracker
        tracker = Sort(max_age=35, min_hits=2, iou_threshold=0.25)
        #columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid']
        person_series_array=np.empty((0,9))
        #columns=['stamp', 'hx1', 'hy1', 'hx2', 'hy2', 'id']
        head_series_array=np.empty((0,6))
        for time_idx, timestamp in enumerate(time_stamp_list):
            #sub_df = df[df['stamp'] == timestamp]
            sub_array=input_array_all[input_array_all[:,0]==timestamp,:]
            detections = np.empty((0, 6))
            # draw the detected results on the image
            for row in sub_array:
                cid = row[7]
                imgid = int(row[1])
                score = row[8]
                if imgid % 2 != 0 or score < self.detect_score_confidence:
                    continue
                # Get bounding box coordinates
                x1, y1, x2, y2 = int(row[3]), int(row[4]), int(row[5]), int(row[6])
                currentArray = np.array([x1, y1, x2, y2, score, cid])
                if cid == 9:
                    hid = -1000

                    sample_array_head=np.array([timestamp, x1, y1, x2, y2, hid]).reshape(1,-1)
                    head_series_array=np.concatenate([head_series_array,sample_array_head],axis=0)
                else:
                    detections = np.vstack((detections, currentArray))

            resultTracker, unmatched_dets = tracker.update(detections)
            # 2,  save the initial detection results and tracking results in dataframe
            for result in resultTracker:
                x1, y1, x2, y2, id,cid = result
                x1, y1, x2, y2 =int(x1),int(y1),int(x2),int(y2)

                # temperaly mask as unknown
                save_x, save_y = -100, -100
                sample_array = np.array([ [id, save_x, save_y, timestamp, x1, y1, x2, y2, cid]])
                person_series_array=np.concatenate([person_series_array,sample_array],axis=0)
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
                cur_id_array=person_series_array[person_series_array[:,3] == timestamp,0]
                previous_1_track_array=person_series_array[person_series_array[:,3] == previous_1_timestamp, 4::]
                previous_1_track_array_id =person_series_array[person_series_array[:,3] == previous_1_timestamp, :]
                for unmatch_row_id in unmatched_rows:
                    cur_box_ = detect_array[unmatch_row_id]
                    previous_match_tracker_index = self.find_related_tracker(cur_box_, previous_1_track_array)
                    if previous_match_tracker_index is not None:
                        #previous_match_tracker_id = previous_1_track_df_id.loc[previous_match_tracker_index].id
                        previous_match_tracker_id = previous_1_track_array_id[previous_match_tracker_index,0]
                        exists_in_cur = previous_match_tracker_id in cur_id_array
                        # make sure not deplicate, and record it
                        if not exists_in_cur:
                            x1, y1, x2, y2 = map(int,cur_box_)
                            cid  = sub_array[(sub_array[:,3]==x1) &(sub_array[:,4]==y1)&(sub_array[:,5]==x2) &(sub_array[:,6]==y2),-2][0]
                            sample_array=np.array([[previous_match_tracker_id, -100, -100, timestamp, x1, y1, x2, y2, cid]])
                            person_series_array=np.concatenate([person_series_array,sample_array],axis=0)
        # 3, combine different id for one object
        ids = person_series_array[:, 0]
        # ['bx1', 'by1', 'bx2', 'by2']
        bbox_data = person_series_array[:, 4:8]
        unique_ids = np.unique(ids)
        average_values_array = np.zeros((len(unique_ids), 5))

        for i, uid in enumerate(unique_ids):
            mask = ids == uid
            average_bbox = bbox_data[mask].mean(axis=0)
            average_values_array[i] = np.array([[uid, *average_bbox]])
        l2_distances = cdist(average_values_array[:,1::], average_values_array[:,1::], metric='euclidean')
        max_distance = np.max(l2_distances)
        l2_distances = l2_distances / max_distance
        upper_distance = np.triu(l2_distances, k=1)
        # finding the close bounding boxes series pairs
        indices = np.where((upper_distance < 0.05) & (upper_distance != 0))
        # Extract values based on the indices
        values = upper_distance[indices]
        # Map the indices back to the original ids
        id0_list = average_values_array[indices[0],0].astype(int)
        id1_list = average_values_array[indices[1],0].astype(int)
        all_id_list, id_counts = np.unique(person_series_array[:,0], return_counts=True)
        for id0, id1 in zip(id0_list, id1_list):
            id0_index= np.argwhere(all_id_list==id0)[0][0]
            id1_index= np.argwhere(all_id_list==id1)[0][0]
            id0_count = id_counts[id0_index]
            id1_count = id_counts[id1_index]
            if id0_count < 10 or id1_count < 10:
                if id0_count <= id1_count:
                    keep_id = id1
                    delete_id = id0
                else:
                    keep_id = id0
                    delete_id = id1
                # check per frame, make sure no duplication
                #  in frame t, keep id not exist then we can change the delete_id to keep_id
                delete_id_stamp = np.unique(person_series_array[person_series_array[:,0] == delete_id, 3])
                keep_id_stamp = np.unique(person_series_array[person_series_array[:,0] == keep_id, 3])
                intersection_stamp = np.intersect1d(delete_id_stamp, keep_id_stamp)
                change_flag = True
                if len(intersection_stamp) != 0 or len(delete_id_stamp) == 0 or len(keep_id_stamp) == 0:
                    change_flag = False

                if change_flag:
                    person_series_array[person_series_array[:,0] == delete_id, 0] = keep_id

        return person_series_array,head_series_array


    def JudgeAndVisualize(self,person_series_array,head_series_array,
                          video_path,source_file,save_series_imgs=True):

        stamp_list = np.unique(person_series_array[:,3])
        for stamp in stamp_list:
            sub_stamp_array = person_series_array[person_series_array[:,3] == stamp,:]
            sub_head_array = head_series_array[head_series_array[:,0] == stamp,:]
            head_detections = sub_head_array[:, 1:5]
            if video_path is not None:
                cap = cv2.VideoCapture(video_path)
                # Set the video position to the desired timestamp in milliseconds
                cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)  # Convert seconds to milliseconds
                # Read the frame
                ret, cur_frame = cap.read()
                if not os.path.exists(source_file + '/' +'{}'.format(self.group_num)):
                    os.mkdir(source_file + '/' +'{}'.format(self.group_num))

            for index, track_obj in enumerate(sub_stamp_array):
                x1, y1, x2, y2, id = track_obj[4], track_obj[5], track_obj[6], track_obj[7], track_obj[0]
                cur_box = np.array([x1, y1, x2, y2])
                corresponding_head = self.find_max_intersection_box(cur_box, head_detections)
                if corresponding_head is not None:
                    hx1, hy1, hx2, hy2 = int(corresponding_head[0]), int(corresponding_head[1]), int(corresponding_head[2]), int(corresponding_head[3])
                    head_series_array[(head_series_array[:,1] == hx1) & (head_series_array[:,3] == hx2) & (
                                head_series_array[:,2] == hy1) & (head_series_array[:,4] == hy2) & (
                                                   head_series_array[:,0] == stamp), 5] = id
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
                person_series_array[(person_series_array[:, 0] == id) & (person_series_array[:, 3] == stamp), 1] = save_x
                person_series_array[(person_series_array[:, 0] == id) & (person_series_array[:, 3] == stamp), 2] = save_y
            if video_path is not None :
                cv2.imwrite(source_file + '/' +'{}'.format(self.group_num) +'/'+'frame_at_{}.jpg'.format(stamp), cur_frame)

        #print(person_series_array)
        id_list = np.unique(person_series_array[:,0])

        if source_file is not None and (not os.path.exists(source_file + '/' +'{}'.format(self.group_num) +'/'+'id_plot')):
            os.mkdir(source_file + '/' +'{}'.format(self.group_num) +'/'+'id_plot')

        stand_candidate_array=np.zeros((person_series_array.shape[0],1))
        person_series_array = np.concatenate([person_series_array,stand_candidate_array],axis=1)
        for i, id in enumerate(id_list):
            if id==325:
                print(id)
            # columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid']
            id_df = person_series_array[person_series_array[:,0] == id].copy()
            id_df_new = self.suspicious_stand_up_judge(id_df)
            # Highlight standing candidates
            stand_points = id_df_new[id_df_new[:,-1] == 1]
            if len(stand_points) != 0:
                person_series_array=self.match_and_merge(person_series_array,id_df_new,match_cols = [3, 4, 5, 6, 7])
                if save_series_imgs:
                    id_array = person_series_array[person_series_array[:, 0] == id].copy()
                    series_img_path = source_file + '/' + str(id) + '_series_'+ str(id_array[0,3])
                    if not os.path.exists(series_img_path):
                        os.mkdir(series_img_path)
                    for data_line in id_array:
                        stamp=data_line[3]
                        bx1, by1, bx2, by2 = map(int, data_line[4:8])
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)  # Convert seconds to milliseconds
                        _,  raw_frame = cap.read()
                        height, width, _ = raw_frame.shape

                        expand_pixels=10
                        bx1 = max(bx1 - expand_pixels, 0)
                        by1 = max(by1 - expand_pixels, 0)
                        bx2 = min(bx2 + expand_pixels, width)
                        by2 = min(by2 + expand_pixels, height)

                        roi=raw_frame[by1:by2,bx1:bx2,:]
                        roi_save_path=os.path.join(series_img_path,str(stamp)+'.png')
                        cv2.imwrite(roi_save_path,roi)

            # Visualization
            if video_path is not None and source_file is not None:
                plt.figure(i)
                plt.scatter(id_df[:,1], id_df[:,2])
                plt.scatter(stand_points[:,1], stand_points[:,2], c='red')

                for x, y, cid, stamp in zip(id_df[:,1], id_df[:,2], id_df[:,8], id_df[:,3]):
                    text = '{}:{}'.format(int(stamp), int(cid))
                    plt.text(x, y, text, fontsize=6, ha='center', va='bottom', alpha=0.7)
                plt.xlabel('X position')
                plt.ylabel('Y Position')
                plt.savefig(f'{source_file}/{self.group_num}/id_plot/id_{id}.png', bbox_inches='tight')
                plt.close()
        return person_series_array
    @staticmethod
    def match_and_merge(person_arr, id_arr, match_cols):
        #  columns=['stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid']
        key_person = person_arr[:, match_cols]
        key_id = id_arr[:, match_cols]
        for i, row in enumerate(key_person):
            matches = np.all(key_id == row, axis=1)
            if np.any(matches):
                person_arr[i, -1] = id_arr[matches][0, -1]
        return person_arr


    def GetFormatOutput(self,person_series_array):
        """
        get the final recognition output result from person_series_df
        output array will be a 0-1 array, 0: not stand up 1: stand up suspiciously
        """
        stamp_list = np.unique(person_series_array[:,3])
        assert len(stamp_list) == len(self.detect_array_queue)
        for stamp,detect_array in self.detect_array_queue:
            input_array= detect_array[:,3:7]
            #innitial the output judge result
            output_judge_array=-1*np.ones(len(input_array))
            sub_stamp_array = person_series_array[person_series_array[:,3] == stamp]
            sub_stamp_array_= sub_stamp_array[:,4:8]
            #match the handled df result box to the input array box
            iou_matrix = box_iou_batch(input_array, sub_stamp_array_)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            # Array of all row indices
            all_row_indices = np.arange(iou_matrix.shape[0])
            # Rows that have been matched
            matched_rows = set(row_ind)
            # Unmatched rows
            unmatched_rows = list(set(all_row_indices) - matched_rows)
            #find the input order of each array
            judge_stand_up=sub_stamp_array[col_ind,-1]
            output_judge_array[row_ind] = judge_stand_up
            self.output_dict[stamp] =list(output_judge_array)

    def main(self,input_array,current_stamp,video_path=None,source_file=None):
        """
        The main process of stand-up recognition algorithm. When the input frame data array reach self.track_frame_num,
        it will output the result
        video_path: the source video, for visualization; can set to None to avoid it
        source_path: the saving path, for visualization; can set to None to avoid it
        current_stamp: the time stamp of current frame
        input_array: Nx9 array, 9='stamp','imgid','actid','x1','y1','x2','y2','cid','score'
          the input data of specific time stamp
        """
        self.frame_count+=1
        self.time_stamp_list.append(current_stamp)
        self.detect_array_queue.append((current_stamp,input_array))
        # initialize the output result in this time stamp
        self.output_dict[current_stamp]=-1*np.ones(len(input_array))
        output_dict = self.output_dict
        self.input_array_list=np.concatenate([self.input_array_list,input_array],axis=0)
        if self.frame_count == self.track_frame_num:
            # columns=['id', 'top_x', 'top_y', 'stamp', 'bx1', 'by1', 'bx2', 'by2', 'cid']
            person_series_array, head_series_array=self.track(self.input_array_list,self.time_stamp_list)
            person_series_array=self.JudgeAndVisualize(person_series_array, head_series_array,video_path,source_file)
            self.GetFormatOutput(person_series_array)
            # clear all the containers (self.time_stamp_list,self.detect_array_queue,self.input_df),
            # and reset the frame_count
            self.detect_array_queue.clear()
            self.frame_count=0
            self.input_array_list=self.input_array_list[0:0]
            self.time_stamp_list.clear()
            self.group_num+=1
            output_dict=self.output_dict.copy()
            self.output_dict.clear()
        return output_dict

if __name__ == '__main__':
    import pandas as pd
    source_file = './source9'
    video_path = os.path.join(source_file, 'students-full.mp4')
    # load raw saved data from db and visualize it
    # Connect to the SQLite database file
    conn = sqlite3.connect(source_file + '/' + 'session_2s.db')
    # Write SQL query to select all data from t_action2
    query = "SELECT * FROM t_actions2"
    # Use pandas read_sql_query() to execute the query and load data into a DataFrame
    df = pd.read_sql_query(query, conn)
    # Display the DataFrame
    #print(df)
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
    start_index=int(np.where(np.array(time_stamp_list)== 0.001)[0])
    end_index=start_index+90
    time_stamp_list_=time_stamp_list[start_index:end_index]
    judgeclass=JudgeSuspiciousStandUp(track_frame_num=15)
    for i in time_stamp_list_:
        input_array = np.array(df.loc[df['stamp']==i].iloc[:,:-1])
        output_dict=judgeclass.main(input_array,i,video_path,source_file)
    print('finish')





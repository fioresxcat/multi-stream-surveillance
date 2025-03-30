import pdb
import logging

class BaseContainerInfo:
    def __init__(self, cam_id, id, cam_fps, frame_size,  skip_frame: int = 1):
        self.cam_id = cam_id
        self.cam_fps = cam_fps
        self.id = id
        self.start_time = None
        self.end_time = None
        self.history = []
        self.camera_direction, self.moving_direction = None, None
        self.frame_w, self.frame_h = frame_size
        self.skip_frame = skip_frame
        self.is_pushed = False
        self.time_since_update = 0
        self.supported_cameras = ['htt', 'hts', 'bst', 'bss', 'hps', 'nct']

        self.min_appear_time = 1.5 # seconds
        self.logger = logging.getLogger(f'camera-{self.cam_id}')


    def __repr__(self):
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={self.info}, camera_direction={self.camera_direction}, num_appear={self.num_appear}, is_done={self.is_done})'
    

    def __str__(self):
        return self.__repr__()

    @property
    def num_appear(self):
        return len(self.history)
    

    @property
    def is_valid_container(self):
        """
        Check if the container is valid based on appearance count and movement direction.

        Returns:
            bool: True if the container meets the minimum appearance duration 
                and has a defined moving direction, False otherwise.
        """
        return self.num_appear >= int(self.cam_fps * self.min_appear_time) / self.skip_frame and self.moving_direction is not None
    

    def get_moving_direction(self, bboxes):
        """
        Determines the moving direction of the container based on bounding boxes.

        Args:
            bboxes (list of tuples): List of bounding boxes (x1, y1, x2, y2) over time.

        Returns:
            str: The moving direction ('l2r', 'r2l', 't2b', 'b2t') or None if undetermined.
        """
        if self.cam_id not in self.supported_cameras:
            raise NotImplementedError(f"Camera {self.cam_id} is not supported yet")

        MIN_VALID_PERCENT = 0.7
        DIFF_THRESHOLD = 3  # Minimum difference to consider movement

        if any(el in self.cam_id for el in ['htt', 'hts', 'hps', 'bst', 'bss']):
            # Horizontal movement (left-to-right or right-to-left)
            x_centers = [(bb[0] + bb[2]) / 2 for bb in bboxes]
            movements = [
                'l2r' if (x_centers[i + 1] - x_centers[i]) >= DIFF_THRESHOLD else 'r2l'
                for i in range(len(x_centers) - 1)
                if abs(x_centers[i + 1] - x_centers[i]) >= DIFF_THRESHOLD
            ]
        elif any(el in self.cam_id for el in ['nct', 'ncs']):
            # Vertical movement (top-to-bottom or bottom-to-top)
            y_centers = [(bb[1] + bb[3]) / 2 for bb in bboxes]
            movements = [
                't2b' if (y_centers[i + 1] - y_centers[i]) >= DIFF_THRESHOLD else 'b2t'
                for i in range(len(y_centers) - 1)
                if abs(y_centers[i + 1] - y_centers[i]) >= DIFF_THRESHOLD
            ]
        else:
            return None

        if not movements:
            return None

        # Count the occurrences of each movement direction
        direction_counts = {direction: movements.count(direction) for direction in set(movements)}
        # Determine the dominant direction
        for direction, count in direction_counts.items():
            if count / len(movements) > MIN_VALID_PERCENT:
                return direction

        return None


    def update_history(self, time_stamp, bb):
        """
        Updates the history of bounding boxes and determines the moving direction
        and camera direction of the container.

        Args:
            time_stamp (float): The timestamp of the current frame.
            bb (tuple): The bounding box of the container in the current frame.
        """
        MIN_TIME_TO_GET_DIRECTION = 1.2  # Minimum time (in seconds) to determine direction
        min_frames_required = int(MIN_TIME_TO_GET_DIRECTION * self.cam_fps) / self.skip_frame
        # Append the current bounding box and timestamp to the history
        self.history.append((time_stamp, bb))
        self.time_since_update = 0

        # Determine the moving direction if enough frames are available
        if len(self.history) >= min_frames_required:
            bboxes = [entry[1] for entry in self.history[0::2]]  # Take every second frame for speed calculation

            # If moving direction is not yet determined, calculate it
            if self.moving_direction is None:
                self.moving_direction = self.get_moving_direction(bboxes)

            # If moving direction is determined, infer the camera direction
            if self.moving_direction and self.camera_direction is None:
                self.camera_direction = self._infer_camera_direction()


    def _infer_camera_direction(self):
        """
        Infers the camera direction based on the moving direction and camera type.

        Returns:
            str: The inferred camera direction ('in' or 'out').
        """
        if self.cam_id not in self.supported_cameras:
            raise NotImplementedError(f"Camera {self.cam_id} is not supported yet")

        if any(el in self.cam_id for el in ['htt', 'bst', 'hps']):
            return 'out' if self.moving_direction == 'r2l' else 'in'
        elif any(el in self.cam_id for el in ['hts', 'bss']):
            return 'in' if self.moving_direction == 'r2l' else 'out'
        elif any(el in self.cam_id for el in ['nct', 'ncs']):
            return 'in' if self.moving_direction == 't2b' else 'out'
        else:
            raise NotImplementedError(f"Camera {self.cam_id} is not supported yet")


    def update_info(self, new_info):
        raise NotImplementedError("This method should be implemented in subclasses.")
    

    
class ContainerOCRInfo(BaseContainerInfo):
    def __init__(self, cam_id, id, cam_fps, frame_size, skip_frame):
        super().__init__(cam_id, id, cam_fps, frame_size, skip_frame)
        self.supported_cameras = [cam_id+'-ocr' for cam_id in self.supported_cameras]
        self.info = {
            'owner_code': (None, 0),  # value, score
            'container_number': (None, 0),
            'check_digit': (None, 0),
            'container_type': (None, 0),
            'rear_license_plate': (None, 0),
            'front_license_plate': (None, 0),
        }
        self.info_time_since_update = {field: 0 for field in self.info.keys()}
        self.score_threshold = 0.5


    def update_info(self, new_info):
        for label in self.info:
            if label not in new_info:
                if self.info[label][1] > 0:
                    self.info_time_since_update[label] += 1
            else:
                value, score = new_info[label]
                if score > self.info[label][1]:
                    self.info[label] = (value, score)
                    self.info_time_since_update[label] = 0
                elif self.info[label][1] > 0:  # score > 0 means this label has been update at least 1 time
                    self.info_time_since_update[label] += 1


    def get_complete_labels(self):
        """
        Retrieve complated fields based on their value, score, 
        and the time since their last update.

        This function filters labels from the `self.info` dictionary where:
        - The label's value is not None.
        - The label's score exceeds the `score_threshold`.
        - The time since the label's last update exceeds a calculated threshold 
          (`min_no_update_frame`), which is determined by the camera's frame rate 
          (`self.cam_fps`) and the `MIN_NO_UPDATE_TIME`.

        Returns:
            list: A list of labels that satisfy the above conditions.
        """
        MIN_NO_UPDATE_TIME = 1.5 # seconds
        min_no_update_frame = int(MIN_NO_UPDATE_TIME * self.cam_fps) / self.skip_frame

        complete_labels = []
        for label, (value, score) in self.info.items():
            if score > self.score_threshold and self.info_time_since_update[label] >= min_no_update_frame:  # no update in last n seconds
                complete_labels.append(label)
        return complete_labels


    def get_incomplete_labels(self):
        """
        Retrieve a list of labels that are incomplete based on the current camera direction.
        This method compares the labels in `self.info` with the complete labels obtained 
        from `self.get_complete_labels()` and filters out specific labels depending on 
        the camera direction ('in' or 'out').
        Returns:
            list: A list of incomplete labels after applying the camera direction filter.
        """
        # Filter incomplete labels based on camera direction
        complete_labels = self.get_complete_labels()
        incomplete_labels = [label for label in self.info if label not in complete_labels]
    
        if self.camera_direction == 'out':
            exclude_labels = ['front_license_plate']
        elif self.camera_direction == 'in':
            exclude_labels = {'owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate'}
        incomplete_labels = [label for label in incomplete_labels if label not in exclude_labels]
        
        return incomplete_labels
    

    @property
    def is_done(self):
        if self.camera_direction is None:
            return False
        required_labels = ['front_license_plate'] if self.camera_direction == 'in' else ['owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate']
        return set(self.get_complete_labels()) == set(required_labels)
    


class ContainerDefectInfo(BaseContainerInfo):
    def __init__(self, cam_id, id, cam_fps, frame_size, skip_frame):
        super().__init__(cam_id, id, cam_fps, frame_size, skip_frame)
        self.supported_cameras = [cam_id+'-defect' for cam_id in self.supported_cameras]

        self.max_final_results = 5
        self.max_cand_results = 50
        self.cand_results = []
        self.info = []
        self.max_image_buffer_size = 5  # ideally multiple of defect detector batch size
        self.image_buffer = []
        self.is_done = False
        self.last_timestamp = None


    def __repr__(self):
        print_info = [el['names'] for el in self.info]
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={print_info}, moving_direction={self.moving_direction}, num_appear={self.num_appear}, is_done={self.is_done})'


    def add_image_to_buffer(self, timestamp, im):
        self.image_buffer.append((timestamp, im))
        self.last_timestamp = timestamp


    def add_cand_result(self, image, boxes, scores, names, timestamp):
        self.cand_results.append({
            'image': image, 'boxes': boxes, 'scores': scores, 'names': names, 'timestamp': timestamp
        })


    def gather_final_results(self):
        """
        from cand_results, gather final results
        """
        self.cand_results.sort(key=lambda x: len(x['names']), reverse=True)
        last_timestamp = None
        index = 0
        min_time_diff = 0.2
        while len(self.info) < self.max_final_results and len(self.cand_results) > 0:
            res = self.cand_results[index]
            if len(res['names']) == 0:
                break
            if last_timestamp is None or res['timestamp'] - last_timestamp > min_time_diff:
                self.info.append(res)
                last_timestamp = res['timestamp']
            index += 1
            if index == len(self.cand_results):
                break

        self.is_done = True
    

    @property
    def is_full_buffer(self):
        return len(self.image_buffer) == self.max_image_buffer_size

    @property
    def is_full_result(self):
        return len(self.cand_results) == self.max_cand_results
class ContainerInfo:
    def __init__(self, cam_id, id, frame_size):
        self.cam_id = cam_id
        self.id = id
        self.start_time = None
        self.end_time = None
        self.info = {
            'owner_code': (None, 0),  # value, score
            'container_number': (None, 0),
            'check_digit': (None, 0),
            'container_type': (None, 0),
            'rear_license_plate': (None, 0),
            'front_license_plate': (None, 0),
        }
        self.history = []
        self.direction = None
        self.score_threshold = 0.5
        self.frame_w, self.frame_h = frame_size
        self.pushed_to_queue = False
        self.time_since_update = 0


    def __repr__(self):
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={self.info}, direction={self.direction}, num_appear={self.num_appear}, is_full={self.is_full})'
    

    def __str__(self):
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={self.info}, direction={self.direction}, num_appear={self.num_appear}, is_full={self.is_full})'


    @property
    def num_appear(self):
        return len(self.history)
    

    def update_info(self, new_info):
        for label in new_info:
            if label not in self.info:
                continue
            value, score = new_info[label]
            if score > self.info[label][1]:
            # if True:
                self.info[label] = (value, score)
    

    def update_history(self, time_stamp, bb):
        self.history.append((time_stamp, bb))
        self.time_since_update = 0
        if len(self.history) > 5 and self.direction is None:
            if self.cam_id in ['htt', 'hts']:
                first_bboxes = [el[1] for el in self.history[:5]]
                centers = [(bb[0] + bb[2]) / 2 for bb in first_bboxes]
                num_near_left = sum([1 if center - self.frame_w//2 < 0 else 0 for center in centers])
                num_near_right = sum([1 if center - self.frame_w//2 > 0 else 0 for center in centers])
                if self.cam_id in ['htt']:
                    self.direction = 'out' if num_near_right >= num_near_left else 'in'
                elif self.cam_id in ['hts']:
                    self.direction = 'in' if num_near_right >= num_near_left else 'out'
            else:
                raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')


    def get_complete_labels(self):
        complete_labels = []
        for label, (value, score) in self.info.items():
            if value is not None and score > self.score_threshold:
                complete_labels.append(label)
        return complete_labels


    def get_incomplete_labels(self):
        incomplete_labels = [label for label, (value, score) in self.info.items() if value is None or score < self.score_threshold]
    
        if self.direction == 'out':
            incomplete_labels = [label for label in incomplete_labels if label != 'front_license_plate']
        elif self.direction == 'in':
            exclude_labels = {'owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate'}
            incomplete_labels = [label for label in incomplete_labels if label not in exclude_labels]
        
        return incomplete_labels
    

    @property
    def is_full(self):
        if self.direction is None:
            return False
        required_labels = ['front_license_plate'] if self.direction == 'in' else ['owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate']
        return set(self.get_complete_labels()) == set(required_labels)
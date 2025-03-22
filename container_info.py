class ContainerInfo:
    def __init__(self, cam_id, id, cam_fps, frame_size):
        self.cam_id = cam_id
        self.cam_fps = cam_fps
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
        self.info_time_since_update = {field: 0 for field in self.info.keys()}
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
    

    def determine_direction(self, bboxes):
        centers = [(bb[0] + bb[2]) / 2 for _, bb in self.history]
        movements = ['l2r' if centers[i+1] - centers[i] >= 0 else 'r2l' for i in range(len(centers)-1)]
        num_rights, num_lefts = movements.count('l2r'), movements.count('r2l')
        if num_rights / len(movements) > 0.7:
            return 'l2r'
        elif num_lefts / len(movements) > 0.7:
            return 'r2l'
        else:
            return None
        

    @property
    def is_valid_container(self):
        """
            appear at least 10 times and do move in a single direction
        """
        if self.num_appear > 10 and self.determine_direction(self.history) is not None:
            return True

        return False


    def update_info(self, new_info):
        for label in self.info:
            if label not in new_info:
                self.info_time_since_update[label] += 1
            else:
                value, score = new_info[label]
                if score > self.info[label][1]:
                    self.info[label] = (value, score)
                    self.info_time_since_update[label] = 0
                else:
                    self.info_time_since_update[label] += 1


    def update_history(self, time_stamp, bb):
        self.history.append((time_stamp, bb))
        self.time_since_update = 0
        if len(self.history) > 5 and self.direction is None:
            if self.cam_id in ['htt', 'hts']:
                bboxes = [el[1] for el in self.history]
                moving_direction = self.determine_direction(bboxes)
                if self.cam_id in ['htt']:
                    self.direction = 'out' if moving_direction == 'r2l' else 'in'
                elif self.cam_id in ['hts']:
                    self.direction = 'in' if moving_direction == 'r2l' else 'out'
            else:
                raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')


    def get_complete_labels(self):
        complete_labels = []
        for label, (value, score) in self.info.items():
            if value is not None and score > self.score_threshold and self.info_time_since_update[label] > 1.5 * self.cam_fps:  # no update in 1.5 seconds
                complete_labels.append(label)
        return complete_labels


    def get_incomplete_labels(self):
        # incomplete_labels = [label for label, (value, score) in self.info.items() if value is None or score < self.score_threshold]
        complete_labels = self.get_complete_labels()
        incomplete_labels = [label for label in self.info if label not in complete_labels]
    
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
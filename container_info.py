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
        self.direction, self.moving_direction = None, None
        self.score_threshold = 0.5
        self.frame_w, self.frame_h = frame_size
        self.pushed_to_queue = False
        self.time_since_update = 0


    def __repr__(self):
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={self.info}, direction={self.direction}, num_appear={self.num_appear}, is_full={self.is_done})'
    

    def __str__(self):
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={self.info}, direction={self.direction}, num_appear={self.num_appear}, is_full={self.is_done})'


    @property
    def num_appear(self):
        return len(self.history)
    
    
    def get_moving_direction(self, bboxes):
        centers = [(bb[0] + bb[2]) / 2 for bb in bboxes]
        if self.cam_id in ['htt', 'hts', 'bst', 'bss', 'hps']:
            movements = []
            diff_threshold = 3
            for i in range(len(centers)-1):
                diff = centers[i+1] - centers[i]
                if abs(diff) < diff_threshold:
                    continue
                movements.append('l2r' if diff >= 0 else 'r2l')
            num_rights, num_lefts = movements.count('l2r'), movements.count('r2l')
            if len(movements) == 0:
                return None
            if num_rights / len(movements) > 0.7:
                return 'l2r'
            elif num_lefts / len(movements) > 0.7:
                return 'r2l'
            else:
                return None
        else:   
            raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')
                

    @property
    def is_valid_container(self):
        """
            appear at least 10 times and do move in a single direction
        """
        return self.num_appear >= 50 and self.moving_direction is not None
    

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
        if len(self.history) >= 10 and self.direction is None:
            if self.cam_id in ['htt', 'hts']:
                bboxes = [el[1] for el in self.history]
                self.moving_direction = self.get_moving_direction(bboxes)
                if self.moving_direction is not None:
                    if self.cam_id in ['htt']:
                        self.direction = 'out' if self.moving_direction == 'r2l' else 'in'
                    elif self.cam_id in ['hts']:
                        self.direction = 'in' if self.moving_direction == 'r2l' else 'out'
            else:
                raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')

        if len(self.history) >= 10 and self.moving_direction is None:
            # num_first_frames = int(1.2*self.cam_fps)
            bboxes = [el[1] for el in self.history[:]]
            self.moving_direction = self.get_moving_direction(bboxes)


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
    def is_done(self):
        if self.direction is None:
            return False
        required_labels = ['front_license_plate'] if self.direction == 'in' else ['owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate']
        return set(self.get_complete_labels()) == set(required_labels)
    


class ContainerDefectInfo:
    def __init__(self, cam_id, id, cam_fps, frame_size, max_frame_result=3):
        self.cam_id = cam_id
        self.cam_fps = cam_fps
        self.id = id
        self.start_time = None
        self.end_time = None
        self.max_frame_result = max_frame_result
        self.info = [
            {'image': None, 'boxes': [], 'scores': [], 'cl_names': []} for _ in range(self.max_frame_result)
        ]
        self.images = []
        self.history = []
        self.direction, self.moving_direction = None, None
        self.frame_w, self.frame_h = frame_size
        self.pushed_to_queue = False
        self.time_since_update = 0
        self.is_done = False


    def __repr__(self):
        print_info = [el['cl_names'] for el in self.info]
        return f'ContainerInfo(id={self.id}, start_time={self.start_time}, info={print_info}, direction={self.direction}, num_appear={self.num_appear}, is_full={self.is_full})'
    

    def __str__(self):
        return self.__repr__()


    @property
    def num_appear(self):
        return len(self.history)
    

    def get_moving_direction(self, bboxes):
        centers = [(bb[0] + bb[2]) / 2 for bb in bboxes]
        if self.cam_id in ['htt', 'hts', 'bst', 'bss', 'hps']:
            movements = ['l2r' if centers[i+1] - centers[i] >= 0 else 'r2l' for i in range(len(centers)-1)]
            num_rights, num_lefts = movements.count('l2r'), movements.count('r2l')
            if num_rights / len(movements) > 0.7:
                return 'l2r'
            elif num_lefts / len(movements) > 0.7:
                return 'r2l'
            else:
                return None
        else:   
            raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')
                

    @property
    def is_valid_container(self):
        """
            appear at least 10 times and do move in a single direction
        """
        return self.num_appear >= 50 and self.moving_direction is not None
        

    def update_image(self, timestamp, im):
        self.images.append((timestamp, im))


    def update_info(self, new_info):
        assert len(new_info) == len(self.images)
        for index, (im, (boxes, scores, cl_names)) in enumerate(zip(self.images, new_info)):
            # self.info[index]['image'] = im
            self.info[index]['boxes'] = boxes
            self.info[index]['scores'] = scores
            self.info[index]['cl_names'] = cl_names
        self.is_done = True


    def update_history(self, timestamp, bb):
        self.history.append((timestamp, bb))
        self.time_since_update = 0
        if len(self.history) >= 10 and self.direction is None:
            if self.cam_id in ['htt', 'hts', 'hps']:
                # num_first_frames = int(1.2*self.cam_fps)
                bboxes = [el[1] for el in self.history[:]]
                self.moving_direction = self.get_moving_direction(bboxes)
                if self.moving_direction is not None:
                    if self.cam_id in ['htt', 'hps']:
                        self.direction = 'out' if self.moving_direction == 'r2l' else 'in'
                    elif self.cam_id in ['hts']:
                        self.direction = 'in' if self.moving_direction == 'r2l' else 'out'
            else:
                raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')

        if len(self.history) >= 10 and self.moving_direction is None:
            # num_first_frames = int(1.2*self.cam_fps)
            bboxes = [el[1] for el in self.history[:]]
            self.moving_direction = self.get_moving_direction(bboxes)


    @property
    def is_full(self):
        return len(self.images) >= self.max_frame_result
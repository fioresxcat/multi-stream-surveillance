import pdb
import cv2
import numpy as np

from utils.utils import total_time
from modules.base import BaseModule


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, img):
        """Return updated labels and image with added border."""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        return img
    



class BaseObjectDetection(BaseModule):
    def __init__(self, common_config, model_config):
        super(BaseObjectDetection, self).__init__(common_config, model_config)
        
        
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    
    def nms(self, dets, scores, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0] # pick maxmum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou=0.45,
            agnostic=False,
            multi_label=False,
            max_det=300,
            nc=0,  # number of classes (optional)
            max_time_img=0.05,
            max_nms=100,
            max_wh=7680,
    ):
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        # time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        # t = time()
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose()[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, classes, mask = np.split(x, (4, 4+nc), axis=1)
            box = self.xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

            if multi_label:
                i, j = np.where(classes > conf_thres)
                x = np.cat((box[i], x[i, 4 + j, np.newaxis], j[:, np.newaxis].astype(float), mask[i]), axis=1)
            else:  # best class only
                # conf, j = classes.max(axis=1, keepdims=True)
                conf = np.max(classes, axis=1, keepdims=True)
                j = np.expand_dims(np.argmax(classes, axis=1), axis=-1)
                x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.reshape(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms(boxes, scores, iou)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
            # if (time() - t) > time_limit:
            #     break  # time limit exceeded

        return output


    def clip_coords(self, coords, shape):
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
        return coords


    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None, normalize=False):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
        coords[..., 0] /= gain
        coords[..., 1] /= gain
        coords = self.clip_coords(coords, img0_shape)
        if normalize:
            coords[..., 0] /= img0_shape[1]  # width
            coords[..., 1] /= img0_shape[0]  # height
        return coords
    
    
    def scale_boxes(self, img1_shape, boxes, img0_shape):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        # clip_boxes(boxes, img0_shape)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
        return boxes


    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect


    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    def sort_box(self, boxes, kps):
        bbs_clusters = [(b, k) for b, k in zip(boxes, kps)]
        bbs_clusters.sort(key=lambda x: x[0][0])

        clusters, y_min, cluster_kps = [], [], []
        for tgt_node, kp,  in bbs_clusters:
            if len (clusters) == 0:
                clusters.append([tgt_node])
                cluster_kps.append([kp])
                y_min.append(tgt_node[1])
                continue
            matched = None
            for idx, clt in enumerate(clusters):
                src_node = clt[-1]
                overlap_y = ((src_node[3] - src_node[1]) + (tgt_node[3] - tgt_node[1])) - (max(src_node[3], tgt_node[3]) - min(src_node[1], tgt_node[1]))
                overlap_x = ((src_node[2] - src_node[0]) + (tgt_node[2] - tgt_node[0])) - (max(src_node[2], tgt_node[2]) - min(src_node[0], tgt_node[0]))
                if overlap_y > 0.5*min(src_node[3] - src_node[1], tgt_node[3] - tgt_node[1]) and overlap_x < 0.6*min(src_node[2] - src_node[0], tgt_node[2] - tgt_node[0]):
                    distance = tgt_node[0] - src_node[2]
                    if matched is None or distance < matched[1]:
                        matched = (idx, distance)
            if matched is None:
                clusters.append([tgt_node])
                cluster_kps.append([kp])
                y_min.append(tgt_node[1])
            else:
                idx = matched[0]
                clusters[idx].append(tgt_node)
                cluster_kps[idx].append(kp)
        zip_clusters = list(zip(clusters, y_min, cluster_kps))
        zip_clusters.sort(key=lambda x: x[1])
        # return boxes in order
        sorted_boxes, sorted_kps = [], []
        for cluster in zip_clusters:
            for box, kp in zip(cluster[0], cluster[2]):
                sorted_boxes.append(box)
                sorted_kps.append(kp)
        return sorted_boxes, sorted_kps
    
    def is_poly_belong(self, text, block):
        xmin_block, ymin_block, xmax_block, ymax_block = block
        x1, y1 = text[0]
        x2, y2 = text[1]
        x3, y3 = text[2]
        x4, y4 = text[3]
        xmin_text = min(x1, x2, x3, x4)
        xmax_text = max(x1, x2, x3, x4)
        ymin_text = min(y1, y2, y3, y4)
        ymax_text = max(y1, y2, y3, y4)
        x_center = xmin_text + (xmax_text - xmin_text)/2.0
        y_center = ymin_text + (ymax_text - ymin_text)/2.0
        if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
            return True
        return False

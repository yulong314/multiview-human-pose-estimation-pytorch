import os.path as osp
import numpy as np
import pickle
import collections
from xtcocotools.coco import COCO

from .joints_dataset import JointsDataset
#serials,intrinsics, extrinsics, Distortions
class Multiview_coco_h36m(JointsDataset):
    def __init__(self, cfg, image_set, is_train, annfile, img_prefix, serials, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.annfile = annfile
        self.img_prefix = img_prefix
        self.serials = serials
        self.dataset_name = 'Multiview_coco_h36m'
        self.num_joints = 1
        self.img_prefix = img_prefix
        self.idxLoaded =[]
        self.actual_joints = {0:'elbow'}
        if is_train:
            self.times = 10
        else:
            self.times = 1
        coco_style = True
        if coco_style:
            self.coco = COCO(annfile)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = self._get_db()
        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        self.u2a_mapping = {0:0}
        pass

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.num_joints
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                assert keypoints[:, 2] == 2, 'keypoints[:, 2] must be 2'
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                # center, scale = self._xywh2cs(*obj['bbox'])
                box = obj['bbox']
                center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)                
                gt_db.append({
                    'image': image_file,
                    'rotation': 0,
                    'joints_2d': joints_3d,
                    'joints_3d': joints_3d,
                    'joints_vis': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id,
                    'center':center,
                    'scale':scale,
                    'source': 'coco',
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _xywh2cs(self, x, y, w, h, padding=1.0):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """



        # aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
        #     'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # if (not self.test_mode) and np.random.rand() < 0.3:
        #     center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        # if w > aspect_ratio * h:
        #     h = w * 1.0 / aspect_ratio
        # elif w < aspect_ratio * h:
        #     w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale 

        return center, scale        

    def _get_key_str(self, filepath):
        # "file_name": "../../unmerged/dongshiling/val/color/4100/00033.jpg"
        pathlist = filepath.split('/')
        person = pathlist[-4]
        filename = osp.basename(filepath)
        cur_dir = osp.dirname(filepath)
        curview = osp.basename(cur_dir)
        # parent_dir = osp.dirname(cur_dir)
        camera_id = self.serials.index(curview)
        frameNumStr = filename.split(".")[0]
        digitPart = ''.join(filter(str.isdigit, frameNumStr))
        # suffix = ''.join(filter(str.isalpha,frameNumStr))
        # frameNum = int(digitPart)      
        key=person+digitPart
        return key, camera_id

    def get_group(self, db):
        """group images which taken at the same time"""
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            imgpath = db[i]['image']
            keystr,camera_id = self._get_key_str(imgpath)
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(self.serials)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        npgroup = np.array(filtered_grouping)
        sum = np.count_nonzero(npgroup, axis=0)
        # if self.is_train:
        #     filtered_grouping = filtered_grouping[::5]
        # else:
        #     filtered_grouping = filtered_grouping[::64]

        return filtered_grouping
        pass
    def __getitem__(self, idx):
        idx1 = idx % self.group_size
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx1]
        for index, item in enumerate(items):
            i, t, w, m = super().__getitem__(item)
            if m is not None:
                input.append(i)
                target.append(t)
                weight.append(w)
                m['camindex'] = index
                meta.append(m)
        self.idxLoaded.append(idx)
        return input, target, weight, meta
    def __len__(self):
        return self.group_size * self.times 

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
        gt = np.array(gt)
        pred = pred[:, su, :2]

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)

def test():
    from core.config import config
    import os
    print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))    
    #annfile, img_prefix, serials
    serials = "4105,4097,4112,4102,4103,4113,4101,4114,4100,4099,4098".split(",")
    annfile = "/2t/data/recordedSamples/samples/20220708/merged/19/val.json"
    img_prefix = osp.dirname(annfile)
    coco_h36m = Multiview_coco_h36m( config, config.DATASET.TRAIN_SUBSET, annfile, img_prefix, serials, True)
    pass
if __name__ == '__main__':
    test()
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from imantics import Mask
import cv2
import numpy as np
import json
import os
from scipy.spatial import ConvexHull
from detectron2.projects import point_rend
from shapely.geometry import Polygon, Point

# --------------------- tham số RUN-------------------
class_name = ['daydien', 'cachdienslc', 'cachdientt', 'cotthephinh', 'cotdonthan', 'tacr', 'daycs','cachdientt:vobat']
class_choice = ['daydien', 'cachdienslc', 'cachdientt', 'cotthephinh', 'cotdonthan', 'tacr', 'daycs','cachdientt:vobat']
threshold = 0.65
path_kq_DT = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\DT'
path_to_jpg = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\GT'
path_model = "../model/model_0189999.pth"
# ------------------------------------------------------------------------------

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("../detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.DATASETS.TRAIN = ("train_data",)
cfg.DATASETS.TEST = ("test_data",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, path_model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.BATH_SIZE_PER_IMAGE = 128
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 8
cfg.INPUT.MIN_SIZE_TEST = 416
cfg.INPUT.MAX_SIZE_TEST = 608
predictor = DefaultPredictor(cfg)

'''
chuyển mask color dị vật thành polygon bao dị vật
'''
def mask_to_polygons(mask):
    mang = 1 * mask * 255  # chuyển mang type bool (True,False) về dạng (225,0)
    polygons = Mask(mang).polygons()

    if polygons.mask().area() > 0:
        res_all = []
        mang_new = []
        for polygon in polygons:
            res = [[polygon[i * 2], polygon[i * 2 + 1]] for i in range(int(len(polygon) / 2))]
            res_all.append(res)

        area_max = 0
        poly_choice = []
        for re in res_all:
            # print('re : ', re)
            # print('len(re) : ', len(re))
            if len(re) >= 4 and Polygon(re).area > area_max:
                area_max = Polygon(re).area
                poly_choice = re
        mang_new = poly_choice
        mang_new = np.array(mang_new).astype(float).tolist()

    else:
        mang_new = []

    return mang_new


class Metadata:
    def get(self, _):
        return class_name


labels_edit = [i.replace(':', '_') for i in class_name]

json_files = [pos_jpg for pos_jpg in os.listdir(path_to_jpg) if pos_jpg.endswith('.jpg')]

for img3 in json_files:
    '''' # ---------------- lấy lại form file json cũ -----------------------------'''
    ''''
    Đọc lại file json cũ để kế thừa cấu trúc của file json để k phải tạo mới
    '''
    with open(f'./DIJ_00A.json') as file:
        # Load its content and make a new dictionary
        data = json.load(file)
    data['shapes'].clear()
    # -----------------------------------------------------------------------
    print(f'--------------------------Anh detect {img3}------------------------------------------')
    im = cv2.imread(f'{path_to_jpg}/{img3}')
    outputs = predictor(im)

    # Select the first two classes from the COCO dataset
    COCO_CLASSES = class_name
    COCO_CLASSES_subset = class_choice

    # Filter the segmentation results
    instances = outputs["instances"]
    selected_indices = [
        i for i, label in enumerate(instances.pred_classes)
        if COCO_CLASSES[label.item()] in COCO_CLASSES_subset
    ]
    instances = instances[selected_indices]
    instances = instances[instances.scores > threshold]

    if len(instances.pred_classes) > 0:

        v = Visualizer(im[:, :, ::-1], Metadata, scale=1.2)
        v = v.draw_instance_predictions(instances.to("cpu"))
        img = v.get_image()[:, :, ::-1]
        result_out = []

        for i in range(len(labels_edit)):
            vars()['result_' + labels_edit[i] + '_all'] = []
        for (bbox_raw, score, class_idx, masks) in zip(instances.pred_boxes.tensor,
                                                       instances.scores,
                                                       instances.pred_classes,
                                                       instances.pred_masks):
            # print('masks : ', masks)
            # print('name : ', class_name[class_idx.cpu().numpy()])

            for i in range(len(labels_edit)):
                if class_idx.cpu().numpy() == i:
                    vars()['result_' + labels_edit[i] + '_all'].append([class_name[class_idx.cpu().numpy()], mask_to_polygons(masks)])

        result_tong = []
        for i in range(len(labels_edit)):
            result_tong.append(vars()['result_' + labels_edit[i] + '_all'])

        for result_one in result_tong:
            for result in result_one:

                data['shapes'].append({"label": result[0],
                                       "points": result[1],
                                       "group_id": None,
                                       "shape_type": "polygon",
                                       "flags": {}})

        for i in range(len(labels_edit)):
            del globals()['result_' + labels_edit[i] + '_all']

        data['shapes'] = [one for one in data['shapes'] if len(one['points']) > 0]
        a = img3.strip(".jpg")

        data['version'] = "4.5.9"
        data['imagePath'] = f"DT_{a}.jpg"
        import base64

        cv2.imwrite(f'{path_kq_DT}/DT_{a}.jpg', im)
        encoded = base64.b64encode(
            open(f'{path_kq_DT}/DT_{a}.jpg', "rb").read())
        data['imageData'] = f"{encoded.decode('utf-8')}"
        with open(f'{path_kq_DT}/DT_{a}.json', 'w') as file:
            json.dump(data, file, indent=4)
    else:
        a = img3.strip(".jpg")

        data['version'] = "4.5.9"
        data['imagePath'] = f"DT_{a}.jpg"
        import base64

        cv2.imwrite(f'{path_kq_DT}/DT_{a}.jpg', im)
        encoded = base64.b64encode(
            open(f'{path_kq_DT}/DT_{a}.jpg', "rb").read())
        data['imageData'] = f"{encoded.decode('utf-8')}"
        with open(f'{path_kq_DT}/DT_{a}.json', 'w') as file:
            json.dump(data, file, indent=4)


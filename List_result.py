import os, json, sys
import cv2


def check_json(path):
    path_to_json = path
    jpg_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.jpg')]
    for i in jpg_files:
        a = i.strip(".jpg")
        if os.path.isfile(f'{path}/{a}.json') is not True:
            sys.exit(f"anh {i} trong folder Image_GT chua dc gan nhan")


def List_result(path):
    path_to_json = path
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    mang_all = []
    for i in json_files:
        with open(f'{path}/{i}') as file:
            data = json.load(file)

        mang = []
        for n in range(len(data['shapes'])):
            mang.append([data['shapes'][n]['label'], data['shapes'][n]['points'],  data['imagePath']])

        mang_all.append(mang)
    return mang_all



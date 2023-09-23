from shapely.geometry import Polygon, MultiPolygon
from List_result import List_result, check_json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json
import sys
from scipy.spatial import ConvexHull
from shapely.validation import make_valid
''''#Hàm trả về chỉ số IoU giữa 2 polygons và polygons giao giữa 2 polygons cho trước'''

# --------------------- tham số RUN-------------------
path_kq = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\KQ'
path_GT = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\GT'
path_DT = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\DT'
# ------------------------------------------------------------------------------

def find_iou(polygon1, polygon2):
    polygon3 = Polygon(polygon1).buffer(0.001)
    polygon4 = Polygon(polygon2).buffer(0.001)
    intersect = polygon3.intersection(polygon4).area
    # print('intersect : ', intersect)
    listPoly = []
    poly = []
    # print('polygon3.intersection(polygon4) : ', type(polygon3.intersection(polygon4)))
    if isinstance(polygon3.intersection(polygon4), Polygon):
        for x, y in (polygon3.intersection(polygon4)).exterior.coords:
            listPoly.append([int(x), int(y)])
    if isinstance(polygon3.intersection(polygon4), MultiPolygon):
        # print('polygon3.intersection(polygon4) : ', polygon3.intersection(polygon4))
        for poly_one in (polygon3.intersection(polygon4)).geoms:
            for x, y in poly_one.exterior.coords:
                poly.append([int(x), int(y)])
        # print('listPoly : ', np.array(poly))
        poly = np.array(poly)
        hull = ConvexHull(poly)
        for simplex in hull.vertices:
            listPoly.append([poly[simplex, 0], poly[simplex, 1]])

        # listPoly = np.array(listPoly).tolist()
        # print('listPoly : ', (listPoly))
    if isinstance(polygon3.intersection(polygon4), MultiPolygon) is False and isinstance(polygon3.intersection(polygon4), Polygon) is False:
        sys.exit(f"check file json trong folder Image_DT")
        # pass
    union = polygon3.union(polygon4).area
    iou = intersect / union
    return iou, listPoly


def convert_mang(poly):
    mang = []
    for i in poly:
        mang.append((i[0], i[1]))
    return mang


''''# hàm tô màu polygons'''


def draw_color(img, poly, color):
    drw = ImageDraw.Draw(img, 'RGBA')
    drw.polygon(xy=poly, fill=color)
    del drw


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


''''# Hàm tô màu hộp ghi chú của tứng đối tượng'''


def draw_text_rectangle(img, toado_text, text, toa_box, color_text, color_box, size_font):
    I1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('./font/FreeMono.ttf', size_font)
    shape = bounding_box(toa_box)
    I1.rectangle(shape, outline=color_box, width=3)
    bbox = I1.textbbox(toado_text, text, font=myFont)
    I1.rectangle(bbox, outline=color_box, fill=color_box)
    I1.text(toado_text, text, font=myFont, fill=color_text, stroke_width=1)


dictionary = {
     "shapes": [

     ]
}

path_KQ = path_kq

''''# khởi tạo một file json'''
with open(f"{path_KQ}/annotations.json", "w") as outfile:
# with open("D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/KQ_IOU/annotations.json", "w") as outfile:
    json.dump(dictionary, outfile, indent=4)

''''# đọc file json vừa khởi tạo để thêm thông tin'''
# with open('D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/KQ_IOU/annotations.json') as file:
with open(f'{path_KQ}/annotations.json') as file:
    data = json.load(file)

''''# đọc vào ảnh và file json của 2 folder GT <tập ảnh dc gán nhã trước> va DT<tập ảnh detect ra> '''
# List_folder = ['D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/Image_GT', 'D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/Image_DT']
List_folder = [path_GT, path_DT]
check_json(List_folder[0])
mang_goc = List_result(List_folder[0])
mang_detect = List_result(List_folder[1])

''''# Lấy từng file json'''
for mang_one_goc, mang_one_detect in zip(mang_goc, mang_detect):
    name = mang_one_goc[0][2]
    print('detect_image : ', name)
    if len(mang_one_goc) == 0:
        sys.exit(f"anh {name} trong folder Image_GT chua dc gan nhan")
    a = mang_one_goc[0][2].strip(".jpg")
    ''''# đường dẫn lưu file txt'''

    # path = f'D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/KQ_IOU/{a}.txt'
    path = f'{path_KQ}/{a}.txt'
    img_gt = Image.open(f'{List_folder[0]}/{mang_one_goc[0][2]}')
    img_dt = Image.open(f'{List_folder[0]}/{mang_one_goc[0][2]}')
    img_iou = Image.open(f'{List_folder[0]}/{mang_one_goc[0][2]}')

    fig = plt.figure(figsize=(15, 10))
    rows = 2
    columns = 2
    danh_gia_goc_all = []
    danh_gia_detect_all = []
    check_repeat = []
    check_repeat_result_goc = []

    ''''# sẽ có 2 TH xảy ra ảnh được detect sẽ detect ra đối tượng hoặc k detect dc đối tượng nào cả 
    # với TH ảnh detect , detect ra đối tượng
    - Ta se duyệt qua từng đối tượng trong ảnh GT <ảnh được gán nhãn > Xem nó thuộc trường hợp nào trong 3 trường hợp 
    đối với ảnh GT : Good , No confidence ,No detect
    Điều kiện để xếp vào từng trường hợp :
    1, detect Good : IoU > 60 , đối tượng dc đưa ra so IoU phải cùng tên
    2, detect No confidence : IoU <= 60 , đối tượng dc đưa ra so IoU phải cùng tên
    3, detect No detect : có 2 trường hợp tạo thành
        3.1 : chỉ số IoU Max == 0
        3.2 : chỉ số IoU Max > 0 , nhưng đối tượng dc đưa ra so IoU không cùng tên
    
    Trong đối tượng của ảnh GT hoặc ảnh DT có nhiều nhãn đối tượng trên cùng 1 đối tượng (bài toán xém , hay vỡ bát)
    nên 1 đối tượng có thể đóng 2 vai trò (goog , no confidence ) or no detect việc này làm cho đối tượng kiểm tra k dc
    minh bạch về bản chất . nên những đối tượng nào đã kiểm tra với vai trò (good , no confidence ) rồi thì k xét để 
    kiểm tra no detect nữa.
    
    chú ý : một đối tượng có thể có nhiều nhãn dc xét là good hay no confidence
     
    '''
    if len(mang_one_detect) > 0:
        examination_object = []
        global mang_one_goc_new
        ''''# vẽ màu và nhãn nên đối tượng thuộc tập ảnh GT'''
        for idx_goc, result_goc in enumerate(mang_one_goc):  # duy cac doi tuong trong anh

            draw_color(img_gt, poly=convert_mang(result_goc[1]), color=(0, 255, 0, 125))
            draw_text_rectangle(img_gt, (bounding_box(result_goc[1])[0][0], (bounding_box(result_goc[1])[0][1] - 35) if (bounding_box(result_goc[1])[0][1] - 35) > 0 else (bounding_box(result_goc[1])[0][1] - 35) + 35), result_goc[0], bounding_box(result_goc[1]), (0, 0, 0), (255, 255, 0), 30)

        ''''#kiểm tra vị trí có nhãn đối tượng là good và no confidence để sau này 
        loại bỏ k dùng để xét no detect nữa'''

        for idx_goc, result_goc in enumerate(mang_one_goc):  # duy cac doi tuong trong anh
            for idx, result_detect in enumerate(mang_one_detect):
                iou = find_iou(result_goc[1], result_detect[1])[0]
                #  danh_gia_goc = [f'Image: {result_goc[2]}', f'GT: {result_goc[0]}', f'DT: {result_detect[0]}',
                #  f'IoU: {round(iou, 2)}'] #  Tam thoi k show ten anh
                danh_gia_goc = [f'GT: {result_goc[0]}', f'DT: {result_detect[0]}', f'IoU: {round(iou, 2)}']
                danh_gia_goc_dk = [result_goc[2], result_goc[0], result_detect[0], round(iou, 2)]
                if (danh_gia_goc_dk[1] == danh_gia_goc_dk[2]) and (danh_gia_goc_dk[3] > 0.60):
                    check_repeat_result_goc.append(idx_goc)
                    check_repeat.append(idx)
                if (danh_gia_goc_dk[1] == danh_gia_goc_dk[2]) and (0 < danh_gia_goc_dk[3] <= 0.60):
                    check_repeat_result_goc.append(idx_goc)
                    check_repeat.append(idx)
        # -----------------------------------------------------------------------------------------------

        ''''# xóa nhưng đối tượng ở vị trí good , no confiden trong ảnh GT dc lưu trong mảng [mang_one_goc] đồng thời tạo ra
        # mảng [mang_one_goc_new] chứa đầy đủ các đối tượng'''
        ds_del = []
        for i in range(len(list(set(check_repeat_result_goc)))):
            del_one = mang_one_goc.pop(sorted(list(set(check_repeat_result_goc)))[i] - i)
            ds_del.append(del_one)
        mang_one_goc_new = ds_del + mang_one_goc

        # -------------------------------------------------------------------------------------------------------------

        ''''# vẽ màu lên các đối tượng ở vị trí có labels good và no confidence'''
        for idx_goc, result_goc in enumerate(mang_one_goc_new):  # duy cac doi tuong trong anh
            list_iou = []
            for idx, result_detect in enumerate(mang_one_detect):
                iou = find_iou(result_goc[1], result_detect[1])[0]
                list_Poly = find_iou(result_goc[1], result_detect[1])[1]
                #danh_gia_goc = [f'Image: {result_goc[2]}', f'GT: {result_goc[0]}', f'DT: {result_detect[0]}',f'IoU: {round(iou, 2)}']
                danh_gia_goc = [f'GT: {result_goc[0]}', f'DT: {result_detect[0]}',  f'IoU: {round(iou, 2)}'] #  Tam thoi k show ten anh
                danh_gia_goc_dk = [result_goc[2], result_goc[0], result_detect[0], round(iou, 2)]
                list_iou.append(iou)

                if (danh_gia_goc_dk[1] == danh_gia_goc_dk[2]) and (danh_gia_goc_dk[3] > 0.60):
                    danh_gia_goc.append("evaluate : Good")
                    danh_gia_goc_all.append(danh_gia_goc)
                    giao_iou_check = list_Poly
                    draw_color(img_iou, poly=convert_mang(giao_iou_check), color=(255, 0, 255, 125))
                    draw_text_rectangle(img_iou, (bounding_box(giao_iou_check)[0][0], (bounding_box(giao_iou_check)[0][1] - 35) if (bounding_box(giao_iou_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_check)[0][1] - 35) + 35),
                                        "Good", bounding_box(giao_iou_check), (0, 0, 0), (255, 255, 0), 30)
                    examination_object.append({
                        "GT": f'{danh_gia_goc_dk[1]}',
                        "DT": f'{danh_gia_goc_dk[2]}',
                        "IoU": f'{danh_gia_goc_dk[3]}',
                        "evaluate": 'Good'
                    })

                if (danh_gia_goc_dk[1] == danh_gia_goc_dk[2]) and (0 < danh_gia_goc_dk[3] <= 0.60):
                    danh_gia_goc.append("evaluate : No confidence")
                    danh_gia_goc_all.append(danh_gia_goc)
                    giao_iou_check = list_Poly
                    draw_color(img_iou, poly=convert_mang(giao_iou_check), color=(128, 128, 0, 125))
                    draw_text_rectangle(img_iou, (bounding_box(giao_iou_check)[0][0], (bounding_box(giao_iou_check)[0][1] - 35) if (bounding_box(giao_iou_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_check)[0][1] - 35) + 35), "No confidence", bounding_box(giao_iou_check), (0, 0, 0), (255, 255, 0), 30)
                    examination_object.append({
                        "GT": f'{danh_gia_goc_dk[1]}',
                        "DT": f'{danh_gia_goc_dk[2]}',
                        "IoU": f'{danh_gia_goc_dk[3]}',
                        "evaluate": 'No confidence'
                    })


        ''''# vẽ màu lên các đối tượng có label dc xét là no detect'''

        for idx_goc, result_goc in enumerate(mang_one_goc):  # duy cac doi tuong trong anh
            list_iou = []
            for idx, result_detect in enumerate(mang_one_detect):
                iou = find_iou(result_goc[1], result_detect[1])[0]
                list_Poly = find_iou(result_goc[1], result_detect[1])[1]
                #danh_gia_goc = [f'Image: {result_goc[2]}', f'GT: {result_goc[0]}', f'DT: {result_detect[0]}',f'IoU: {round(iou, 2)}']
                danh_gia_goc = [f'GT: {result_goc[0]}', f'DT: {result_detect[0]}',  f'IoU: {round(iou, 2)}'] #  Tam thoi k show ten anh
                danh_gia_goc_dk = [result_goc[2], result_goc[0], result_detect[0], round(iou, 2)]
                list_iou.append(iou)

            tam_goc = np.array(list_iou)
            #danh_gia_No_Detect = [f'Image: {result_goc[2]}', f'GT: {result_goc[0]}',f'DT: {mang_one_detect[np.argmax(tam_goc)][0]}', f'IoU: {round(np.max(tam_goc), 2)}']
            danh_gia_No_Detect = [f'GT: {result_goc[0]}', f'DT: {mang_one_detect[np.argmax(tam_goc)][0]}', f'IoU: {round(np.max(tam_goc), 2)}'] #  Tam thoi k show ten anh
            danh_gia_No_Detect_dk = [result_goc[2], result_goc[0], mang_one_detect[np.argmax(tam_goc)][0], round(np.max(tam_goc), 2)]

            if danh_gia_No_Detect_dk[3] == 0:
                danh_gia_No_Detect.append("evaluate : No detect")
                danh_gia_No_Detect[2] = 'DT: NULL'
                danh_gia_goc_all.append(danh_gia_No_Detect)
                giao_iou_check = result_goc[1]
                draw_color(img_iou, poly=convert_mang(giao_iou_check), color=(255, 102, 0, 125))
                draw_text_rectangle(img_iou, (bounding_box(giao_iou_check)[0][0],
                                              (bounding_box(giao_iou_check)[0][1] - 35) if (bounding_box(
                                                  giao_iou_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_check)[0][
                                                                                            1] - 35) + 35),
                                    "No detect", bounding_box(giao_iou_check), (0, 0, 0), (255, 255, 0), 30)
                examination_object.append({
                    "GT": f'{danh_gia_No_Detect_dk[1]}',
                    "DT": 'NULL',
                    "IoU": f'{danh_gia_No_Detect_dk[3]}',
                    "evaluate": 'No detect'
                })

            if (danh_gia_No_Detect_dk[1] != danh_gia_No_Detect_dk[2]) and danh_gia_No_Detect_dk[3] > 0:
                danh_gia_No_Detect.append("evaluate : No detect")
                danh_gia_goc_all.append(danh_gia_No_Detect)
                giao_iou_check = result_goc[1]
                draw_color(img_iou, poly=convert_mang(giao_iou_check), color=(255, 102, 0, 125))
                draw_text_rectangle(img_iou, (bounding_box(giao_iou_check)[0][0],
                                              (bounding_box(giao_iou_check)[0][1] - 35) if (bounding_box(
                                                  giao_iou_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_check)[0][
                                                                                            1] - 35) + 35),
                                    "No detect", bounding_box(giao_iou_check), (0, 0, 0), (255, 255, 0), 30)
                examination_object.append({
                    "GT": f'{danh_gia_No_Detect_dk[1]}',
                    "DT": f'{danh_gia_No_Detect_dk[2]}',
                    "IoU": f'{danh_gia_No_Detect_dk[3]}',
                    "evaluate": 'No detect'
                })

        """
        Sau khi các ảnh trong tập ảnh DT đã dc xác định là đối tượng good và no confidence với ảnh GT rồi thì ta sẽ 
        loại bỏ những nhãn này k dùng chúng để tìm đối tượng phát sinh nữa vì nó đã dc xếp . 
        bây giờ ta sẽ kiểm  tra xe có đối tượng nào phát sinh không bằng cách lấy tất cả các nhãn ở trong ảnh detect còn
        lại  so với ảnh GT nếu mà nhãn đó thỏa mãn 2 TH :
        1, IoU Max dc tính bằng find_iou(DT,GT) == 0
        2, IoU Max dc tính bằng find_iou(DT,GT) > 0 và khác tên đối tượng so với ảnh GT
        """

        ''''# vẽ màu cho các đối tượng thuộc tập ảnh DT'''
        arising_object = []
        for result_detect1 in mang_one_detect:  # duy cac doi tuong trong anh
            draw_color(img_dt, poly=convert_mang(result_detect1[1]), color=(255, 0, 0, 125))
            draw_text_rectangle(img_dt, (bounding_box(result_detect1[1])[0][0],
                                         (bounding_box(result_detect1[1])[0][1] - 35) if (bounding_box(
                                             result_detect1[1])[0][1] - 35) > 0 else (bounding_box(result_detect1[1])[
                                                                                          0][1] - 35) + 35),
                                result_detect1[0], bounding_box(result_detect1[1]), (0, 0, 0), (255, 255, 0), 30)

        '''' # Xóa các nhãn của ảnh DT khi đã dc phân là good và no confidence với ảnh GT'''
        for i in range(len(list(set(check_repeat)))):
            del mang_one_detect[sorted(list(set(check_repeat)))[i] - i]
        # -----------------------------------------------------------------------------------------------------

        for result_detect1 in mang_one_detect:  # duy cac doi tuong trong anh
            iou1 = []
            list_Poly1 = []
            for result_goc1 in mang_one_goc_new:
                iou1.append(find_iou(result_detect1[1], result_goc1[1])[0])
                list_Poly1.append(find_iou(result_detect1[1], result_goc1[1])[1])

            tam_detect = np.array(iou1)
            #danh_gia_detect = [f'Image: {result_detect1[2]}', f'GT: {result_detect1[0]}',f'DT: {mang_one_goc_new[np.argmax(tam_detect)][0]}',f'IoU: {round(np.max(tam_detect), 2)}']
            danh_gia_detect = [f'GT: {result_detect1[0]}', f'DT: { mang_one_goc_new[np.argmax(tam_detect)][0]}', f'IoU: {round(np.max(tam_detect), 2)}'] #  Tam thoi k show ten anh
            danh_gia_detect_dk = [result_detect1[2], result_detect1[0],  mang_one_goc_new[np.argmax(tam_detect)][0], round(np.max(tam_detect), 2)]
            if danh_gia_detect_dk[3] == 0:
                danh_gia_detect.append("evaluate : Detect false")
                danh_gia_detect[2] = 'GT: NULL'
                danh_gia_detect_all.append(danh_gia_detect)
                giao_iou_false_check = result_detect1[1]
                draw_color(img_iou, poly=convert_mang(giao_iou_false_check), color=(0, 255, 255, 125))
                draw_text_rectangle(img_iou, (bounding_box(giao_iou_false_check)[0][0], (bounding_box(giao_iou_false_check)[0][1] - 35) if (bounding_box(giao_iou_false_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_false_check)[0][1] - 35) + 35), "Detect_false", bounding_box(giao_iou_false_check), (0, 0, 0), (255, 255, 0), 30)
                arising_object.append({
                    "GT": 'NULL',
                    "DT": f'{danh_gia_detect_dk[1]}',
                    "IoU": f'{danh_gia_detect_dk[3]}',
                    "evaluate": 'Detect false'
                })

            if danh_gia_detect_dk[3] > 0 and (danh_gia_detect_dk[1] != danh_gia_detect_dk[2]):
                danh_gia_detect.append("evaluate : Detect false")
                danh_gia_detect_all.append(danh_gia_detect)
                giao_iou_false_check = result_detect1[1]
                draw_color(img_iou, poly=convert_mang(giao_iou_false_check), color=(0, 255, 255, 125))
                draw_text_rectangle(img_iou, (bounding_box(giao_iou_false_check)[0][0], (bounding_box(giao_iou_false_check)[0][1] - 35) if (bounding_box(giao_iou_false_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_false_check)[0][1] - 35) + 35), "Detect_false", bounding_box(giao_iou_false_check), (0, 0, 0), (255, 255, 0), 30)
                arising_object.append({
                    "GT": 'NULL',
                    "DT": f'{danh_gia_detect_dk[1]}',
                    "IoU": f'{danh_gia_detect_dk[3]}',
                    "evaluate": 'Detect false'
                })
        '''' # Trường hợp ảnh detect k detect được đối đối tượng nào đồng nghĩa với tất cả đối tượng trong 
        tập ảnh GT đều là no detect hết'''
    else:
        examination_object = []
        for result_goc in mang_one_goc:  # duy cac doi tuong trong anh
            draw_color(img_gt, poly=convert_mang(result_goc[1]), color=(0, 255, 0, 125))
            draw_text_rectangle(img_gt, (bounding_box(result_goc[1])[0][0],(bounding_box(result_goc[1])[0][1] - 35) if
                                        (bounding_box(result_goc[1])[0][1] - 35) > 0 else
                                        (bounding_box(result_goc[1])[0][1] - 35) + 35),result_goc[0], bounding_box(result_goc[1]), (0, 0, 0), (255, 255, 0), 30)
            #danh_gia_goc = [f'Image: {result_goc[2]}', f'GT: {result_goc[0]}', 'DT:NULL', f'IoU: 0.0']
            danh_gia_goc = [f'GT: {result_goc[0]}', 'DT:NULL', f'IoU: 0.0'] #  Tam thoi k show ten anh
            danh_gia_goc_dk = [result_goc[2], result_goc[0], 'NULL', 0.0]
            danh_gia_goc.append("evaluate : No detect")
            danh_gia_goc_all.append(danh_gia_goc)
            giao_iou_check = result_goc[1]
            draw_color(img_iou, poly=convert_mang(giao_iou_check), color=(255, 102, 0, 125))
            draw_text_rectangle(img_iou, (bounding_box(giao_iou_check)[0][0],
                                          (bounding_box(giao_iou_check)[0][1] - 35) if (bounding_box(
                                              giao_iou_check)[0][1] - 35) > 0 else (bounding_box(giao_iou_check)[0][
                                                                                        1] - 35) + 35),
                                "No detect", bounding_box(giao_iou_check), (0, 0, 0), (255, 255, 0), 30)
            examination_object.append({
                "GT": f'{danh_gia_goc_dk[1]}',
                "DT": 'NULL',
                "IoU": f'{danh_gia_goc_dk[3]}',
                "evaluate": 'No detect'
            })
        arising_object = []

    data['shapes'].append({"name": name,
                           "examination_object": examination_object,
                           "arising_object": arising_object})

    ''''# Xuất kết quả ra file txt'''
    file = open(path, 'w', encoding='utf-8')
    file.writelines(f"{50*'-'}>examination object<{50*'-'}\n")
    for goc_all in danh_gia_goc_all:
        file.writelines(f"{goc_all}\n")
    file.writelines("\n\n")
    file.writelines(f"{50*'-'}>arising object<{50*'-'}\n")
    for detect_all in danh_gia_detect_all:
        file.writelines(f"{detect_all}\n")
    ''''# vẽ ảnh GT , DT , IoU , phần ghi chú lên matplotlib'''
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_gt)
    plt.axis('off')
    plt.title("Image_GT")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_dt)
    plt.axis('off')
    plt.title("Image_DT")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(img_iou)
    plt.axis('off')
    plt.title("Image_IoU")

    fig.add_subplot(rows, columns, 4)
    plt.axis([0, 15, 0, 15])

    text = f"{50*'-'}> examination object <{50*'-'}"
    text1 = f"{50*'-'}> arising object <{50*'-'}"

    moc = 0
    plt.text(0.5, 14, text, style='italic', fontsize=7)
    for vt, danh_gia in enumerate(danh_gia_goc_all):
        plt.text(0.5, 14 - vt -1, danh_gia, style='italic', fontsize=7)
        moc = 14 - vt -1

    plt.text(0.5, moc - 2, text1, style='italic', fontsize=7)
    for vt1, danh_gia1 in enumerate(danh_gia_detect_all):
        plt.text(0.5, (moc - 2) - vt1 - 1, danh_gia1, style='italic', fontsize=7)

    plt.title("Note")

    ''''#đường dẫn lưu các ảnh vẽ trực quan'''
    # plt.savefig(f'D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/KQ_IOU/{a}.jpg')
    plt.savefig(f'{path_KQ}/{a}.jpg')
    plt.close(fig)
    # plt.show()
    print('detect oki')
    ''''# đường dẫn lưu file json ghi chú tất cả'''
# with open('D:/Bat_thuong_tren_cot_dien/Test_find_iou_19052022/KQ_IOU/annotations.json', 'w') as file:
with open(f'{path_KQ}/annotations.json', 'w') as file:
    json.dump(data, file, indent=4)



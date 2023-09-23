import json
import matplotlib.pyplot as plt

# --------------------- tham số RUN-------------------
path = 'F:\DJI_chia_du_lieu\KIM_NGUU_ALL\Data_training\Data_training_09-04-2023_Train_Test\Danh_gia\KQ/'
# ------------------------------------------------------------------------------

'''# Đọc file annotations.json tổng kết'''
with open(path + 'annotations.json') as file:
    data = json.load(file)

''''# Lấy tổng số nhãn của đối tượng kiểm tra'''
total_labels_examination_object = []
for idx, result_one in enumerate(data["shapes"]):
    for idx1, obj_one in enumerate(result_one['examination_object']):
        total_labels_examination_object.append(obj_one['GT'])

''''# Loại bỏ những nhãn trùng nhau của đối tượng kiểm tra'''
labels = list(set(total_labels_examination_object))
labels_edit = [i.replace(':', '_') for i in labels]

''''# khai báo các biến đại diện cho đối tượng thuộc đối tượng cần kiểm tra'''
for i in range(len(labels_edit)):
    locals()[labels_edit[i]+'_examination'] = []

''''# khai báo trạng thái good , no confidence , no detect  tương ứng đối với từng biến thuộc đối tượng cần kiểm tra'''
for i in range(len(labels_edit)):
    locals()[labels_edit[i]+'_examination'+'_good'] = []
    locals()[labels_edit[i]+'_examination'+'_no_confidence'] = []
    locals()[labels_edit[i]+'_examination'+'_no_detect'] = []

''''# lọc và lấy ra các labels tương ứng với từng đối tượng cần kiểm tra'''
for i in range(len(labels_edit)):
    for result_one in data["shapes"]:
        for obj_one in result_one['examination_object']:
            if obj_one['GT'] == labels[i]:
                locals()[labels_edit[i]+'_examination'].append(obj_one['GT'])

''''# lọc và lấy ra các label các đối tương ứng với từng trạng thái good , no confindence ,
# no detect của đối tượng cần kiểm tra'''
for i in range(len(labels_edit)):
    for result_one in data["shapes"]:
        for obj_one in result_one['examination_object']:
            if obj_one['GT'] == labels[i] and obj_one['evaluate'] == 'Good':
                locals()[labels_edit[i]+'_examination'+'_good'].append(obj_one)
            if obj_one['GT'] == labels[i] and obj_one['evaluate'] == 'No confidence':
                locals()[labels_edit[i]+'_examination'+'_no_confidence'].append(obj_one)
            if obj_one['GT'] == labels[i] and obj_one['evaluate'] == 'No detect':
                locals()[labels_edit[i]+'_examination'+'_no_detect'].append(obj_one)

''''# khái báo biến chứa tất cả nhãn chứa đối tượng phát sinh '''
arising_object_all = []
for idx, result_one in enumerate(data["shapes"]):
    for idx1, obj_one in enumerate(result_one['arising_object']):
        arising_object_all.append(obj_one['DT'])

''''# lọc các nhãn trùng nhau của đối tượng phát sinh'''
labels_arising = list(set(arising_object_all))
labels_arising_edit = list(set([i.replace(':', '_') for i in labels_arising]))

''''# khai báo các biến tương ứng với đối tượng phát sinh'''
for i in range(len(labels_arising_edit)):
    locals()[labels_arising_edit[i]+'_arising_object'] = []

''''# lấy từng nhãn tương ứng với từng đối tượng phát sinh'''
for i in range(len(labels_arising_edit)):
    for result_one in data["shapes"]:
        for obj_one in result_one['arising_object']:
            if obj_one['DT'] == labels_arising[i] and obj_one['evaluate'] == 'Detect false':
                locals()[labels_arising_edit[i]+'_arising_object'].append(obj_one['DT'])

''''# khai báo biến ghi chú tương ứng với từng đối tượng kiểm tra để hỗ trợ vẽ đồ thị trực quan sau này'''
for i in range(len(labels_edit)):
    locals()['note_'+labels_edit[i]] = []

''''# nạp thông tin cho từng biến ghi chú liên quan đến số lượng tổng label tượng ứng với từng nhãn từng trạng thái good,
# no confindence , no detect'''
for i in range(len(labels_edit)):
    locals()['note_'+labels_edit[i]] = {"good": len(locals()[labels_edit[i]+'_examination'+'_good']),
                                        "No_confidence": len(locals()[labels_edit[i]+'_examination'+'_no_confidence']),
                                        "No_detect": len(locals()[labels_edit[i]+'_examination'+'_no_detect'])}

''''# khai báo ghi chú của từng đối tượng phát sinh'''
note_arising = []
for i in range(len(labels_arising_edit)):
    note_arising.append({f"{labels_arising[i]}": len(locals()[labels_arising_edit[i]+'_arising_object'])})

note_arising_edit = {}
for i in note_arising :
    note_arising_edit.update(i)

''''# khai báo màu cần để vẽ biểu đồ , khai báo bao nhiêu cũng dc . lúc vẽ màu sẽ dc lấy đến vị trí 0 -> đến 
# vt = tổng sl  cột cần vẽ'''
color = ['green', 'yellow', 'red', 'blue', 'orange', 'violet', 'pink', 'grey', 'brown']


def lam_tron(num):
    return float(("%.3f" % num)[:-1])


plt.figure(figsize=(15, 10))
for i in range(len(labels)):
    ax = plt.subplot(4, 2, i + 1)
    ax.bar(locals()['note_'+labels_edit[i]].keys(), locals()['note_'+labels_edit[i]].values(), color=color)
    ax.set(title=f"{labels[i]}", ylabel="Total labels")
    ax.text(0, len(locals()[labels_edit[i]+'_examination'+'_good']), str(lam_tron(lam_tron(len(locals()[labels_edit[i] + '_examination' + '_good'])/len(locals()[labels_edit[i]+'_examination']))*100)) + '%',
            va='center', ha='center')
    ax.text(1, len(locals()[labels_edit[i]+'_examination'+'_no_confidence']), str(lam_tron(lam_tron(len(locals()[labels_edit[i] +'_examination'+ '_no_confidence'])/len(locals()[labels_edit[i]+'_examination']))*100)) + '%',
            va='center', ha='center')
    ax.text(2, len(locals()[labels_edit[i]+'_examination'+'_no_detect']), str(lam_tron(lam_tron(len(locals()[labels_edit[i] +'_examination'+ '_no_detect'])/len(locals()[labels_edit[i]+'_examination']))*100)) + '%',
            va='center', ha='center')
plt.savefig('results/a.jpg')


plt.figure(figsize=(15, 10))
plt.pie(note_arising_edit.values(),
        labels=note_arising_edit.keys(),
        colors=['red', 'blue', 'green'],
        autopct='%1.1f%%',
        shadow=False)
plt.title("Arising Object", fontsize=18)
plt.savefig('results/b.jpg')


plt.figure(figsize=(15, 5))
plt.bar(note_arising_edit.keys(), note_arising_edit.values(), color=color)
plt.ylabel('Total labels', fontsize=16)
plt.title("Arising Object", fontsize=18)
for idx, result in enumerate(labels_arising):
    plt.text(idx, len(locals()[labels_arising_edit[idx]+'_arising_object']), str(len(locals()[labels_arising_edit[idx]+'_arising_object'])) + ' --object',
             va='center', ha='center')
plt.savefig('results/c.jpg')
plt.show()


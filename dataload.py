import numpy as np
import h5py
from matplotlib import image
import matplotlib.pyplot as plt
import glob


def load_visual(name):
    f = h5py.File(name+'.mat', 'r')
    key = 'visual'
    data = f.get('/'+name[5:-5])[key]
    data = np.array(data)
    num_data = data.shape[1]
    data_list = []
    for i in range(num_data):
        tmp = f[data[0, i]]
        string = ''.join([chr(j[0]) for j in f[tmp[0,0]][:]])
        data_list.append(string)

    return data_list


def load_best_beam_mhot(name):
    f = h5py.File(name+'.mat', 'r')
    key = 'best_beam_mhot'
    data = f.get('/'+name[5:-5])[key]
    data = np.array(data)
    num_data = data.shape[1]
    data_list = []
    for i in range(num_data):
        tmp = np.array(f[data[0, i]])
        data_list.append(tmp.astype(np.float32))

    return data_list


def load_visual_and_beam(name, name_visual=None):
    visual = load_visual(name_visual)
    beam = load_best_beam_mhot(name)

    data = zip(visual, beam)
    return list(data)


def load_bbox(path):
    txt_files = glob.glob(path + '/*.txt')
    bboxes = []
    img_names = []
    for f in txt_files:
        if 'class' in f:
            continue
        bbox = np.loadtxt(f, dtype=np.float32, delimiter=' ')
        if len(bbox.shape) < 2:
            bbox = bbox.reshape((-1, 5))
        bboxes.append(bbox)
        img_names.append(f.split('\\')[-1][:-4])
    return list(zip(img_names, bboxes))


def preprocess_data(bboxes, beams):
    max_len = 0

    for bbox in bboxes:
        bbox[:, 0] += 1
        max_len = max(max_len, bbox.shape[0])

    bboxes = [np.concatenate((bbox, np.zeros((max_len - bbox.shape[0], 5), dtype=np.float32)), 0) for bbox in bboxes]

    bboxes = np.stack(bboxes, 0)
    beams = np.concatenate(beams, 0)

    return bboxes, beams


def load_data(name='cam5'):
    bbox_path = 'data/' + name + '/bbox_labels_out'
    name = 'data/dataset_small_decouple_upsample1_' + name

    data_bbox = load_bbox(bbox_path)
    data_beam = load_visual_and_beam(name=name, name_visual=name)

    data_bbox = sorted(data_bbox, key=lambda x: x[0])
    data_beam = sorted(data_beam, key=lambda x: x[0])

    data_scene = [d[0] for d in data_bbox]
    data_bbox = [d[1] for d in data_bbox]
    data_beam = [d[1] for d in data_beam]

    data_bbox, data_beam = preprocess_data(data_bbox, data_beam)

    return {'x': data_bbox, 'y': data_beam, 'z': data_scene}



if __name__ == "__main__":
    data = load_data('cam5')
    print('done')
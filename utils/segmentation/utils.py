import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from utils.metrics import Evaluator


def forward_all(net_inference, dataloader, visualize=False, opt=None):
    evaluator = Evaluator(21)
    evaluator.reset()
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample['image'].cuda(), sample['label'].cuda()

            activations = net_inference(image)

            image = image.cpu().numpy()
            label = label.cpu().numpy().astype(np.uint8)

            logits = activations[list(activations.keys())[-1]] if type(activations) != torch.Tensor else activations
            pred = torch.max(logits, 1)[1].cpu().numpy().astype(np.uint8)
            
            evaluator.add_batch(label, pred)

            # print(label.shape, pred.shape)
            if visualize:
                for jj in range(sample["image"].size()[0]):
                    segmap_label = decode_segmap(label[jj], dataset='pascal')
                    segmap_pred = decode_segmap(pred[jj], dataset='pascal')

                    img_tmp = np.transpose(image[jj], axes=[1, 2, 0])
                    img_tmp *= (0.229, 0.224, 0.225)
                    img_tmp += (0.485, 0.456, 0.406)
                    img_tmp *= 255.0
                    img_tmp = img_tmp.astype(np.uint8)

                    cv2.imshow('image', img_tmp[:, :, [2,1,0]])
                    cv2.imshow('gt', segmap_label)
                    cv2.imshow('pred', segmap_pred)
                    cv2.waitKey(0)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print("Acc: {}".format(Acc))
    print("Acc_class: {}".format(Acc_class))
    print("mIoU: {}".format(mIoU))
    print("FWIoU: {}".format(FWIoU))
    if opt is not None:
        with open("seg_result.txt", 'a+') as ww:
            ww.write("{}, quant: {}, relu: {}, equalize: {}, absorption: {}, correction: {}, clip: {}, distill_range: {}\n".format(
                opt.dataset, opt.quantize, opt.relu, opt.equalize, opt.absorption, opt.correction, opt.clip_weight, opt.distill_range
            ))
            ww.write("Acc: {}, Acc_class: {}, mIoU: {}, FWIoU: {}\n\n".format(Acc, Acc_class, mIoU, FWIoU))


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])
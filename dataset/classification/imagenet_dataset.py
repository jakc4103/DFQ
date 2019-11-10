import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

class ImageNetDataset(Dataset):
    """
    ImageNet validation dataset
    """

    def __init__(self,
                 args,
                 base_dir='D:/workspace/dataset/ILSVRC',
                 split='val',
                 ):
        """
        :param base_dir: path to ImageNet dataset directory
        :param split: train/val/test
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'Data/CLS-LOC')
        # self._cat_dir = os.path.join(self._base_dir, label)

        self.split = split

        self.args = args

        file_label = os.path.join(self._base_dir, 'val.txt')

        # self.im_ids = []
        self.images = []
        self.categories = []


        with open(file_label, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            tok = line.split(' ')
            _image = os.path.join(self._image_dir, self.split, tok[0])
            # print(_image)
            assert os.path.isfile(_image)
            # self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(int(tok[1]))

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        
        if self.split == "train":
            _img = self.transform_tr(_img)
        else:
            _img = self.transform_val(_img)

        return {'image': _img, 'label': _target}


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = self.categories[index]

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.args.crop_size, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                 ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            transforms.Resize(int(self.args.crop_size/0.875)),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                 ])

        return composed_transforms(sample)

    def __str__(self):
        return 'ImageNet(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 224
    args.crop_size = 224

    imagenet_dataset = ImageNetDataset(args, split='val')

    dataloader = DataLoader(imagenet_dataset, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            print("category: {}".format(sample['label'][jj]))

            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)

            plt.show(block=True)



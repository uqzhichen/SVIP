import copy, torch
import random
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np


class ZSLDataset(data.Dataset):

    def __init__(self, args, infos, mode, feature=True, GBU=False, dataset=None, image_size=224):
        super(ZSLDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.allclasses = copy.deepcopy(infos['allclasses'])
        self.GBU = GBU
        self.dataset = dataset
        self.image_size = image_size
        if mode == 'train':
            images, labels = infos['trainval_files'], infos['trainval_label']
            if dataset == "CUB":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/CUB_CROP", "data/CUB_200_2011/CUB_200_2011")
                    images_new.append(image_path)
                images = images_new
            if dataset == "AWA2":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new
            if dataset == "SUN":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new

        elif mode == 'test-seen':
            images, labels = infos['test_seen_files'], infos['test_seen_label']
            if dataset == "CUB":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/CUB_CROP", "data/CUB_200_2011/CUB_200_2011")
                    images_new.append(image_path)
                images = images_new
            if dataset == "AWA2":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new
            if dataset == "SUN":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new

        elif mode == 'test-unseen':
            images, labels = infos['test_unseen_files'], infos['test_unseen_label']
            if dataset == "CUB":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/CUB_CROP", "data/CUB_200_2011/CUB_200_2011")
                    images_new.append(image_path)
                images = images_new
            if dataset == "AWA2":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new
            if dataset == "SUN":
                images_new = []
                for image_path in images:
                    image_path = image_path.replace("/home/uqzche20/Datasets/", "data/")
                    images_new.append(image_path)
                images = images_new

        else: 'invalid mode = {:}'.format(mode)
        if 'image2feat' in infos.keys():
              self.image2feat = infos['image2feat']
        else:
            self.image2feat = None

        self.images          = images
        self.labels          = copy.deepcopy(labels)
        self.current_classes = sorted(list(set(self.labels)))
        self.num_classes     = len(self.current_classes)
        self.attributes      = infos['attributes'].clone().float()
        self.oriCLS2newCLS   = dict()
        for i, cls in enumerate(self.current_classes):
          self.oriCLS2newCLS[cls] = i
        self.return_label_mode = 'original'
        self.feature = feature

        self.train_transform = transforms.Compose(
          [transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)), \
           transforms.RandomHorizontalFlip(p=0.5)])
        self.awa_transform = transforms.Compose(
          [transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)), \
           transforms.RandomHorizontalFlip(p=0.5)])

        self.cub_transform = transforms.Compose([
            transforms.Resize(int(self.image_size*8./7.)),
            transforms.CenterCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
        ])
        self.cub_transform1 = transforms.Compose([
            transforms.Resize(int(self.image_size*8./7.)),
            transforms.CenterCrop(self.image_size),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
        ])
        self.awa_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
        ])
        self.cub_valid_transform = transforms.Compose(
          [transforms.Resize(int(self.image_size*8./7.)), transforms.CenterCrop(self.image_size)])
        self.valid_transform = transforms.Compose(
          [transforms.Resize(int(self.image_size*8./7.)), transforms.CenterCrop(self.image_size)])
        self.awa_valid_transform = transforms.Compose(
          [transforms.Resize((self.image_size, self.image_size)), ])
        self.totensor = transforms.Compose(
                  [transforms.ToTensor(), \
                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    def set_return_label_mode(self, mode):
        assert mode in ['original', 'new', 'combine']
        self.return_label_mode = mode

    def set_return_img_mode(self, mode):
        assert mode in ['original', 'rotate', 'original_augment']
        self.return_img_mode = mode

    def get_image(self, image_path):
        image = pil_loader(image_path)
        if self.dataset == 'AWA2':
            if self.mode == 'train':
                return_img = self.awa_valid_transform(image)
            else:
                return_img = self.valid_transform(image)
        elif self.dataset == 'CUB':
            if self.mode == 'train':
                return_img = self.cub_transform(image)
            else:
                return_img = self.cub_valid_transform(image)
        elif self.dataset == 'APY':
            if self.mode == 'train':
                return_img = self.train_transform(image)
            else:
                return_img = self.valid_transform(image)
        else:
            return_img = self.valid_transform(image)

        return_img = self.totensor(return_img)
        return return_img

    def get_image_index(self, index):
        assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
        ori_label = self.labels[index]
        if self.return_label_mode == 'original':
          return_label = ori_label
        elif self.return_label_mode == 'new':
          return_label = self.oriCLS2newCLS[ori_label]
        elif self.return_label_mode == 'combine':
          return_label = (self.oriCLS2newCLS[ ori_label ], ori_label)
        else:
          raise ValueError('invalid mode = {:}'.format(self.return_label_mode))

        image_path = self.images[index]

        return image_path


    def __getitem__(self, index):
        assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
        ori_label = self.labels[index]

        image = pil_loader(self.images[index])
        if self.dataset == 'AWA2':
            if self.mode == 'train':
                return_img = self.awa_valid_transform(image)
            else:
                return_img = self.valid_transform(image)
        elif self.dataset == 'CUB':
            if self.mode == 'train':
                return_img = self.cub_transform(image)
            else:
                return_img = self.cub_valid_transform(image)
        elif self.dataset == 'APY':
            if self.mode == 'train':
                return_img = self.train_transform(image)
            else:
                return_img = self.valid_transform(image)
        elif self.dataset == 'SUN':
            if self.mode == 'train':
                return_img = self.train_transform(image)
            else:
                return_img = self.valid_transform(image)
        else:
            return_img = self.valid_transform(image)
        return_img = self.totensor(return_img)

        if self.return_label_mode == 'original':
            return_label = ori_label
        elif self.return_label_mode == 'new':
            return_label = self.oriCLS2newCLS[ori_label]
        elif self.return_label_mode == 'combine':
            return_label = (self.oriCLS2newCLS[ ori_label ], ori_label)
        else:
            raise ValueError('invalid mode = {:}'.format(self.return_label_mode))

        if self.mode == "train":
            return return_img.clone(), return_label, index
        else:
            return return_img.clone(), return_label

    def __repr__(self):
        return '{name}({length:5d} samples with {num_classes} classes [{mode:}])'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode)

    def __len__(self):
        return len(self.labels)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DualMetaSampler(object):

    def __init__(self, dataset, iters, class_per_it, num_shot):
        super(DualMetaSampler, self).__init__()
        self.n_batch = iters
        self.n_cls = class_per_it
        self.n_per = num_shot

        self.cat =  list(np.unique(dataset.labels))
        self.catlocs = {}
        self.ep_per_batch = 1
        for c in self.cat:
            self.catlocs[c] = np.argwhere(dataset.labels == c).reshape(-1)

    def __iter__(self):
    # yield a batch of indexes
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

    def __len__(self):
        return self.n_batch

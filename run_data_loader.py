import os
import glob
import numpy as np
import torch
from PIL import Image
from data_parser import JpegDataset
from torchvision.transforms import *


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def default_loader(path):
    return Image.open(path).convert('RGB')


class RunTimeDataSet(torch.utils.data.Dataset):

    dataset_object = JpegDataset()

    # Arrya for imgs
    IMGS_Array = []

    def __init__(self, root, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)
        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        target_idx = self.classes_dict[item.label]

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return (data, target_idx)

    def __len__(self):
        return 5

    # according to the image name
    def get_frame_names(self):

        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        print("num_frames_necessary:"+str(num_frames_necessary))
        print("num_frames:"+str(num_frames))
        print([frame_names[-1]])
        print([frame_names[-1]]*3)
        print()
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
            # Temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]
        print(frame_names)
        return frame_names


if __name__ == '__main__':
    transform = Compose([
                        CenterCrop(84),
                        ToTensor(),
                        # Normalize(
                        #     mean=[0.485, 0.456, 0.406],
                        #     std=[0.229, 0.224, 0.225])
                        ])
    loader = VideoFolder(root="/home/wenjin/Documents/pycharmworkspace/20bn-jester-v1",
                         csv_file_input="./20bn-jester-v1/annotations/jester-v1-train.csv",
                         csv_file_labels="./20bn-jester-v1/annotations/jester-v1-labels2.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=False,
                         transform=transform,
                         loader=default_loader)

    data_item, target_idx = loader[0]
    # save_images_for_debug("input_images", data_item.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=5, pin_memory=True)


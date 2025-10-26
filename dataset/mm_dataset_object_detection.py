import os
import json
import numpy as np
import math
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from config.config_mm import Config, parse_args, class_dict
from decord import VideoReader
from decord import cpu
import concurrent.futures
DEBUG_MODE = False


class MMDataset(Dataset):
    def __init__(self, data_path, mode, modal, fps, num_frames, len_feature, sampling, seed=-1, supervision='weak'):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.deterministic = True
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.benchmark = False

        self.mode = mode
        self.fps = fps
        self.num_frames = num_frames
        self.len_feature = len_feature

        self.local_data_path = os.path.join(data_path, "train")
        # self.local_data_path = os.path.join(data_path, self.mode)

        # For video processing
        self.transform = self.get_transform(mode, 640)

        split_path = os.path.join(data_path, '{}_fold_1.txt'.format(self.mode))
        # split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        vid_name, human_list_ = self.get_data(index)
        return vid_name, human_list_
    

    def process_video(self, video_path, desired_fps):
        """Load video, adjust to desired FPS, resize frames, and return as numpy array."""
        vr = VideoReader(video_path, ctx=cpu(0))
        original_fps = vr.get_avg_fps()
        frame_interval = round(original_fps / desired_fps)
        usable_frame_count = math.ceil(len(vr) / frame_interval)

        frame_indices = range(0, len(vr), frame_interval)
        frames = vr.get_batch(frame_indices).asnumpy()
        transformed_frames = [self.transform(frame) for frame in frames]

        if DEBUG_MODE:
            print(f"Total frames: {len(vr)}")
            print(f"Original FPS: {original_fps}")
            print(f"Frame interval: {frame_interval}")
            print(f"Usable frame count: {usable_frame_count}")

        return torch.stack(transformed_frames), usable_frame_count


    def get_data(self, index):
        vid_name = self.vid_list[index]
        vid_num_frame = 0
        # Get all filename have vid_name in self.feature_path
        vid_name_all = sorted([f for f in os.listdir(self.local_data_path) if vid_name == '_'.join(f.split('.')[0].split('_')[-2:])])

        # For video processing
        video_paths = [os.path.join(self.local_data_path, vid_n) for vid_n in vid_name_all]
        # Using ThreadPoolExecutor to process videos
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
            args = ((path, self.fps) for path in video_paths)
            future_to_video = {executor.submit(self.process_video, *arg): arg[0] for arg in args}
            results = []
            frame_counts = []
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_data, frame_count = future.result()
                    results.append(video_data)
                    frame_counts.append(frame_count)
                    if DEBUG_MODE:
                        print(f"Processed and resized {video_path}")
                except Exception as exc:
                    print(f'{video_path} generated an exception: {exc}')

        results = [result for result in results]
        combined_video_data = torch.stack(results)

        return vid_name, combined_video_data
    
    
    def get_transform(self, mode, input_size):
        if mode == "train":
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                    ])

        else:
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                    ])

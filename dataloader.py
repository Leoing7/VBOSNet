import csv
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms


class ContrastiveAugmentor:

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, img):
        augment_method = random.choice([self.spatial_augmentation, self.temporal_augmentation, self.color_augmentation])
        return augment_method(img)

    def spatial_augmentation(self, img):
        img_array = np.array(img)
        for i in range(img_array.shape[0]):
            np.random.shuffle(img_array[i])
        return Image.fromarray(img_array).resize(self.img_size)

    def temporal_augmentation(self, img):
        img_array = np.array(img)
        h, w, c = img_array.shape
        window_height = int(h * random.uniform(0.5, 0.9))
        start_idx = random.randint(0, h - window_height)
        window = img_array[start_idx:start_idx + window_height, :, :]
        return Image.fromarray(window).resize((w, h))

    def color_augmentation(self, img):
        color_factors = {
            'brightness': random.uniform(0.7, 1.3),
            'contrast': random.uniform(0.7, 1.3),
            'saturation': random.uniform(0.7, 1.3),
            'sharpness': random.uniform(0.7, 1.3),
        }
        img = ImageEnhance.Brightness(img).enhance(color_factors['brightness'])
        img = ImageEnhance.Contrast(img).enhance(color_factors['contrast'])
        img = ImageEnhance.Color(img).enhance(color_factors['saturation'])
        img = ImageEnhance.Sharpness(img).enhance(color_factors['sharpness'])
        if random.random() > 0.5:
            radius = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        return img.resize(self.img_size)


def Img2SingleWavelengthImg(image_rgb, Wave_RGB):
    target_color = np.array([int(Wave_RGB[:2], 16), int(Wave_RGB[2:4], 16), int(Wave_RGB[4:6], 16)])
    distances = np.sqrt(((image_rgb - target_color) ** 2).sum(axis=2))
    max_distance = np.sqrt(255 ** 2 * 3)
    similarity = 1 - distances / max_distance
    SWI = similarity
    return SWI


class MWLIPreprocessing:

    def __init__(self):
        self.MWLI_dict = {'MWLI_3': ['630', '456', '486']}
        self.wave_dict = {'456': '002FFF', '486': '00EFFF', '630': 'FF4F00'}

    def MWLIPreprocessing(self, img):
        wavelengths = self.MWLI_dict['MWLI_3']
        SWLI_1 = Img2SingleWavelengthImg(img, self.wave_dict[wavelengths[0]])
        SWLI_2 = Img2SingleWavelengthImg(img, self.wave_dict[wavelengths[1]])
        SWLI_3 = Img2SingleWavelengthImg(img, self.wave_dict[wavelengths[2]])
        MWLI_mat = np.zeros((SWLI_1.shape[0], SWLI_1.shape[1], 3))
        MWLI_mat[:, :, 0] = SWLI_1
        MWLI_mat[:, :, 1] = SWLI_2
        MWLI_mat[:, :, 2] = SWLI_3
        MWLI = MWLI_mat * 255
        image = MWLI.astype(np.uint8)
        image = Image.fromarray(image, 'RGB')
        return image

    def __call__(self, img):
        return self.MWLIPreprocessing(img)


class SpO2Dataset(Dataset):

    def __init__(self, root_dir, img_paths, transform=None, name='pure', cache_in_ram=False, metadata=None):
        self.root_dir = root_dir
        self.img_paths = img_paths
        self.transform = transform
        self.name = name
        self.metadata = metadata
        self.cache = {}
        if cache_in_ram:
            self.cache = {p: self._read_image(p) for p in self.img_paths}
        self.augmentor = ContrastiveAugmentor(img_size=(224, 224))
        self.MWLI = MWLIPreprocessing()
        self.ip_dict, self.label_dict = self.__know__()

    def __len__(self):
        return len(self.img_paths)

    def __IpandLabel__(self, image_path):
        if self.metadata is not None:
            subject, spo2 = self.metadata[image_path]
        elif self.name == 'pure':
            parts = Path(image_path).stem.split('_')
            subject, spo2 = parts[0], float(parts[3])
        else:
            raise ValueError('Non-PURE datasets require a path/subject/spo2 manifest.')
        return str(subject), (float(spo2) - 80.0) / 20.0

    def _read_image(self, image_path):
        if self.metadata is not None:
            path = Path(self.root_dir) / image_path
        else:
            subject, _ = self.__IpandLabel__(image_path)
            path = Path(self.root_dir) / subject / image_path
        with Image.open(path) as image:
            return image.convert('RGB')

    def _load_image(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path].copy()
        return self._read_image(image_path)

    def __know__(self):
        ip_dict = defaultdict(lambda: defaultdict(list))
        label_dict = defaultdict(lambda: defaultdict(list))
        for img_path in self.img_paths:
            ip, label = self.__IpandLabel__(img_path)
            label = str(label)
            ip_dict[ip][label].append(img_path)
            label_dict[label][ip].append(img_path)
        return {s: dict(groups) for s, groups in ip_dict.items()}, {l: dict(groups) for l, groups in label_dict.items()}

    def __getImage__(self, image_path):
        ip, label = self.__IpandLabel__(image_path)
        image = self._load_image(image_path)
        return image

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        ip, label = self.__IpandLabel__(image_path)
        label = str(label)
        image = self._load_image(image_path)


        available_labels = [l for l in self.ip_dict[ip].keys() if l != label]


        if not available_labels:
            available_labels = [label]
        chosen_neg_label = random.choice(available_labels)
        available_neg_images = [img for img in self.ip_dict[ip][chosen_neg_label] if img != image_path]
        if not available_neg_images:
            available_neg_images = [image_path]
        image_negative_name = random.choice(available_neg_images)
        image_negative = self.__getImage__(image_negative_name)


        available_ips = [i for i in self.label_dict[label].keys() if i != ip]
        if not available_ips:
            available_ips = [ip]
        chosen_pos_ip = random.choice(available_ips)
        available_pos_images = [img for img in self.label_dict[label][chosen_pos_ip] if img != image_path]
        if not available_pos_images:
            available_pos_images = [image_path]
        image_positive_name = random.choice(available_pos_images)
        image_positive = self.__getImage__(image_positive_name)


        image = image.resize((224, 224))
        image_1 = self.augmentor(image)
        image_positive = image_positive.resize((224, 224))
        image_negative = image_negative.resize((224, 224))
        image_mwli = self.MWLI(image)
        return (
            transforms.ToTensor()(image),
            transforms.ToTensor()(image_1),
            transforms.ToTensor()(image_positive),
            transforms.ToTensor()(image_negative),
            transforms.ToTensor()(image_mwli),
            torch.tensor(float(label)),
        )


class InferenceDataset(SpO2Dataset):
    def __getitem__(self, index):
        name = self.img_paths[index]
        _, normalized_label = self.__IpandLabel__(name)
        image = self._load_image(name).resize((224, 224))
        tensor = transforms.ToTensor()
        return tensor(image), tensor(self.MWLI(image)), normalized_label * 20.0 + 80.0, name


def list_samples(root, manifest=None):
    if manifest is not None:
        with open(manifest, encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows or not {'path', 'subject', 'spo2'}.issubset(rows[0]):
            raise ValueError('Manifest columns must be path,subject,spo2.')
        metadata = {r['path']: (r['subject'], float(r['spo2'])) for r in rows}
        if len(metadata) != len(rows):
            raise ValueError('Duplicate path in manifest.')
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in metadata):
            raise ValueError('Manifest paths must be relative to the data root.')
        return sorted(metadata), metadata
    paths = sorted(p.name for subject in Path(root).iterdir() if subject.is_dir()
                   for p in subject.glob('*.png'))
    if len(set(paths)) != len(paths):
        raise ValueError('PURE filenames must be unique.')
    for name in paths:
        parts = Path(name).stem.split('_')
        if len(parts) != 4 or not (Path(root)/parts[0]/name).is_file():
            raise ValueError(f'Unexpected PURE filename: {name}')
        float(parts[3])
    if not paths:
        raise ValueError('No samples found.')
    return paths, None


def subject_of(name, metadata=None):
    return metadata[name][0] if metadata is not None else Path(name).stem.split('_')[0]

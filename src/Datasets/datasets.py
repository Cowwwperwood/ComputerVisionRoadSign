import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor
import albumentations as A
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from albumentations import Compose, Normalize, Resize, RandomCrop, HorizontalFlip


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        with open(path_to_classes_json, 'r') as file:
            self.classes_info = json.load(file)
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        self.samples = []
        for folder in root_folders:
            for class_name in os.listdir(folder):
                class_folder = os.path.join(folder, class_name)
                if os.path.isdir(class_folder):
                    class_idx = self.class_to_idx.get(class_name, -1)
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, class_idx))

        self.classes_to_samples = {idx: [] for idx in self.class_to_idx.values()}
        for idx, (_, class_idx) in enumerate(self.samples):
            self.classes_to_samples[class_idx].append(idx)

        self.transform = Compose([
            Resize(256, 256),  
            RandomCrop(224, 224, p=0.5), 
            HorizontalFlip(p=0.5), 
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            Resize(224, 224),  
            ToTensorV2(), 
        ])



    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path, class_idx = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        transformed = self.transform(image=np.array(image))
        image_tensor = transformed["image"]

        return image_tensor, img_path, class_idx

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, 'r') as file:
            class_data = json.load(file)

        class_to_idx = {class_name: idx for idx, (class_name, _) in enumerate(class_data.items())}

        classes = [class_name for class_name in class_data.keys()]

        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)

    def get_rare_images(self) -> typing.List[str]:
        """
        Функция для нахождения путей к изображениями с типом "rare" и подсчета их количества.
        Возвращает список путей и количество изображений.
        """
        rare_images = []
        for img_path, class_idx in self.samples:
            class_name = self.classes[class_idx]

            if class_name in self.classes_info and self.classes_info[class_name].get("type") == "rare":
                rare_images.append(img_path)

        print(f"Количество изображений с типом 'rare': {len(rare_images)}",total)
        return rare_images




class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root

        with open(path_to_classes_json, 'r') as f:
            self.classes_info = json.load(f)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_info.keys())}

        self.samples = []
        for img_name in os.listdir(self.root):
            img_path = os.path.join(self.root, img_name)
            if img_name.endswith(".png"): 
                self.samples.append(img_path)

        self.targets = None
        if annotations_file is not None:
            annotations = pd.read_csv(annotations_file)
            self.targets = {row["filename"]: self.class_to_idx[row["class"]] for _, row in annotations.iterrows()}

        self.transform = Compose([
            Resize(256, 256),  
            #RandomCrop(224, 224, p=0.5),  
            #HorizontalFlip(p=0.5),  
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            Resize(224, 224),  
            ToTensorV2(),  
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int, str]:
        """
        Возвращает кортеж: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1"),
        аннотация (имя класса или "unknown" при отсутствии разметки).
        """
        img_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image=np.array(image))["image"]

        if self.targets is not None:
            target = self.targets.get(os.path.basename(img_path), -1)
            annotation = (
                list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(target)]
                if target != -1
                else "unknown"
            )
        else:
            target = -1
            annotation = "unknown"

        return image, img_path, target, annotation


    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class TestDataRare(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета, включающего только редкие (rare) классы.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root

        with open(path_to_classes_json, 'r') as f:
            self.classes_info = json.load(f)

        self.classes_info = {cls: info for cls, info in self.classes_info.items() if info["type"] == "freq" }

        self.class_to_idx = {cls: info["id"] for cls, info in self.classes_info.items()}

        self.samples = []
        for img_name in os.listdir(self.root):
            img_path = os.path.join(self.root, img_name)
            if img_name.endswith(".png"):  
                self.samples.append(img_path)

        self.targets = None
        if annotations_file is not None:
            annotations = pd.read_csv(annotations_file)
            self.targets = {
                row["filename"]: self.class_to_idx[row["class"]]
                for _, row in annotations.iterrows()
                if row["class"] in self.class_to_idx
            }

        self.transform = Compose([
            Resize(256, 256),
            #RandomCrop(224, 224, p=0.5),  
            #HorizontalFlip(p=0.5), 
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
            Resize(224, 224),  
            ToTensorV2(), 
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int, str]:
        """
        Возвращает кортеж: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1"),
        аннотация (имя класса или "unknown" при отсутствии разметки).
        """
        img_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image=np.array(image))["image"]

        if self.targets is not None:
            target = self.targets.get(os.path.basename(img_path), -1)
            annotation = (
                list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(target)]
                if target != -1
                else "unknown"
            )
        else:
            target = -1
            annotation = "unknown"

        return image, img_path, target, annotation

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)




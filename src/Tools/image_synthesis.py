import os
import random
import re
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor

class SignGenerator:
    def __init__(self, icons_dir, backgrounds_dir, output_dir):
        self.icons_dir = icons_dir
        self.backgrounds_dir = backgrounds_dir
        self.output_dir = output_dir
        self.background_images = list(Path(backgrounds_dir).glob('*.jpg')) 
        print(f"Найдено фонов: {len(self.background_images)}")
        self.icon_files = list(Path(icons_dir).glob('*.png'))  
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def resize_icon(self, icon, min_size=16, max_size=128):
        size = random.randint(min_size, max_size)
        return icon.resize((size, size), Image.Resampling.LANCZOS)

    def add_padding(self, icon, max_padding_pct=0.15):
        icon_size = icon.size[0]
        padding = random.randint(0, int(icon_size * max_padding_pct))
        new_size = (icon_size + padding, icon_size + padding)
        new_icon = Image.new('RGBA', new_size, (255, 255, 255, 0))  
        new_icon.paste(icon, (padding // 2, padding // 2))  
        return new_icon

    def rotate_icon(self, icon, min_angle=-15, max_angle=15):
        angle = random.randint(min_angle, max_angle)
        return icon.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    def blur_icon(self, icon, min_angle=-90, max_angle=90):
        angle = random.randint(min_angle, max_angle)
        kernel = np.ones((3, 3)) / 9.0 
        icon_array = np.array(icon.convert("RGBA"))
        icon_blurred = np.zeros_like(icon_array)
        for channel in range(4): 
            icon_blurred[..., channel] = convolve(icon_array[..., channel], kernel, mode='reflect')
        return Image.fromarray(icon_blurred.astype(np.uint8))

    def apply_gaussian_blur(self, icon, radius=2):
        return icon.filter(ImageFilter.GaussianBlur(radius))

    def apply_random_dullness(self, icon):
        enhancer = ImageEnhance.Brightness(icon)
        dullness_factor = random.uniform(0.5, 0.8) 
        return enhancer.enhance(dullness_factor)

    def embed_on_background(self, icon, background):
        icon_size = icon.size[0]
        bg_resized = background.resize((icon_size + random.randint(0, int(icon_size * 0.15)),
                                        icon_size + random.randint(0, int(icon_size * 0.15))))
        bg_resized.paste(icon, (random.randint(0, bg_resized.width - icon_size),
                                random.randint(0, bg_resized.height - icon_size)), icon)
        return bg_resized

    def generate_one_icon(self, icon_class):
        icon_files = [file for file in self.icon_files if re.search(f'{icon_class}', file.stem)]
        if not icon_files:
            raise ValueError(f"Нет иконок для класса {icon_class} в директории {self.icons_dir}")

        icon_path = random.choice(icon_files)
        icon = Image.open(icon_path).convert("RGBA")
        icon_resized = self.resize_icon(icon)
        icon_with_padding = self.add_padding(icon_resized)
        icon_rotated = self.rotate_icon(icon_with_padding)
        icon_blurred = self.blur_icon(icon_rotated)
        icon_gaussian_blurred = self.apply_gaussian_blur(icon_blurred)
        icon_dulled = self.apply_random_dullness(icon_gaussian_blurred)  

        if not self.background_images:
            raise ValueError("Нет доступных фонов для встраивания. Проверьте папку с фонами.")

        background = random.choice(self.background_images)
        background_image = Image.open(background).convert("RGBA")
        final_image = self.embed_on_background(icon_dulled, background_image)
        return final_image


    def generate_samples_from_image(self, image_path, n=10):
        icon = Image.open(image_path).convert("RGBA")


        for i in range(n):
            if not self.background_images:
                raise ValueError("Нет доступных фонов в указанной директории.")
            icon_resized = self.resize_icon(icon)
            icon_with_padding = self.add_padding(icon_resized)
            icon_rotated = self.rotate_icon(icon_with_padding)
            icon_blurred = self.blur_icon(icon_rotated)
            icon_gaussian_blurred = self.apply_gaussian_blur(icon_blurred)
            icon_dulled = self.apply_random_dullness(icon_gaussian_blurred)  

            background = random.choice(self.background_images)
            background_image = Image.open(background).convert("RGBA")

            final_image = self.embed_on_background(icon_dulled, background_image)

            output_image_path = Path(self.output_dir) / f"sample_{i}.png"
            final_image.save(output_image_path)

        print(f"Генерация {n} сэмплов завершена, сохранено в {self.output_dir}.")

    def generate_all_data(self, num_images_per_class=1000):
        icon_classes = set()
        for file in self.icon_files:
            class_name = file.stem
            icon_classes.add(class_name)

        if not icon_classes:
            raise ValueError("Не удалось извлечь классы из имен файлов.")

        with ThreadPoolExecutor() as executor:
            for icon_class in icon_classes:
                output_class_dir = Path(self.output_dir) / icon_class
                output_class_dir.mkdir(parents=True, exist_ok=True)
                futures = [executor.submit(self.generate_one_icon, icon_class) for _ in range(num_images_per_class)]
                for i, future in enumerate(futures):
                    image = future.result()  
                    image.save(output_class_dir / f"{i}.png")


generator = SignGenerator(icons_dir, backgrounds_dir, output_dir)
generator.generate_all_data(num_images_per_class=1000)
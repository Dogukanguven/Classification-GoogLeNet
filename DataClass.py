"""
@author: Dogukan
"""
import torch, os
import numpy as np
import albumentations as albu
from matplotlib import pyplot as plt


class dataclass():
    def __init__(self, df, root, phase):
        self.df = df
        self.length = df.shape[0]
        self.root = root
        if phase == "train":
            self.transforms = albu.Compose([        # Verinin transformunu yapan satır.
                albu.SmallestMaxSize(256),          # Görüntüyü resize eden satır.
                albu.HorizontalFlip(p=0.5),         # Görüntüyü çerviren satır.
                albu.Cutout(),                      # Görüntüye küçük kayıplar veren satır.
                albu.RGBShift(),                    # Görüntünün rengini kaydıran satır.
                albu.Rotate(limit=(-90, 90)),       # Görüntünün rotasyonunu ayarlayan satır.
            ])
        elif phase == "val" or phase == "test":
            self.transforms = albu.Compose([
                albu.Resize(256, 256),              # Görüntüyü resize eden satır.
            ])

    def __getitem__(self, index):
        label = self.df.iloc[index, 1:].to_numpy()  # Dataframe den seçilen indexe ait verilerden label niteliğini alır.
        image_id = self.df.iloc[index, 0]           # Seçilen indexteki resim id sini çeker
        path = os.path.join(self.root, str(image_id) + ".jpg")
        img = plt.imread(path)                      # Resim okunur.
        img = self.transforms(image=np.array(img))  # Resim transform yapılır. albu kütüphanesi yardımıyla.
        img = np.transpose(img['image'], (2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float)  # Fotografın np arrayı bir tensor a çevrilir.
        label = np.argmax(label)                    # Label verilerinin içerisinden 1 olan label ın indexi seçilir.
        return img, label                           # Resim ve etiketi geri döndürülür.

    def __len__(self):
        return self.length

    def label_name(self, label):                    # Tahmin değeri en yüksek olan sınıfın adını verir.
        breeds = self.df.columns.values
        breeds = breeds[1:]
        idx = np.argmax(label)
        return breeds[idx]
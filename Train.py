"""
@author: Dogukan
"""
from torchvision import models
from DataClass import dataclass
from torchsummary import summary
from torch.utils.data import DataLoader
from livelossplot import PlotLossesPoutyne
from poutyne.framework import ModelCheckpoint
from poutyne.framework import Model, EarlyStopping
from sklearn.model_selection import train_test_split
import os, torch, random, warnings, numpy as np, pandas as pd


def Train(pytorch_model, model_name, train_loader, valid_loader, learning_rate, epochs):
    plotlosses = PlotLossesPoutyne()                                                    # Plotlosses'ın tanımlanması.
    callbacks = [                                                                       # Callbacks ler.
        ModelCheckpoint(model_name + '_last_epoch.ckpt', temporary_filename='last_epoch.ckpt.tmp'),
        EarlyStopping(monitor='val_acc', patience=0, verbose=True, mode='max'),
        plotlosses
    ]
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=learning_rate)           # Optimizasyon fonksiyonu.
    loss_function = torch.nn.CrossEntropyLoss()                                         # Loss fonksiyonu.
    model = Model(pytorch_model, optimizer, loss_function, batch_metrics=['accuracy'])  # Model tanımlayıcısı
    model.to(device)
    model.fit_generator(train_loader, valid_loader, epochs=epochs, callbacks=callbacks) # Eğtimi başlatan satır.
    torch.save(model.state_dict(), 'New-Weights.pt')                                    # Modeli diske kaydeden satır.
    return None


def Veri_Oku(NUM_CLASSES):
    labels = pd.read_csv("csv-file/labels.csv")
    selected_breed_list = list(
        labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
    labels = labels[labels['breed'].isin(selected_breed_list)].reset_index(drop=True)
    df1 = labels['breed']
    df2 = labels["id"]
    df1 = pd.get_dummies(df1)
    df = pd.concat([df2, df1], axis=1)
    return df


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 61
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    etiket = ['circle', 'cross', 'heptagon', 'hexagon', 'octagon', 'pentagon',
              'quarter_circle', 'rectangle', 'semi_circle', 'square', 'star', 'trapezoid',
              'triangle']
    Data_Frame = Veri_Oku(len(etiket))                                                  # Dosyadan veri setini okuyan satır.
    df_train, df_val = train_test_split(Data_Frame, test_size=0.1, random_state=42)     # Train verisini bölen satır.
    traindata = dataclass(df_train, root="train", phase="train")                        # Train verisi sınıfını oluşturan komut
    valdata = dataclass(df_val, root="train", phase="val")                              # Validasyon verisi sınıfını oluşturan komut
    train_loader = DataLoader(traindata, batch_size=64, num_workers=0)                  # Train verisini paketleyen satır.
    valid_loader = DataLoader(valdata, batch_size=64, num_workers=0)                    # Validasyon Verisini paketleyen satır.

    dataiter = iter(train_loader)                                                       # Veriden tek paket alan satır.
    image, label = dataiter.next()                                                      # Paketten görüntü ve etiket alan satır

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Hardware used: ' + str(device))

    googlenet = models.googlenet(pretrained=True).to(device)                            # Model yapısının yüklendiği satır.
    summary(model=googlenet, input_size=(3, 256, 256), batch_size=32)                   # Modoel Özeti

    learning_rate = 0.001
    epochs = 30
    Train(googlenet, 'GoogleNet', train_loader, valid_loader, learning_rate, epochs)    # Eğitimin başlatıldığı satır.

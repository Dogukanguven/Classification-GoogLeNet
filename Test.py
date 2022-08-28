"""
@author: Dogukan
"""
import numpy as np
import pandas as pd
import seaborn as sn
import torch, torchvision
from sklearn import metrics
from DataClass import dataclass
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def test_submission(model, testloader):                     # Test fonksiyonu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cuda seçimi yapan satır.
    model.to(device)                                        # Modelin yüklendiği satır.
    sub_output = []                                         # Tahmin listesi
    model.train(False)                                      # Her ihtimale karşı modelin train ayarını kapatan satır.
    real_list = []                                          # Gerçek etiketlerin listesi.
    full = len(testloader)
    sayac = 0
    for data in testloader:                                 # Modele tahmin yaptıran döngü
        sayac += 1
        print(str(sayac) + " / " + str(full))
        inputs, labels = data                               # Bir batch lik veriden görüntü ve etiketi alan satır.
        real_list.append(int(labels))                       # Görüntünün gerçek etiketini listeye ekleyen satır.
        inputs = inputs.to(device)                          # Görüntüyü cuda ya uygun hale getiren satır.
        outputs = model(inputs)                             # Görüntünün predict edildiği satırç
        sub_output.append(outputs.data.cpu().numpy())       # Tahmin sonuçlarının listeye eklendiği satır.

    sub_output = np.concatenate(sub_output)                 # Sub_output içerisindeki ayrı dizileri birleştiren satır.

    for idx, row in enumerate(sub_output.astype('float')):
        sub_output[idx] = np.exp(row)/np.sum(np.exp(row))   # Başarı değerlerini dönüştüren döngü.
    predict = []
    for idx in sub_output:                                  # Tek görüntü için en yüksek değerli tahmini seçen döngü.
        predict.append(np.argmax(idx))
    return pd.Series(predict), pd.Series(real_list)


def Read_Data(NUM_CLASSES):
    labels = pd.read_csv("csv-file/Test_labels.csv")
    selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
    labels = labels[labels['breed'].isin(selected_breed_list)].reset_index(drop=True)
    df1 = labels['breed']
    df2 = labels["id"]
    df1 = pd.get_dummies(df1)
    df = pd.concat([df2, df1], axis=1)
    return df


if __name__ == '__main__':
    etiket = ['circle', 'cross', 'heptagon', 'hexagon', 'octagon', 'pentagon',
              'quarter_circle', 'rectangle', 'semi_circle', 'square', 'star', 'trapezoid',
              'triangle']
    Data_Frame = Read_Data(len(etiket))                                 # Dosyadan veri setini okuyan satır.
    testdata = dataclass(Data_Frame, root="test", phase="test")         # Test verisi sınıfını oluşturan komut
    test_loader = DataLoader(testdata, batch_size=1, num_workers=0)     # Veriyi paketleyen satır.

    model = torchvision.models.googlenet(pretrained=True).to('cuda')    # Model yapısının yüklendiği satır.
    model.load_state_dict(torch.load("GoogleNet_last_epoch.ckpt"))      # Modelin yüklendiği satır.
    Predict, Test = test_submission(model, test_loader)                 # Test işlemini başlatan satır.

    print("\n\n")
    print("Accuracy score: " + str(metrics.accuracy_score(Test, Predict)))
    # Doğru sınıflandırmanın toplama bölümüdür.
    print("Precision score: " + str(metrics.precision_score(Test, Predict, average='micro')))
    # Tüm sınıflardan, doğru olarak ne kadar tahmin edildiğinin bir ölçüsüdür.
    print("Recall score: " + str(metrics.recall_score(Test, Predict, average='micro')))
    # Positive olarak tahmin etmemiz gereken işlemlerin ne kadarını Positive olarak tahmin ettiğimizi gösterir.
    print("F1 score: " + str(metrics.f1_score(Test, Predict, average='micro')))
    # gerçek pozitif değerlerin oranının (recall) ve hassasiyetin (precision) harmonik ortalamasıdır.

    cnf_matrix = metrics.confusion_matrix(Test, Predict)                # Confusion Matrisini oluşturan satır.
    df_cm = pd.DataFrame(cnf_matrix, index=etiket, columns=etiket)      # Matrisi dataframe e çeviren satır.
    plt.figure(figsize=(12, 12), dpi=600)
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, square=True, cbar=False, fmt='g')
    plt.savefig("Confusion.png")
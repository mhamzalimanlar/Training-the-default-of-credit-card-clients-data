#kullanılacak kütüphaneler import edlilmiştir.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("dataset/default of credit card clients.csv")#veri okunmuştur ve df değişkenine atanmıştır.
#basit ön veri analizi 
print(df.head(50)) #tahmin edilecek sütunu belirlemek için verinin ilk 50 satırı gözlemlenmiştir.
print(df.info()) # sütun(değişken) içerikleri(veri tipi,eksik veri bilgisi, veri boyutu) incelenmiştir.
#tahmin edilecek(bağımlı)değişken ve üzerinden tahmin edilecek(bağımsız)değişkenleri ayırmak için sütun(değişken) isimlerine bakılmıştır.
print(df.columns)
X=df[['ID','EDUCATION', 'MARRIAGE','AGE','PAY_1','LIMIT_BAL',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX']]#bağımsız değişkenler belirlenmiş ve X değişkenine atanmıştır.
y=df.dpnm  # seçilen bağımlı değişken (tahmin edilecek değişken)y değişkenine atanmıştır

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=39)#test,eğitim verisi oluşturularak ilgili değişkenlere atanmıştır.

#borcunu ödememiş = 0,borcunu ödemiş=1
a=df["dpnm"].value_counts()/df.dpnm.size*100 #yüzdelikler hesaplanarak dengesiz dağılım kontrol edilmiştir.
print(a)# müşterilerin yüzde kaçının borcunu ödemediği yüzde kaçının ödediği yazdırılmıştır.

# normalizasyon

scaler = MinMaxScaler()#Min-Max:En küçük değer 0 ve en büyük değer 1 olacak şekilde veriyi normalize eder.
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)#normalizasyon işlemi eğitim verisine uygulanmıştır
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)#normalizasyon işlemi test verisine uygulanmıştır.

# Train

lr = LogisticRegression() #lojistik regresyon modeli lr değişkenine atanmıştır.
model = lr.fit(X_train, y_train) # eğitim verileri lojistik regresyon modeline göre eğitilmiştir.
y_pred = lr.predict(X_test) # test verisi üzerinden ödeme durumu tahmin edilmiştir. (tahmin edilerek elde edilen bağımlı değişken )

accuracy1 = accuracy_score(y_pred=y_pred, y_true=y_test)#test verisi üzerinden tahmin edilen ödeme durumunun tahmin(model)doğruluğu hesaplanmıştır

print("Model(Test) accuracy : {}".format(accuracy1)) #hesaplanan tahmin (model) doğruluğu ekrana yazdırılmıştır.

y_pred_train = lr.predict(X_train) #eğitilen verinin tahmin doğruluğunun hesaplanabilmesi için eğitim verisi ile ödeme durumu tahmin edilmiştir.
accuracy = accuracy_score(y_pred=y_pred_train, y_true=y_train)#test verisi üzerinden eğitilen verinin tahmin doğruluğu hesaplanmıştır
print("Train accuracy : {}".format(accuracy))#hesaplanan eğitilen verinin tahmin(eğitim) doğruluğu ekrana yazdırılmıştır.
print(classification_report(y_pred=y_pred, y_true=y_test))# precision,recall,f1-score metrikleri hesaplanmıştır

#Confusion Matrix

cm1 = cm(y_pred=y_pred,y_true=y_test) #confusıon matrix tahmin edilen ve test veriler ile oluşturulmuştur.
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, cmap = 'Greens')#confusıon matrix  elde edilmiştir
plt.ylabel('Gerçekleşen')#confusıon matrix'in x ve y eksenlerine etiket verilmiştir.
plt.xlabel('Tahmin Edilen')
plt.title('Model Doğruluğu: {0}'.format(accuracy1), size = 12)#model doğruluğu confusıon matrix'e dahil edilmiştir.
plt.show()#confusıon matrix gösterilmesi show komutuyla sağlanmıştır.
print("confusion matrix \n",cm1)#confusıon matrix ekrana yazdırılmıştır.


"""
(default of credit card veri setinde;
Müşterilerin eğitim,medeni durumu ,yaşı gibi değişkenlerden yola çıkılarak kredi borçlarını ödeyip ödemediği tahmin edilmiştir.)

(veride çok değişken olduğundan Logistic Regression modeli seçilmiştir.)
Logistic Regression modelinde kullanılmak üzere veri seti;
%80‘i eğitim %20’si test verisi olmak üzere sklearn.model_selection kütüphanesinin train_test_split fonksiyonu kullanarak ayrılmıştır.
(veri seti,test ve eğitim verisi oluşturarak modelin veriyi anlayıp tahmin edeceği değerler(eğiteceği veriyi)'i görmesi için,
eğitim verisi ve test verisi olarak bölünmüştür.)


kullanıcıların %77.88'i borcunu ödemezken %22.12'si borcunu ödemiştir buradan verinin dengesiz dağılan veri olduğu gözlemlenmiştir.

Veriyi modele hazırlamak için normalizasyon tekniği kullanılmıştır.Normalizasyon ile veri tekrarlarını ortadan kaldırarak doğruluk arttırılmıştır.

eğitim verisi doğruluk değeri tahminlerimizin %81'inin doğru olabileceğini göstermektedir.Doğruluk değerleri veri setimiz dengesiz dağılan veri 
olduğu için bizi yanıltmaktadır.

Lojistik regresyon modeli uygulandığında Doğruluk Skoru %80 olarak bulunmuştur. Fakat veri seti dengesiz dağılan bir veri seti olduğu için yalnızca 
Doğruluk skoru üzerinden  modelin yeterli olduğunu söylemek yanlış olacaktır.Bu yüzden classification report tablosundan borç ödeme durumunun 
Precision,Recall ve f1-skorlarına bakıldığında aslında borç ödeme durumu (f1skor=0.32) 0'a yakın olduğu için modelin yetersiz olduğu görülmektedir.
Ancak borç ödememe durumunun aynı değerlerine bakıldığında borç ödememe durumu(f1skor=0.88)1'e yakın olduğu için modelin yeterli olduğu görülmektedir.
f1 skorun doğruluk yerine kullanılmasının nedeni verimiz gibi dengesiz dağılan verilerde hatalı model seçimi yapmamaktır.Yani borç ödememe durumunun,
f1 skorundan dolayı borç ödememe durumu için doğru model seçildiği sonucu çıkarılmıştır.

Recall (Duyarlılık):borcunu ödememiş müşteri olarak tahmin edilen müşteriler %98 oranında doğru olarak tahmin edilmiştir.
                    borcunu ödemiş müşteri olarak tahmin edilen müşteriler %20 oranında doğru olarak tahmin edilmiştir.

Precision (Kesinlik):borcunu ödemiş müşteri olarak tahmin edilen müşterilerin %75'i gerçekten borcunu ödemiş müşterilerdir.
                     borcunu ödememiş müşteri olarak tahmin edilen müşterilerin %80'i gerçekten borcunu ödememiş müşterilerdir.


Confusıon Matrix'in yorumlanması ;

tn(true negative):borcunu ödemeyen müşteriyi ödemeyen müşteri olarak tahmin etme 
tn = 4526 müşteri 
fn(false negative):borcunu ödemeyen müşteriyi ödeyen müşteri olarak tahmin etme 
fp = 94 müşteri 
tp(true pozitif):borcunu ödeyen müşteriyi ödeyen müşteri olarak tahmin etme 
tp = 277 müşteri
fp(false positive):borcunu ödeyen müşteriyi ödemeyen müşteri olarak tahmin etme 
fn = 1103 müşteri 
(confusion(karışıklık)matrisininin tn değerinden görüldüğü üzere borcunu ödemeyen müşteriyi ödemeyen müşteri olarak tahmin etme oranı daha yüksektir.)
"""

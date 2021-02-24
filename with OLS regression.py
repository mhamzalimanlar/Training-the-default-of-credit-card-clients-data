#kullanılacak kütüphaneler import edlilmiştir.
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("dataset/default of credit card clients.csv")#verinin okunması ve df değişkenine atanması 
#basit ön veri analizi 
print(df.head(50)) #verini ilk 50 satırı gözlemlenmiştir.
print(df.info()) # sütun(değişken) içerikleri(veri tipi,eksik veri bilgisi,veri boyutu vb.) incelenmiştir.
print(df.columns)#bağımlı değişken ve bağımsız değişkenleri belirlemek ve ayırmak için sütun(değişken) isimlerine bakılmıştır.
X=df[['ID','EDUCATION', 'MARRIAGE','AGE','PAY_1','LIMIT_BAL',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX']]#bağımsız değişkenler belirlenmiş ve X değişkenine atanmıştır.
X = X.drop(X.columns[[0,8,9,10,12,13,14,15,16,18,19,20,21,22]],axis=1)#elenecek anlamsız değişkenler drop() methodu kullanılarak indis numaralarıyla silinmiştir
y=df.dpnm  # seçilen bağımlı değişken y değişkenine atanmıştır.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)#test,eğitim verisi oluşturularak ilgili değişkenlere atanmıştır.

# OLS Regresyon 
X_multi = sm.tools.tools.add_constant(X_train, prepend=True, has_constant="skip")
model = sm.OLS(y_train, X_multi)#ols regresyon modeli eğitim verisi üzerinde uygulanarak model değişkenine atanmıştır.
res = model.fit() #eğitim verisi ols regresyon modeline göre eğitilmiştir.
print(res.summary())#sonuç(ols regresyon sonuç tablosu)ekrana yazdırılmıştır.
"""
OLS Regression modelinde kullanılmak üzere veri seti;
%70‘i eğitim %30’u test verisi olmak üzere sklearn.model_selection kütüphanesinin train_test_split fonksiyonu kullanarak ayrılmıştır.
(veri seti,test ve eğitim verisi oluşturarak modelin veriyi anlayıp eğitim verisini görmesi için,
eğitim verisi ve test verisi olarak bölünmüştür)

OLs Regresyon tablosuna bakıldığında;(P değeri 0,05 olarak belirlenmiştir.)
bağımlı değişkene (modele) anlamlı etkisi bulunmayan bağımsız değişkenler(Pdeğeri>0.05 olanlar) silinerek uygun bir model kullanılmaya çalışılmıştır.

[ID, PAY_4, PAY_5 , PAY_6 , BILL_AMT2 , BILL_AMT3 , BILL_AMT4 , BILL_AMT5, BILL_AMT6, PAY_AMT2 , PAY_AMT3 , PAY_AMT4, PAY_AMT5,PAY_AMT6]
isimli değişkenler P>0.05 olduğundan silinmiştir.

silinen değişkenler ;PAY_3(P= 0,003),PAY_AMT1 (P=0.002) değişkenlerinin P değerlerinde çok küçük azalmalara neden olarak sıfırlamıştır.
Yani bu iki değişken anlamlı etkisi bulunmayan değişkenler silindikten sonra istatistiksel olarak yüksek düzeyde anlamlı değişken'iken 
çok yüksek düzeyde anlamlı değişken olmuşlardır.


EDUCATION,MARRIAGE,AGE,PAY_1,LIMIT_BAL,PAY_2,BILL_AMT1,değişkenleri ise modelimiz için istatiksel olarak çok yüksek düzeyde anlamlı değişkenlerdir.
Çünkü herbirinin P değeri 0.000'dır.Yani P değerleri 0.001'den küçüktür.

Diğerlerinden farklı olarak değerlendirecek olursak SEX(P=0.008) değişkeni modelimiz için yüksek düzeyde anlamlı değişkendir.
Çünkü P değeri>0.001 den büyüktür.

R kare değeri = 0.127 olarak hesaplandığından eğitim verilerinde kullanılan modelin uyumlu olmadığı sonucu çıkarılmıştır.

çarpıklık katsayısına bakıldığında değer +1 e daha yakındır.yani veri sağa çarpıktır.Skew=1.20
basıklık Kurtosis değerine bakıldığında değer (k > 3) 3'ten büyüktür.yani veri sivridir.Kurtosis=3.18
"""
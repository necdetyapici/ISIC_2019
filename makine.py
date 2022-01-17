
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

dataMeta = pd.read_csv("ISIC_2019_Training_Metadata.csv" ,                          #MetaData dataframenin 
                   usecols =['image','age_approx','anatom_site_general','gender'])  #istenilen kolonlarını çektik.



dataMeta=dataMeta.dropna()                                                          #Herhangi NaN değeri varsa o satırı sildik.


groundTruth= pd.read_csv("ISIC_2019_Training_GroundTruth.csv")                          #GroundTruth dataframei çektik.



sinifDf = pd.DataFrame([x for x in np.where(groundTruth == 1,                           #GroundTruth tablosunun kolonlarında hucre değeri 
                                            groundTruth.columns,'').flatten().tolist()  # 1 olanları alarak sinifDf dataframeni oluşturduk.
                             if len(x) > 0], columns= (["Sinif"]))


mainDf = pd.concat([dataMeta, sinifDf], axis=1, join='inner')                           #dataMeta ile sinifDf dataframelerini satır 
                                                                                       #bazında birleştirdik. Önceliği dataMeta dataframede oldu.


dictData = {                                                                            #Kategorileştirmek için sınıf kodu için sözlük hazırladık.
    'MEL': 'Melanoma',
    'NV': 'Melanocytic nevus',
    'BCC': 'Basal cell carcinoma',
    'AK': 'Actinic keratosis',
    'BKL': 'Benign keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular lesion',
    'SCC': 'Squamous cell carcinoma',
    'UNK': 'None of the others'
}


mainDf['Sinif Turleri'] = mainDf['Sinif'].map(dictData.get) #Sınıfın Türleri adında bir kolon oluşturup 
                                                                 #sözlüğümüzü yolladık. Sözlüğümüze göre kolonu doldurdu.


mainDf['SinifId'] = pd.Categorical(mainDf['Sinif']).codes    #SınıfId adında kolon oluşturduk. İçine ise kanser 
                                                             #türlerinin Idsini otomatik gönderdik.(Id leri 
                                                             #alfabetik sıraya ve başlangıç 0 olarak atıyor.)


my_colors = list(islice(cycle(['#4169E1', '#4682B4', '#6495ED','#1E90FF', '#00BFFF', '#87CEFA','#ADD8E6','#B0E0E6','#B0C4DE']), None, len(mainDf))) #Grafiklerde Kullanılacak renkleri tanımladık.



#Sınıf Türlerinin sahip oldukları örnek sayılarının dağılımını gösteren grafik
fig1 , ax1 = plt.subplots(1, 1, figsize= (10,5))                                     
mainDf['Sinif Turleri'].value_counts().plot(kind='bar', ax=ax1, color=my_colors)     
ax1.set_title('Sınıf Türlerinin sahip oldukları örnek sayılarının dağılımını gösteren grafik',
              fontsize=16)
plt.legend()
fig1.tight_layout()
fig1.savefig('sinifdagilim.png')




#Lezyonun bulunduğu bölgeye örnek sayılarının dağılımını gösteren grafik
fig2 , ax2 = plt.subplots(1, 1, figsize= (10, 5))
mainDf['anatom_site_general'].value_counts().plot(kind='bar',ax=ax2, color=my_colors)         
ax2.set_title('Lezyonun bulunduğu bölgeye örnek sayılarının dağılımını gösteren grafik',
              fontsize=16)
plt.legend()
plt.tight_layout()
fig2.savefig('lezyondagilim.png')


#Cinsiyete göre verilerin dağılımını gösteren grafik
fig3 , ax3 = plt.subplots(1, 1, figsize= (10, 5))
mainDf['gender'].value_counts().plot(kind='bar',ax=ax3,color=my_colors)                      
ax3.set_title('Cinsiyete göre verilerin dağılımını gösteren grafik',
              fontsize=16)
plt.legend()
plt.tight_layout()
fig3.savefig('cinsiyetdagilim.png')



#Verinin yaş dağılımını gösteren grafik
fig4 , ax4 = plt.subplots(1, 1, figsize= (10, 5))
mainDf['age_approx'].hist(bins=25, color = "#20B2AA")
ax4.set_title('Verinin yaş dağılımını gösteren grafik',
              fontsize=16)
plt.tight_layout()
fig4.savefig('yasdagilim.png')



#Sınıfı Melanoma olan örneklerde cinsiyet dağılımını veren grafik.
melGender=mainDf[['SinifId','gender']]                                          #mainDf dataframeden Sınıf ve gender kolonunun alıyoruz.
filtreMell = melGender.SinifId == 4                                             #MEL sınıfının Id'si 4.
filtreMel = melGender[filtreMell]                                               #Oluşturduğumuz filtreyi melGender dataframe atayarak filtreliyoruz.
fig5 , ax5 = plt.subplots(1,1,figsize = (10,5))
ax5.set_title('Sınıfı Melanoma olan örneklerde cinsiyet dağılımını veren grafik',
              fontsize=16)
filtreMel['gender'].value_counts().plot(kind='bar',ax=ax5, color=my_colors)                           
plt.legend()
plt.tight_layout()
fig5.savefig('melanomacinsiyet.png')



#Sınıfı Melanoma olan örneklerde lezyonun bulunduğu bölgeye göre dağılımını veren grafik.
melAnatom=mainDf[['SinifId','anatom_site_general']]                                                         #mainDf dataframeden Sınıf ve anatom_site_general kolonunun alıyoruz.
filtreAnatomm = melAnatom.SinifId == 4                                                                      #MEL sınıfının Id'si 4.
filtreAnatom = melAnatom[filtreAnatomm]                                                                     #Oluşturduğumuz filtreyi melAnatom dataframe atayarak filtreliyoruz.
fig6 , ax6 = plt.subplots(1,1,figsize = (10,5))
ax6.set_title('Sınıfı Melanoma olan örneklerde lezyonun bulunduğu bölgeye göre dağılımını veren grafik',
              fontsize=16)
filtreAnatom['anatom_site_general'].value_counts().plot(kind='bar',ax=ax6, color=my_colors)   
plt.legend()
plt.tight_layout()
fig6.savefig('melanomalezyon.png')


mainDf.to_csv('resultData.csv', index = False)               #Eğitimler için en son ulaştığımız DataFrame i csv olarak kaydediyoruz.

mainDf=mainDf.drop(['Sinif Turleri','SinifId'],axis=1)




#EĞİTİM





le = preprocessing.LabelEncoder()                           #Kategorileri Sayısallaştırdık.
dtype_object=mainDf.select_dtypes(include=['object'])
for x in dtype_object.columns:
    mainDf[x]=le.fit_transform(mainDf[x])


X = mainDf.iloc[:,1:4]                                      #Bağımlı ve Bağımsız Değişkenleri tanımladık.
y = mainDf.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 )    #Eğitim ve test için verileri %80 - %20 böldük.

sc = StandardScaler()                               #Sayısal verileri ölçeklendirdik.
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

algorithms=[]                                       #Algoritmaların başarı durumlarını grafikleştirmek için 
score=[]                                                   #2 tane dizi oluşturduk ve algoritma değerleri içine atacağız.




#KNN

knn=KNeighborsClassifier(n_neighbors=244, algorithm='kd_tree') 
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)                                
ac = accuracy_score(y_test, y_pred)                                     
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred)) 
algorithms.append("KNN")
score.append(ac)
print('KNN accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nConfusion Matrix:')
print(cm)

fig7,ax7=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax7)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" KNN Confusion Matrix")
fig7.savefig('knnConfusionMatrix.png')




#Navie-Bayes

nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Navie-Bayes")
score.append(ac)

print('Navie-Bayes accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nKonfüzyon Matrisi:')
print(cm)


fig8,ax8=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax8)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" Navie-Bayes Confusion Matrix")
fig8.savefig('navieBayesConfusionMatrix.png')



##Support Vector Machine

svm = SVC(random_state=1,kernel='linear', C=1, degree=3, gamma='scale', max_iter=-1)
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Support Vector Machine")
score.append(ac)
print('Support Vector Machine accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nKonfüzyon Matrisi:')
print(cm)

fig9,ax9=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax9)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" Support Vector Machine Confusion Matrix")
fig9.savefig('supportVectorMachineConfusionMatrix.png')


#DecisionTree

dt=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Decision Tree")
score.append(ac)
print('Decision Tree accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nKonfüzyon Matrisi:')
print(cm)

fig10,ax10=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax10)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" Decision Tree Confusion Matrix")
fig10.savefig('decisionTreeConfusionMatrix.png')


# LogisticRegression


lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Logistic Regression")
score.append(ac)
print('Logistic Regression accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nKonfüzyon Matrisi:')
print(cm)

fig11,ax11=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax11)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" LogisticRegression Konfüzyon Matrisi")
fig11.savefig('logisticRegressionConfusionMatrix.png')





# Yapay Sinir Ağları


sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.01, max_iter=100)
sknet.fit(X_train, y_train)
y_pred = sknet.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Yapay Sinir Ağları")
score.append(ac)
print('Yapay Sinir Ağları accuracy: %' + str(ac * 100))
print('--------------------------------------\n')
print(cr)
print('--------------------------------------\nKonfüzyon Matrisi:')
print(cm)

fig12,ax12=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax12)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title(" Yapay Sinir Ağları Konfüzyon Matrisi")
fig12.savefig('yapaySinirAglariConfusionMatrix.png')


#Algoritmaların Başarı Düzeylerinin Grafiği


x_pos = [i for i, _ in enumerate(algorithms)]
xada=pd.DataFrame(score,algorithms)
print(xada)
fig13,ax13=plt.subplots(figsize=(10,5))
plt.bar(x_pos, score, color=my_colors)
plt.ylabel("Başarı Oranı")
plt.title("Algoritmaların Başarı Oranları")
plt.xticks(x_pos, algorithms,rotation=90)
fig13.savefig('basariOrani.png')
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D

class DataBase:
    def __init__(self):
        self.connection = pymysql.connect(
            host = 'localhost',
            user = 'root',
            password = 'n0m3l0',
            db = 'ortopeñabd'
        )

        self.cursor = self.connection.cursor()

        print('Conexion Establecida')

    def select_user(self,id):
        sql = 'SELECT * FROM ctratamientos where id_tra={}'.format(id)

        try:
            self.cursor.execute(sql)
            user = self.cursor.fetchone()
            print('ID: ', user[0])
            print('Nombre: ', user[1])
        except Exception as e:
            print('No se pudo encontrar la tabla ' , e)

    def clustering():
        print(datos.head())

        #Vemos informacion estadística
        print(datos.describe())

        #Mostrar registros de cada tipo
        print(datos.groupby('categoria').size())

        #Graficar los datos para ver la dispersion
        #datos.drop(['categoria',1]).hist()
        #plt.show()

        #Cargamos las variables de seaborn en x, y la categoria en y
        x= np.array(datos[["op","ex","ag"]])
        y = np.array(datos['tratamiento'])
        x.shape

        #Grafica en 3D con 9 colores por las categorias
        fig = plt.figure()
        ax = Axes3D(fig)
        colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
        asignar = []
        for row in y:
            asignar.append(colores[row])
        ax.scatter(x[:,0],x[:,1],x[:,2], c=asignar,s=60)
        plt.show()

        #Obtenemos el valor de K para el KMeans
        Nc = range(1,20)
        kmeans = [KMeans(n_clusters=i) for i in Nc]
        kmeans
        score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
        score
        plt.plot(Nc,score)
        plt.xlabel('Numero de Clusters')
        plt.ylabel('Puntaje')
        plt.title('Curva Elbow')
        plt.show()

        #Ajustamos con mas clusters
        kmeans = KMeans(n_clusters=5).fit(x)
        centroides = kmeans.cluster_centers_
        print(centroides)

        #Prediciendo los clusters
        labels = kmeans.predict(x)
        #Obtenemos los centros de los clusters
        C = kmeans.cluster_centers_
        colores = ['red','green','blue','cyan','yellow']
        asignar = []
        for row in labels:
            asignar.append(colores[row])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x[:,0], x[:,1], x[:,2], c=asignar,s=60)
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
        plt.show()

        #Obteniendo los datos y ploteandolos
        f1 = datos['op'].values
        f2 = datos['ex'].values
        plt.scatter(f1,f2,c=asignar,s=70)
        plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
        plt.show()

        #Obteniendo los valores y ploteandolos
        f1 = datos['op'].values
        f2 = datos['ag'].values
        
        plt.scatter(f1, f2, c=asignar, s=70)
        plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
        plt.show()

        #Nuevamente
        f1 = datos['ex'].values
        f2 = datos['ag'].values
        
        plt.scatter(f1, f2, c=asignar, s=70)
        plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
        plt.show()

        #Veamos cuantos usuarios tiene cada cluster
        copia = pd.DataFrame()
        copia['usuario']=datos['usuario'].values
        copia['categoria']=datos['categoria'].values
        copia['label'] = labels;
        cantidadGrupo = pd.DataFrame()
        cantidadGrupo['color'] = colores
        cantidadGrupo['cantidad'] = copia.groupby('label').size()
        print(cantidadGrupo)

        #Veamos la diversidad en los rubros laborales de cada uno
        group_referrer_index = copia['label'] == 0
        group_referrals = copia[group_referrer_index]

        diversidadGrupo = pd.DataFrame()
        diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
        diversidadGrupo['cantidad']= group_referrals.groupby('categoria').size()
        print(diversidadGrupo)

        #Vemos el representante del grupo, el usuario centroide
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,x)
        print(closest)

        users = datos['usuario'].values
        for row in closest:
            print(users[row])

        x_new = np.array([[45.92,57.74,15.66]])
        new_labels = kmeans.predict(x_new)
        print(new_labels)



#database = DataBase()
#database.select_user(1)
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### kmeans 


# In[1]:


#importamos las librerias
import xlrd
import math 
import pandas as pd


# In[2]:


#cargamos los datos
prueba= pd.read_excel("C:/datos/base.xlsx")


# In[3]:


#comprobamos si los datos estan correctos
prueba.head()


# In[4]:


from sklearn.datasets import make_blobs


# In[5]:


import numpy as np
import matplotlib.pyplot as plt 


# In[6]:


#Miramos las variables que deben ser numericas
prueba.info()


# In[7]:


#eliminamos la primera columna 


# In[8]:


#ELIMINAMOS LAS COLUMNAS NO NUMERICAS
datos=prueba.drop(['SEXO','Ubicación geográfica', 'ESTADO CIVIL','ID'],axis=1)


# In[9]:


#chequemaos las variables
datos.info()


# In[10]:


#miramos promedio min o max
datos.describe()


# In[11]:


#normalizamos los datos
datos_nor=(datos-datos.min())/(datos.max()-datos.min())
datos_nor


# In[12]:


#los datos estan normalizaddos con min en cero y max en 1
datos_nor.describe()


# In[13]:


# KMeans está en el paquete sklearn.cluster
from sklearn.cluster import KMeans


# In[14]:



# calculamos los cluster codo de yambu

wcss =  [] 
for i in range (1, 10):
    kmeans = KMeans (n_clusters=i, max_iter = 300 ) 
    kmeans.fit(datos_nor)    #aplico k means a la base de datos
    wcss.append(kmeans.inertia_)


# In[15]:


## Graficamos los resultados
plt.plot(range(1,10), wcss)
plt.title("codo de jambu")
plt.xlabel("Numero de cluster")
plt.ylabel("WCSS")
plt.show()


# In[16]:



### aplicando el metodo kmeans a la base de datos
clustering = KMeans (n_clusters = 3, max_iter = 300)
clustering.fit(datos_nor)


# In[17]:


#agregamos la clasificacion al archivo original
prueba ['KMeans_clusters']= clustering.labels_
prueba.head()


# In[18]:


#APLICAREMOS EL ANALISIS DE COMPONENTES PARA DARNOS UNA iDEA DE COMO SE FORMARON LOS CLUSTERS
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca_prueba= pca.fit_transform(datos_nor)
pca_prueba_df = pd.DataFrame(data= pca_prueba, columns= ["Componente1","Componente2"])
pca_nombres_prueba= pd.concat([pca_prueba_df, prueba[['KMeans_clusters']]],axis=1)
pca_nombres_prueba


# In[19]:


pca_nombres_prueba.Componente1


# In[20]:


pca_nombres_prueba


# In[26]:


fig = plt.figure(figsize= (6,6))

ax=fig.add_subplot(1,1,1)
ax.set_xlabel("Componente1", fontsize=15)
ax.set_ylabel("Componente2", fontsize=15)
ax.set_title("Componentes Principales", fontsize=20)

color_theme = np.array(["blue","green","orange"])

ax.scatter(x= pca_nombres_prueba.Componente1, y=pca_nombres_prueba.Componente2,
            c=color_theme[pca_nombres_prueba.KMeans_clusters],s=40)
          
plt.show()


# In[ ]:


#graficamos


# In[ ]:





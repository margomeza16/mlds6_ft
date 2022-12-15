import os
import zipfile
import numpy as np
import tensorflow as tf
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

orig_zip_train = '/content/drive/MyDrive/MachineLearningAvanzado/train.zip'
orig_zip_test = '/content/drive/MyDrive/MachineLearningAvanzado/test.zip'
orig_zip_val = '/content/drive/MyDrive/MachineLearningAvanzado/val.zip'
ruta_dest_train = '/tmp/train/'
ruta_dest_test = '/tmp/test/'
ruta_dest_val = '/tmp/val/'
class DataLoader:
"""
    Loader for zip files with endoscopies images.

    Attributes
    ----------
        train_gen: Contiene las imágenes de etrenamiento obtenidas al aplicar data augmentation
        X_train_prep: Arreglo numpy con las imagenes de entrenamiento obtenidas del archivo .zip cargado, prepocesadas cono resnet_v2.
        X_val_prep: Arreglo numpy con las imagenes de validación preporcesadas con resnet_v2.
        X_test_prep: Arreglo numpy con las imagenes de test preprocesadas con resnet_v2,
        Y_train : Etiquetas de clases de las imagenes de entrenamiento codificadas con one-hot representation.
        Y_test: Etiquetas de clases de las imagenes de test codificadas con one-hot representation.
        Y_val: Etiquetas de clases de las imagenes de validación codificadas con one-hot representation.
    
    
    """


    def __init__(self):
#        self.loader = mnist.load_data


    def load_file(orig, dest):
"""
	Descomprime y descarga archivos zip con imagenes
	
	Attributes
	----------
	orig: ruta origen de archivo .zip con imagenes
	dest: Ruta destino de descarga de los archivos con imagenes descomrpimidas
"""
	zip = zipfile.ZipFile(orig,'r')
    	zip.extractall(destino)
	return 0


   def preproceso(ruta_dest):
"""
       Guarda en arreglos de numpy los conjuntos de imágenes de entrenamiento, validación y prueba

       Atributes
       ---------

       ruta_dest: Ruta de los archivo de imágenes a convertir en erreglo de numpy

       Returns
       -------
       X_train :  Arreglo numpy con el conjunto de imágenes de entrenamiento.
       y_train :  Etiquetas de clasificación de las imágenes de entrenamiento.
       X_test  :  Arreglo numpy con el conjunto de imágenes de test.
       y_test  :  Etiquetas de clasificación de las imágenes de test.
       X_val   :  Arreglo numpy con el conjunto de imágenes de validación.
"""
       all_images = []
       labels = []
       for i, val in enumerate(["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"]):
       temp_path = f"{ruta_dest}{val}"
       for im_path in os.listdir(temp_path):
           all_images.append(np.array(tf.keras.preprocessing.image.load_img(temp_path+im_path,
                                                                         target_size=(224, 224, 3))))
           labels.append(i)

       if "train" in im_path:
    	  X_train = np.array(all_images)
	  y_train = np.array(labels)
          return X_train, y_train
       elif "test" in im_path:
          X_test = np.array(all_images)
	  y_test = np.array(labels)
          return X_test, y_test
	else:
	  X_val = np.array(all_images)
	  y_val = np.array(labels)
          return X_val, y_val

    def __call__(self):
"""
	Realiza llamado a las funciones load_file() y preproceso() por cada conjunto de imagenes descargadas (train, test y validación)
	Codifica las etiquetas de las clases usando one-hot representation.
	Realiza preprocesamiento de la ResNet50V2 para transformar los conjuntos, escalando los pixeles de entrada entre -1 y 1.
        Realiza transformaciones de las imagenes de entrenamiento mediante data augmentation


	Params
	------
	---

	Return
	------
	Retorna las propiedades del objeto DataLoader:
	train_gen: Contiene las imágenes de etrenamiento obtenidas al aplicar data augmentation 
	X_train_prep: Arreglo numpy con las imagenes de entrenamiento obtenidas del archivo .zip cargado, prepocesadas cono resnet_v2.
	X_val_prep: Arreglo numpy con las imagenes de validación preporcesadas con resnet_v2. 
	X_test_prep: Arreglo numpy con las imagenes de test preprocesadas con resnet_v2, 
	Y_train : Etiquetas de clases de las imagenes de entrenamiento codificadas con one-hot representation.
	Y_test: Etiquetas de clases de las imagenes de test codificadas con one-hot representation.
	Y_val: Etiquetas de clases de las imagenes de validación codificadas con one-hot representation.

"""
	load_file(orig_zip_train, ruta_dest_train)
	X_train, y_train = preproceso(ruta_dest_train)
        load_file(orig_zip_test, ruta_dest_test)
        X_test, y_test = preproceso(ruta_dest_test)
        load_file(orig_zip_val, ruta_dest_val)
        X_val, y_val = preproceso(ruta_dest_val)

        #Codificamos las etiquetas usando one-hot representation
        Y_train = to_categorical(y_train)
        Y_test =  to_categorical(y_test)
        Y_val =  to_categorical(y_val)

        #Debido a que utilizaremos la red convlucional ResNet50V2 para la extracción de características,
	#utilizamos el preprocesamiento de la ResNet50V2 para transformar los conjuntos, escalando los pixeles de entrada entre -1 y 1:
  	X_train_prep = tf.keras.applications.resnet_v2.preprocess_input(X_train)
  	X_val_prep = tf.keras.applications.resnet_v2.preprocess_input(X_val)
  	X_test_prep = tf.keras.applications.resnet_v2.preprocess_input(X_test)

       #Definimos las transformaciones para data augmentation
  	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='constant')

  	#Se aplica data augmentation sobre el conjunto de imágenes de entrenamiento
        batch_size=32 
  	train_gen = train_datagen.flow(X_train_prep, Y_train, batch_size=batch_size)

        return train_gen, X_train_prep, X_val_prep, X_test_prep, Y_train,Y_test, Y_val



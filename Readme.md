# 2025-1C-VA
Repositorio de TPs de la materia Visión Artificial
### Grupo nº 5 - Integrantes
* Pompeo, Nicolas Ruben
* Ramos, Alan Gonzalo
* Báez, Matías Nahuel
## Proyecto nº1: Detección y Clasificación de Formas
### Proyecto
Los alumnos elegirán al menos tres objetos reconocibles por su contorno, usarán una imagen de cada uno como referencia de cada tipo de objeto, y les asignarán un nombre a cada uno.

Prepararán el ambiente controlado para evitar correcciones por software que de otro modo pueden resultar muy demandantes, y así poder concentrarse en el objetivo del proyecto.

El sistema detectará y clasificará los objetos en la imagen de la webcam en tiempo real.

### Proceso
Se sugieren los siguientes pasos para el proceso de la imagen de la webcam:
1. Convertir la imagen a monocromática
2. Aplicar un threshold con umbral ajustable con una barra de desplazamiento
   * se pueden incluir opciones de ajuste automático
3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
   * Es buena idea incluir una barra de desplazamiento para ajustar el tamaño del elemento estructural
4. El sistema puede obtener varios contornos en una misma imagen, y los debe procesar a todos individualmente
5. Filtrar contornos que se pueden descartar de antemano
  * para quitar contornos espúreos indeseables
  * por ejemplo, por tener un área muy pequeña
6. Compara cada contorno con todos los objetos de referencia, usando matchShapes()
  * establecer un umbral de distancia máxima de validez
    * es conveniente una barra deslizante para ajustar este valor
  * clasificar la forma según la menor distancia entre los candidatos válidos
    * si no hay ninguno, la forma es desconocida
7. Generar la imagen anotada

### Output
La salida del sistema es una ventana con la imagen original anotada de la siguiente manera:
* localización de objetos relevantes
  * en verde para objetos reconocidos
    * alternativamente se puede asignar un color particular a cada clase de objeto
  * en rojo para objetos desconocidos
  * evitando mostrar contornos de ruidos y elementos espúreos
  * se puede anotar el contorno del objeto o un rectángulo que lo contenga
* etiqueta con el nombre del objeto

## Proyecto nº2: Clasificación con Machine Learning
### Proyecto
Los alumnos elegirán al menos tres objetos reconocibles por su contorno, y elaborarán un dataset de muestras y sus correspondientes etiquetas.  Cada muestra consiste en los siete invariantes de Hu como mínimo; los alumnos podrán agregar otras características.
Entrenarán un algoritmo de aprendizaje automático (usualmente un clasificador basado en decision trees) sobre el dataset de descriptores y guardarán en un archivo el modelo resultante.
Un sistema final, similar al proyecto anterior, detectará los objetos en la imagen de la webcam en tiempo real y los clasificará según las predicciones del modelo entrenado en lugar de usar matchshapes.
A continuación se sugieren maneras de encarar el generador de descriptores, el entrenador y el clasificador.  Los alumnos que lo deseen pueden hacerlo de otras maneras.
#### Generador de descriptores
Conviene comenzar definiendo un diccionario de etiquetas, usualmente una tabla numerada de 1 en adelante (la etiqueta), con su descripción (la forma), por ejemplo:
1. cuadrado
2. triángulo
3. estrella

En adelante, las tablas y el código manejarán el número de la etiqueta en lugar de su descripción.
El dataset a construir consiste en una serie de muestras, cada una con una etiqueta.  En este caso las muestras son los 7 valores numéricos de los invariantes de Hu correspondientes a un contorno, y la etiqueta es un valor entero asignado a esa forma.  Los invariantes de Hu de una muestra tienen esta forma:
	
	[  6.53608067e-04,   6.07480284e-16,  9.67218398e-18, 1.40311655e-19,
	-1.18450102e-37,   8.60883492e-28, -1.12639633e-37  ]
	
Un dataset de invariantes de Hu es un array de dos dimensiones, con una fila para cada contorno analizado y 7 columnas para sendos invariantes de Hu.  El ejemplo ilustra un dataset de dos muestras:

	[
	[ 6.53608067e-04,   6.07480284e-16,  9.67218398e-18, 1.40311655e-19,
	-1.18450102e-37,   8.60883492e-28, -1.12639633e-37  ],
	[ 6.53608067e-04,   6.07480284e-16,  9.67218398e-18, 1.40311655e-19,
	-1.18450102e-37,   8.60883492e-28, -1.12639633e-37  ]
	]
	
El generador de descriptores es una aplicación similar al proyecto 1, que en lugar de clasificar sólo detecta el contorno, y cuando el usuario pulsa la tecla espacio imprime en terminal sus invariantes de Hu.
Con esta aplicación se extraen invariantes de Hu de múltiples imágenes diferentes con la misma forma.  Por ejemplo se puede presentar un mismo cuadrado en diferentes posiciones, rotaciones y escalas, y se puede complementar con otros cuadrados ligeramente diferentes (por ejemplo dibujados a mano alzada).
Luego de evaluar todas las imágenes, se copian los invariantes de Hu desde la terminal y se pegan en una hoja de cálculo, y se le agrega una columna con la etiqueta correspondiente.  El procedimiento se repite para cada forma que se quiera distinguir.  Una buena idea de cara a la etapa siguiente es mezclar aleatoriamente las filas.
#### Entrenador
Esta aplicación entrena la máquina y genera el modelo para inferencia.  Opera exclusivamente sobre el dataset, que puede estar guardado en archivos con algún formato adecuado para ser consumido por el algoritmo de machine learning, o bien se puede pegar directamente en el código con forma de array de dos dimensiones.  
El código puede ser como sigue, requiere instalar scikit-learn:
	
	from sklearn import tree
	from joblib import dump, load
	
	# dataset
	X = [
	[6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
	[6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04],
	[6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
	[1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18],
	[8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37],
	[6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
	[9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16],
	]
	
	# etiquetas, correspondientes a las muestras
	Y = [1, 1, 1, 2, 2, 3, 3]
	
	# entrenamiento
	clasificador = tree.DecisionTreeClassifier().fit(X, Y)
	
	# visualización del árbol de decisión resultante
	tree.plot_tree(clasificador)
	
	# guarda el modelo en un archivo
	dump(clasificador, 'filename.joblib')
	
	# en otro programa, se puede cargar el modelo guardado
	clasificadorRecuperado = load('filename.joblib') 


La aplicación entrena el modelo, y al finalizar lo guarda en un archivo para que pueda ser utilizado para predicción por otra aplicación.
#### Clasificador
Esta aplicación es una versión mejorada del proyecto anterior.  Opera sobre la webcam.  Básicamente reemplaza la clasificación con matchShapes, por la predicción del modelo entrenado.
Al iniciar debe cargar el modelo desde su archivo, y luego, sobre cada contorno, computar su descriptor y alimentar el modelo para predicción:
	from joblib import load
	
	# carga el modelo
	clasificador = load('filename.joblib') 
	
	etiquetaPredicha = clasificador.predict(invariantesDeHu)



La interfaz de usuario puede ser idéntica a la del proyecto anterior.  La etiqueta predicha es un número entero, se puede usar el diccionario para obtener la descripción de texto correspondiente.


## Proyecto nº3: MediaPipe
Enunciado libre, utilizando MediaPipe.
* En este caso, se utiliza MediaPipe para el reconocimiento de lenguaje de señas, utilizando un dataset personalizado.
* A la vez, permite la escritura de palabras, a partir del punto anterior
## Proyecto nº4: ??

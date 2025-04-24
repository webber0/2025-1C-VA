# 2025-1C-VA
Repositorio de TPs de la materia Visión Artificial
### Grupo nº 5 - Integrantes
* Pompeo, Nicolas Ruben
* Ramos, Alan Gonzalo
* Báez, Matías Nahuel
## Proyecto nº1: Detección y Clasificación de Formas
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

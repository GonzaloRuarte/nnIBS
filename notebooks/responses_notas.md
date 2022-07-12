# Responses

Todo el repo parece estar indexado en términos de imagenes.

## Notas

- Parece que las fijaciones de respuesta estan en el tamaño de la pantalla y no de la imagen
- Las notaciones que se usaron fueron las correspondientes a imagenes y no matrices **X:columna:ancho** e **Y:fila:alto**
- Criterio para definir cuando la respuesta del sujeto efectivamente toca el target: círculo inscripto dentro del box del target, distancia de centros menor a la respuesta mas el radio del círculo inscripto (debería ser 72/2=36)
- **OJO**: corregí el error de matlab en la función que parsea el json a pandas: *get_responses_features*.
- **OJO**: tiramos los sujetos que respondieron

## Dudas

- El bbox esta como w,h o h,w? **Es h_0, w_0**, CHEQUEAR
- La respuesta del sujeto estaba en referencia al tamaño de la pantalla? **Todo indica que si**
- La respuesta del sujeto la daban siempre? **Si**
- El tamaño de la respuesta es en pixeles? **Si**
- Cuando grafico la respuesta, al ser una imagen se grafica con el eje (0,0) arriba a la izquierda en vez de abajo a la izquierda. Con los datos pasa lo mismo? Los valores de respuesta y los valores de las sacadas estan en el mismo sentido? Es decir que si y se toma el dato abajo pero se grafica arriba hay que hacer esto: y = y_height - y

## TODOs

- Chequear datos de sujetos, si los ids son los mismos
- Me quedaron cruzados los x e y de Matlab, corregir. **Ver nota arriba**
- ~~Agregar la fijación inicial forzada~~
- ~~Chequear las dimensiones, que es X y que es Y~~
- ~~Cambiar la decisión de como contabilizar si el sujeto vió o no el target a partir de la respuesta subjetiva~~
- Extender scanpath con la respuesta
- Fijarse que pasa con los sujetos que tiene respuesta que caen afuera de la imagen
- Armar un codiguito para ver los sujetos que lo encontraron al target pero en la respuesta parece que no
- Calcular área cubierta?

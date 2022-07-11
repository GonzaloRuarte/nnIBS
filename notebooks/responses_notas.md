# Responses

Todo el repo parece estar indexado en términos de imagenes.

## Notas

- Parece que las fijaciones de respuesta estan en el tamaño de la pantalla y no de la imagen
- Las notaciones que se usaron fueron las correspondientes a imagenes y no matrices **X:columna:ancho** e **Y:fila:alto**
- Criterio para definir cuando la respuesta del sujeto efectivamente toca el target: círculo inscripto dentro del box del target, distancia de centros menor a la respuesta mas el radio del círculo inscripto (debería ser 72/2=36)
- 

## Dudas

- El bbox esta como w,h o h,w? 
- La respuesta del sujeto estaba en referencia al tamaño de la pantalla? **Todo indica que si**
- La respuesta del sujeto la daban siempre? **Si**
- El tamaño de la respuesta es en pixeles? **Si**

## TODOs

- Chequear datos de sujetos, si los ids son los mismos
- ~~Chequear las dimensiones, que es X y que es Y~~
- ~~Cambiar la decisión de como contabilizar si el sujeto vió o no el target a partir de la respuesta subjetiva~~
- Extender scanpath con la respuesta
- Fijarse que pasa con los sujetos que tiene respuesta que caen afuera de la imagen
- Armar un codiguito para ver los sujetos que lo encontraron al target pero en la respuesta parece que no
- Calcular área cubierta?


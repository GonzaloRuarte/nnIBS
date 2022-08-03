# Responses

Todo el repo esta indexado en términos de imagenes.

## Notas

- **25/7** - Las distancias al target no parecen muy cercanas a ser normales ni con logaritmo ni con boxcox. Las distancias a la última fijación
- Parece que las fijaciones de respuesta estan en el tamaño de la pantalla y no de la imagen. Corregido en las respuestas
- Las notaciones que se usaron fueron las correspondientes a imagenes y no matrices **X:columna:ancho** e **Y:fila:alto**
- Criterio para definir cuando la respuesta del sujeto efectivamente toca el target: círculo inscripto dentro del box del target, distancia de centros menor a la respuesta mas el radio del círculo inscripto (debería ser 72/2=36)
- **OJO**: corregí el error de matlab en la función que parsea el json a pandas: *get_responses_features*.
- **OJO**: tiramos los sujetos que respondieron fuera de la imagen con el click, eran un total de 5 sujetos.
- Parece ser que las inclusiones de las fijaciones que caían fuera del borde hacen bastante ruido

## Dudas

- El bbox esta como w,h o h,w? **Es h_0, w_0**:

    ```y1, x1, y2, x2  = trial['target_bbox']```
    ```h1, w1, h2, w2  = trial['target_bbox']```

- La respuesta del sujeto estaba en referencia al tamaño de la pantalla? **Todo indica que si**
- La respuesta del sujeto la daban siempre? **Si**
- El tamaño de la respuesta es en pixeles? **Si**
- Cuando grafico la respuesta, al ser una imagen se grafica con el eje (0,0) arriba a la izquierda en vez de abajo a la izquierda. Con los datos pasa lo mismo? Los valores de respuesta y los valores de las sacadas estan en el mismo sentido? Es decir que si y se toma el dato abajo pero se grafica arriba hay que hacer esto: y = y_height - y.
- **CHEQUEAR** Hay algunos sujetos que parecen tener 4 max_fixations cuando los valores deberían ser 3, 5, 9 y 13.
- **CHEQUEAR** Duda, cuando pongo el `stat=percent´ y ´common_norm=False´ esta efectivamente normalizando por grupos?

## TODOs

- ~~Agregar la fijación inicial forzada~~
- ~~Chequear las dimensiones, que es X y que es Y~~
- ~~Cambiar la decisión de como contabilizar si el sujeto vió o no el target a partir de la respuesta subjetiva~~
- ~~Armar un codiguito para ver los sujetos que lo encontraron al target pero en la respuesta parece que no~~
- ~~Agregar una función para plotear la distribución de los targets~~
- ~~Me quedaron cruzados los x e y de Matlab, corregir. **Ver nota arriba**~~
- Chequear datos de sujetos, si los ids son los mismos
- Extender scanpath con la respuesta
- Fijarse que pasa con los sujetos que tiene respuesta que caen afuera de la imagen
- Agregar duración de la ultima fijación o de la fijación en target
- Calcular área cubierta?

## Notas charla 26/7

- Ver si podemos ajustar dos Gaussianas a distancia a ultima fijacion/ y distancia al target --> comp bimodal
- Distancia a ultima fijaciones podemos probar boxcox
- Mirar una distancia a la sacada promedio

## Semana 1/8

- Cuando ellos calculan la performance hacen --> add human performances --> acceden al atributo *subjects_cumulative_performance* que les guarda una lista en la cual quedan almacenados para cada threshold el rendiemiento de los sujetos (4*57 sujetos)
- Debería decidir que hacer con los sujetos que parecen tener un numero diferente de cantidad de fijaciones máximas pero el sujeto 29 (ISC) y el sujeto 37 (LM) tiene trials con 3 fijaciones máximas y por ende 4 fijaciones (por decisión se las pasamos a máximo 5 fijaciones ,4 *max_saccade*) **Decirle a Juan y Gonzalo**.
- Esos mismos sujetos no tuvieron trials con 13 fijaciones (12 sacadas) por lo cual para usar la función de Fermin y Gonzalo, les asigne a esos sujetos el mismo rendimiento para esos trials que el que tuvieron con 9 fijaciones (8 sacadas)
- No quedan igual los boxplots de performance, no importa lo que haga. **Mirar en detalle y discutir con los chicos** (deben usar otro criterio para decidir target found)
- Arme dos notebooks, una exploratoria y otra de modelos

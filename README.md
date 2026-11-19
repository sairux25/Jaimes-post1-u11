# Laboratorio Unidad 11 — Benchmark CUDA y Aceleradores de IA
Sara Jaimes Castañeda · Arquitectura de Computadores

Este repositorio contiene el post-contenido de la Unidad 11. La Parte 1 implementa en CUDA C la suma de vectores, la multiplicación de matrices con tiling en shared memory y una versión de la suma de vectores con streams y copia asíncrona, y compara los tiempos de CPU y GPU medidos con cudaEvent. La Parte 2 corresponde al análisis comparativo de aceleradores de IA, que se desarrolla en docs/analisis-comparativo.md.

Entorno utilizado

Windows 11 Home con WSL2 y Ubuntu 24.04.5 LTS. La GPU es una NVIDIA GeForce RTX 4050 Laptop con 6141 MiB de memoria, compute capability 8.9 y driver 610.78. Se usó CUDA Toolkit 12.8 con nvcc V12.8.93 y gcc 13.3.0.

Los programas se compilaron desde la raíz del repositorio con los siguientes comandos.

```
nvcc -O2 -arch=sm_89 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -DN=1048576 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -DN=4194304 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -o matMul src/matMul.cu
nvcc -O2 -arch=sm_89 -o vectorAddStream src/vectorAddStream.cu
```

# PARTE 1 — Benchmark CUDA CPU vs GPU

## Checkpoint 1 — Benchmark vectorAdd
Capturas: capturas/CP1_compilacion_vectorAdd.png y capturas/CP1_ejecucion_vectorAdd.png

La primera vez que se compiló solo con nvcc -O2, el kernel tardó 584.96 ms, un tiempo muy alto para sumar 16M elementos. Al no indicar la arquitectura, nvcc genera un código genérico y el driver lo traduce para la GPU que lo ejecuta justo en esa primera corrida, así que ese tiempo terminó dentro de la medición. Para evitarlo se agregó -arch=sm_89 al comando, que corresponde a la GPU de este equipo. Con ese cambio la CPU tardó 42.21 ms, el kernel 7.24 ms y la salida mostró Errores: 0.

## Checkpoint 2 — Multiplicación de matrices con tiling
Capturas: capturas/CP2_compilacion_matMul.png y capturas/CP2_ejecucion_matMul.png

Del programa de matMul solo estaba el kernel con tiling, así que se escribió una versión naive que lee todo desde memoria global, la multiplicación en CPU para comparar y un main que mide los dos kernels con cudaEvent en N=512 y N=1024. Cada kernel se lanza una vez antes de medir para que la carga inicial no se cuente en ninguno. Los resultados coinciden con la CPU, con un error máximo de 4.58e-05 en N=512 y de 9.16e-05 en N=1024, muy por debajo de 1e-3.

| N | Naive (ms) | Tiling (ms) | Speedup |
|---|---|---|---|
| 512 | 0.618 | 0.491 | 1.26x |
| 1024 | 4.252 | 3.363 | 1.26x |

## Checkpoint 3 — README de la Parte 1
Capturas: capturas/CP1_tamanos_vectorAdd.png y capturas/CP1_ejecucion_vectorAdd.png

Para llenar la tabla de vectorAdd hacían falta tres tamaños y el tiempo total con las copias, pero el programa solo medía el kernel con 16M. Por eso N quedó como un valor que se puede cambiar al compilar con -DN, y se agregaron dos eventos más que toman el tiempo desde la primera copia hacia la GPU hasta que el resultado vuelve al host. Con 1M y 4M también se obtuvo Errores: 0.

| N | CPU (ms) | GPU kernel (ms) | GPU total con cudaMemcpy (ms) |
|---|---|---|---|
| 1M (1048576) | 5.99 | 12.66 | 49.76 |
| 4M (4194304) | 8.58 | 3.37 | 28.49 |
| 16M (16777216) | 42.21 | 7.24 | 77.90 |

Con N grande el kernel le gana a la CPU porque cada elemento lo suma un thread distinto y miles de ellos trabajan a la vez, mientras la CPU recorre el arreglo uno por uno. Con 16M el kernel tardó 7.24 ms contra 42.21 ms de la CPU, casi 6 veces menos. Con 1M pasó lo contrario y el kernel marcó 12.66 ms, más que con 16M. Esto se debe a que la medición incluye el primer lanzamiento del kernel, que carga el módulo en la GPU, y con tan pocos datos ese costo fijo pesa más que la suma.

El tiempo total sí puede quedar por encima de la CPU, ya que antes del kernel hay que enviar A y B por el bus PCIe y al final traer C de vuelta, y en una suma esas copias duran mucho más que el cálculo. En este equipo pasó con los tres tamaños, por ejemplo con 1M el total fue 49.76 ms y la CPU solo necesitó 5.99 ms. Cuando N es pequeño la diferencia crece en proporción, porque el costo fijo de arrancar las copias y el kernel se reparte entre muy pocos elementos.

## Checkpoint 4 — Transferencia asíncrona con streams
Capturas: capturas/CP4_compilacion_vectorAddStream.png y capturas/CP4_ejecucion_vectorAddStream.png

Con nvcc -O2 la primera ejecución dio 596.52 ms para la versión síncrona y una reducción de 94.3 %, una cifra que no tenía sentido para un solapamiento real. Como la versión síncrona es la que se mide primero, se quedaba con el tiempo que tarda el driver en traducir el código para la GPU, y al compilar con -arch=sm_89, igual que en vectorAdd, ese tiempo desapareció. Además el programa no revisaba si los resultados eran correctos, entonces se le agregó la misma comparación contra la CPU de vectorAdd para las dos versiones. Entre una y otra se limpian h_C y d_C, para que los streams no aprovecharan el resultado anterior. La versión síncrona tardó 36.97 ms, la de 4 streams 32.87 ms y ninguna tuvo errores.

| Versión | Tiempo total (ms) | Errores |
|---|---|---|
| Síncrona | 36.97 | 0 |
| 4 streams | 32.87 | 0 |
| Reducción | 11.1 % | |

## Decisión Técnica — cudaMemcpyAsync con streams o transferencia síncrona para vectorAdd
Transferir los cerca de 192 MB de N de 16M a un ancho de banda efectivo de 10 a 12 GB/s toma entre 16 y 19 ms, un orden de magnitud más que los 1.8 ms de referencia del kernel. En este equipo el kernel tardó 7.24 ms y la versión síncrona completa 36.97 ms, es decir que unos 30 ms fueron solo copias y por eso vectorAdd es una operación memory-bound.

El ahorro máximo teórico de los streams está acotado por el tiempo de cómputo, porque el solapamiento solo esconde el kernel detrás de las copias y no elimina la transferencia. Con un kernel de 7.24 ms el límite era bajar unos 7 ms de los 36.97 ms, alrededor del 20 %, y la medición dio 4.10 ms menos, un 11.1 %.

Usar streams en vectorAdd aporta poco frente al trabajo de reservar memoria pinned y repartir los datos en chunks, por lo que la transferencia síncrona resulta suficiente. En cambio matMulTiled aprovecharía mucho más el solapamiento, ya que con N de 1024 mueve 12 MB, que a 10 GB/s son cerca de 1.2 ms, mientras su kernel tardó 3.363 ms. Ahí el cómputo alcanza a ocultar toda la copia.

## Decisión Técnica — Cuántos streams usar y qué limita la ganancia
Aumentar NSTREAMS de 4 a 16 no seguiría bajando el tiempo de forma proporcional, porque todos los streams comparten el mismo bus PCIe y los mismos motores de copia. Con 4 streams ya se logró 11.1 %, cerca del tope de unos 7 ms que pone el kernel, y más streams apenas podrían acercarse un poco a ese límite.

El número de copy engines define cuántas copias avanzan a la vez. Si la GPU tiene 1 o 2 motores DMA, con 2 a 4 streams ya hay siempre una copia en marcha mientras otro chunk se procesa. Los demás streams solo esperan su turno y además suman el costo de lanzar más copias y kernels pequeños.

Dividir en más streams no reduce el tiempo total de copia, solo cambia el momento en que se mueve cada parte. Los 192 MB pasan igual por el mismo bus, y por eso con 4 streams el programa siguió tardando 32.87 ms, casi todo en copias.

## Checkpoint 5 — Decisiones técnicas de la Parte 1
Capturas: capturas/CP1_ejecucion_vectorAdd.png y capturas/CP4_ejecucion_vectorAddStream.png

Las dos decisiones se apoyan en los tiempos medidos en este equipo. La ejecución de vectorAdd con 16M muestra el tiempo de kernel que limita el ahorro, y la de vectorAddStream deja ver que con 4 streams la reducción fue de 11.1 %, lejos de crecer al ritmo del número de streams.

# PARTE 2 — Análisis comparativo de aceleradores de IA

El análisis completo está en docs/analisis-comparativo.md. Las cifras de MLPerf se tomaron de los registros oficiales de la versión 4.1 publicados por MLCommons, y las especificaciones y precios de las páginas de NVIDIA, Google Cloud, AMD y Apple.

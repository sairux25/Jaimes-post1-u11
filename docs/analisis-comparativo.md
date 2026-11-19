# Análisis Comparativo de Aceleradores de Hardware para Inteligencia Artificial
Sara Jaimes Castañeda · Arquitectura de Computadores · Unidad 11

## Introducción

Las cargas de trabajo de inteligencia artificial se apoyan cada vez menos en la CPU y cada vez más en hardware especializado. Este documento compara las cuatro familias principales de aceleradores, que son la GPU, la TPU, la NPU y la FPGA, a partir de su arquitectura interna, de los resultados oficiales de MLPerf Inference v4.1, de su costo y del efecto de la cuantización INT8. Con esa base se recomienda un acelerador para tres perfiles de organización y se justifican dos decisiones técnicas sobre el Perfil B de inspección visual industrial.

Todas las cifras provienen de hojas de especificaciones de los fabricantes, de la documentación oficial de Google Cloud o de los registros publicados por MLCommons. Las cifras de MLPerf se tomaron directamente de los archivos mlperf_log_summary.txt del repositorio oficial de resultados v4.1, y la eficiencia energética se calculó con los registros de potencia de ese mismo repositorio.

## 1. Caracterización de cada acelerador

La GPU NVIDIA H100 sigue un modelo SIMT masivo. Sus unidades básicas son los CUDA Cores para cálculo general y los Tensor Cores para multiplicación de matrices. En la versión SXM ofrece 1979 TFLOPS en FP16 y 3958 TOPS en INT8 con dispersión estructurada, la mitad sin ella, junto con 80 GB de HBM3 a 3.35 TB/s y un TDP de hasta 700 W (NVIDIA, 2023). Se programa con CUDA, cuDNN y TensorRT, el ecosistema más maduro de los cuatro.

La TPU de Google es un ASIC basado en arreglos sistólicos. En la TPU v4 cada TensorCore tiene cuatro unidades MXU de 128×128, y cada chip alcanza 275 TFLOPS en BF16 con 32 GiB de HBM2 a 1200 GBps (Jouppi et al., 2023). La TPU v5e entrega 197 TFLOPS en BF16 y 393 TOPS en INT8 con 16 GB de HBM, mientras que la v6e (Trillium) llega a 918 TFLOPS en BF16 y 1836 TOPS en INT8 con 32 GB de HBM (Google Cloud, s.f.-a, s.f.-b). Se programa a través de XLA con JAX, TensorFlow o PyTorch/XLA y solo se puede usar como servicio en la nube.

La NPU de Apple es un motor de flujo de datos fijo integrado en el SoC. El M3 Ultra trae un Neural Engine de 32 núcleos que comparte una memoria unificada de más de 800 GB/s con la CPU y la GPU (Apple, 2025). Apple no publica la cifra de TOPS de este chip ni las precisiones internas del Neural Engine, por lo que su capacidad solo se puede describir de forma cualitativa. Se programa con Core ML, que decide qué partes del modelo corren en el Neural Engine.

La FPGA AMD Alveo U55C es lógica reconfigurable. Tiene 1,304K LUTs, 9,024 bloques DSP y 16 GB de HBM2 con 460 GB/s de ancho de banda (AMD, 2023). La tarjeta consume como máximo 150 W. Se programa con Vitis HLS en C++ o directamente en VHDL o Verilog, lo que exige mucho más trabajo de ingeniería que los otros tres.

| Acelerador | Paradigma | Unidad de cómputo | Rendimiento máximo | Memoria | Modelo de programación | Caso de uso primario |
|---|---|---|---|---|---|---|
| GPU NVIDIA H100 SXM | SIMT masivo | CUDA Cores y Tensor Cores | 1979 TFLOPS FP16 y 3958 TOPS INT8 con dispersión | 80 GB HBM3, 3.35 TB/s | CUDA, cuDNN, TensorRT | Entrenamiento DNN, HPC, inferencia de LLM |
| TPU v5e / v6e | Arreglo sistólico | MXU (128×128 en v4) | 197 / 918 TFLOPS BF16, 393 / 1836 TOPS INT8 | 16 / 32 GB HBM | XLA, JAX, TensorFlow | Entrenamiento e inferencia a escala en la nube |
| NPU Apple M3 Ultra | Flujo de datos fijo | Neural Engine de 32 núcleos | No publicado por Apple | Memoria unificada, más de 800 GB/s | Core ML | Inferencia en el dispositivo |
| FPGA AMD Alveo U55C | Reconfigurable | LUT, DSP y BRAM | Depende del diseño cargado | 16 GB HBM2, 460 GB/s | Vitis HLS, VHDL, Verilog | Inferencia de baja latencia determinística |

## 2. Benchmarks MLPerf Inference v4.1

MLPerf Inference mide el rendimiento con modelos y conjuntos de datos fijos y con una exactitud mínima obligatoria. La tabla reúne resultados de la división closed de datacenter de la versión 4.1 (MLCommons, 2024). En el escenario Offline se reporta el throughput en muestras por segundo, y en el escenario Server se reporta además el percentil 99 de latencia bajo una carga sostenida.

| Modelo | Sistema | Escenario | Resultado | Por acelerador |
|---|---|---|---|---|
| ResNet-50 v1.5 | Dell XE9680, 8× H100 SXM | Offline | 709,849 muestras/s | 88,731 muestras/s |
| ResNet-50 v1.5 | Dell R760xa, 6× UntetherAI speedAI240 Slim | Offline | 334,462 muestras/s | 55,744 muestras/s |
| ResNet-50 v1.5 | Intel, 2 sockets Xeon (Emerald Rapids) | Offline | 25,204.5 muestras/s | Solo CPU |
| ResNet-50 v1.5 | Dell XE9680, 8× H100 SXM | Server | 584,207 muestras/s, p99 de 10.39 ms | 73,026 muestras/s |
| BERT-Large (99 %) | Dell XE9680, 8× H100 SXM | Offline | 70,594 muestras/s | 8,824 muestras/s |
| BERT-Large (99 %) | Dell XE9680, 8× H100 SXM | Server | 56,012 muestras/s, p99 de 76.26 ms | 7,001 muestras/s |
| BERT-Large (99 %) | Intel, 2 sockets Xeon (Emerald Rapids) | Offline | 1,666.82 muestras/s | Solo CPU |
| Stable Diffusion XL | NVIDIA DGX, 8× H100 SXM | Offline | 16.35 muestras/s | 2.04 muestras/s |
| Stable Diffusion XL | Google, 4× TPU v6e | Offline | 5.44 muestras/s | 1.36 muestras/s |
| Stable Diffusion XL | Google, 4× TPU v5e | Offline | 1.75 muestras/s | 0.44 muestras/s |

Para la eficiencia energética solo se pueden comparar los sistemas que enviaron mediciones de potencia. Con los registros oficiales se calculó la potencia media del sistema completo durante la corrida válida.

| Modelo | Sistema | Throughput | Potencia media del sistema | Eficiencia |
|---|---|---|---|---|
| ResNet-50 v1.5 | 6× UntetherAI speedAI240 Slim | 334,462 muestras/s | 1,021.4 W | 327.5 muestras/s/W |
| ResNet-50 v1.5 | 8× H200 SXM (MaxQ) | 556,234 muestras/s | 4,954.0 W | 112.3 muestras/s/W |
| BERT-Large (99 %) | 8× H200 SXM (MaxQ) | 54,063 muestras/s | 5,531.2 W | 9.77 muestras/s/W |
| Stable Diffusion XL | 8× H200 SXM (MaxQ) | 13.20 muestras/s | 5,573.8 W | 0.0024 muestras/s/W |

En ResNet-50 la GPU lidera en throughput, porque sus Tensor Cores y los 3.35 TB/s de su HBM3 mantienen ocupadas miles de unidades con lotes grandes. En eficiencia, en cambio, lidera el acelerador de flujo de datos de UntetherAI con casi tres veces más muestras por vatio que el sistema H200, ya que coloca el cómputo junto a la memoria y evita gran parte del movimiento de datos, que es lo que más energía consume. En BERT-Large la GPU H100 también lidera en throughput, y en esta ronda NVIDIA fue el único remitente con medición de potencia para este modelo, así que no hay con quién comparar su eficiencia. En Stable Diffusion XL cada H100 genera 2.04 imágenes por segundo frente a 1.36 de cada TPU v6e, aunque la v6e mejora 3.1 veces a la v5e gracias a sus MXU más grandes y a su mayor ancho de banda de HBM. Ningún sistema con NPU de Apple ni con FPGA Alveo participó en la división closed de datacenter v4.1, por lo que para esas dos familias solo se cuenta con especificaciones del fabricante.

## 3. Análisis costo-rendimiento y TCO

El costo total de propiedad suma la adquisición, la energía y el esfuerzo de desarrollo durante tres años, que equivalen a 26,280 horas. Para la energía se usó el precio promedio comercial de Estados Unidos en 2024, de 12.75 centavos por kWh (U.S. Energy Information Administration [EIA], 2025), y se supuso uso continuo al TDP máximo, lo que da un límite superior.

| Opción | Adquisición o alquiler | Energía en 3 años | Total aproximado |
|---|---|---|---|
| GPU H100 propia | 30,000 USD (cifra de referencia del enunciado, en el rango de 25,000 a 30,000 USD reportado por Tom's Hardware, 2023) | 700 W × 26,280 h = 18,396 kWh → 2,345 USD | 32,345 USD sin servidor ni refrigeración |
| TPU v5e alquilada, 1 chip | 1.20 USD/h bajo demanda o 0.54 USD/h con compromiso de 3 años (Google Cloud, s.f.-c) | Incluida en el precio | 31,536 USD bajo demanda o 14,191 USD con compromiso |
| TPU v6e alquilada, 1 chip | 2.70 USD/h bajo demanda (Google Cloud, s.f.-c) | Incluida en el precio | 70,956 USD |
| FPGA Alveo U55C | 4,747 USD en la tienda de AMD (AMD, s.f.) | 150 W × 26,280 h = 3,942 kWh → 503 USD | 5,250 USD más el costo alto de diseño en HLS o RTL |
| Mac Studio M3 Ultra | 3,999 USD con 96 GB de memoria unificada (MacRumors, 2025) | No calculada, Apple no publica un TDP del chip | 3,999 USD más energía |

Para una startup con inferencia en la nube por debajo de 100 solicitudes por segundo, comprar una H100 no tiene sentido. Cada H100 del sistema de ocho procesó en promedio 73,026 imágenes por segundo de ResNet-50 en el escenario Server, así que una sola GPU estaría ociosa más del 99 % del tiempo. Alquilar una TPU v5e o una GPU en la nube convierte la inversión inicial en un gasto que crece con el uso, y el compromiso de 3 años baja el costo a menos de la mitad del precio bajo demanda. Por lo tanto, para este escenario conviene el alquiler en la nube.

Para la inferencia en el borde el costo dominante no es el chip sino la integración. En un smartphone o un portátil la NPU ya viene dentro del SoC, de modo que su costo marginal es casi cero y su consumo es mínimo. En una cámara industrial o un vehículo, donde la latencia debe ser fija, la FPGA tiene el menor costo energético de la tabla, aunque el desarrollo en HLS o RTL encarece el proyecto. Por eso se recomienda la NPU integrada para dispositivos de consumo y la FPGA para equipos industriales con requisitos de tiempo real.

## 4. Recomendación fundamentada por perfil

**Perfil A, laboratorio universitario con 10,000 USD para visión y NLP de hasta 7B parámetros.** Se recomienda un Mac Studio M3 Ultra de 3,999 USD y destinar el resto del presupuesto a horas de TPU v5e en la nube. Primero, el presupuesto no alcanza para una H100 de 30,000 USD. Segundo, un modelo de 7B parámetros en FP16 ocupa 7 × 10⁹ × 2 bytes, es decir 14 GB, que caben con holgura en los 96 GB de memoria unificada. Tercero, los 6,001 USD restantes pagan unas 5,000 horas de TPU v5e bajo demanda para los entrenamientos más pesados. La limitación es que el Neural Engine solo acelera inferencia a través de Core ML y el soporte de PyTorch en Apple es menos maduro que CUDA, de modo que el entrenamiento local será lento.

**Perfil B, manufactura con inspección visual en menos de 5 ms en 50 líneas.** Se recomienda una FPGA por línea. Primero, la lógica fija en hardware da una latencia máxima determinística, sin sistema operativo ni planificador. Segundo, cada línea queda aislada, así que una falla no detiene las demás. Tercero, la U55C consume como máximo 150 W, frente a 700 W de una H100. La limitación es el costo, que a 4,747 USD por tarjeta suma 237,350 USD para 50 líneas, y la dificultad de actualizar el modelo, porque cada cambio requiere volver a sintetizar el diseño. Las decisiones técnicas del final amplían este perfil.

**Perfil C, proveedor de nube que entrena modelos de 70B en clústeres de 512 aceleradores.** Se recomienda la GPU H100 SXM, con la TPU v5p como alternativa. Primero, los 140 GB de pesos en FP16 de un modelo de 70B no caben en una sola tarjeta, así que el entrenamiento depende de la comunicación colectiva, y la H100 ofrece NVLink de 900 GB/s dentro del nodo (NVIDIA, 2023). Segundo, la H100 lidera el throughput en los tres modelos de MLPerf de la sección 2. Tercero, CUDA tiene el mayor soporte de bibliotecas de entrenamiento distribuido. La limitación es el consumo, ya que 512 GPU a 700 W suman 358.4 kW solo en aceleradores. La TPU v4 demostró escalar a 4,096 chips con interconexión óptica reconfigurable (Jouppi et al., 2023), pero obliga a quedarse en Google Cloud.

## 5. Cuantización INT8 frente a FP32

Reducir la precisión de FP32 o FP16 a INT8 permite que las unidades matriciales procesen el doble de operaciones por ciclo. Este patrón se repite en varias generaciones de hardware, como se ve en las hojas de especificaciones oficiales.

| Acelerador | FP16 / BF16 | INT8 | Relación |
|---|---|---|---|
| NVIDIA T4 | 65 TFLOPS | 130 TOPS | 2× (NVIDIA, 2019) |
| NVIDIA A100 | 312 TFLOPS | 624 TOPS | 2× (NVIDIA, 2020) |
| NVIDIA H100 SXM | 1979 TFLOPS | 3958 TOPS | 2×, ambas con dispersión (NVIDIA, 2023) |
| Google TPU v5e | 197 TFLOPS | 393 TOPS | 2× (Google Cloud, s.f.-a) |
| Google TPU v6e | 918 TFLOPS | 1836 TOPS | 2× (Google Cloud, s.f.-b) |

La ganancia en exactitud tiene un costo, que MLPerf controla con un límite estricto. La siguiente tabla usa los modelos de referencia publicados en el repositorio oficial de MLCommons (MLCommons, s.f.).

| Modelo | Precisión | Exactitud reportada | Restricción MLPerf |
|---|---|---|---|
| ResNet-50 v1.5 | FP32 (referencia) | 76.456 % top-1 | Referencia |
| ResNet-50 v1.5 | INT8 calibrado | 75.790 % top-1 | ≥ 99 % de la referencia, es decir ≥ 75.69 % |
| BERT-Large | FP32 (referencia) | F1 de 90.874 | Referencia |
| BERT-Large | INT8 con ajuste fino (versión PyTorch) | F1 de 90.633 | ≥ 99 % (89.965) en bert-99 y ≥ 99.9 % (90.783) en bert-99.9 |
| BERT-Large | INT8 simétrico por tensor con ajuste fino (versión ONNX) | F1 de 90.067 | Cumple bert-99, no cumple bert-99.9 |

La versión INT8 de ResNet-50 pierde 0.67 puntos y cumple el límite del 99 %. En BERT-Large los dos modelos INT8 cumplen el 99 % pero ninguno alcanza el 99.9 %, lo que muestra que los modelos de lenguaje son más sensibles a la cuantización. MLPerf exige además que esta exactitud se mida sobre el conjunto de validación completo y no sobre una muestra.

Cuantizar no es un simple cambio de tipo de dato. El rango de valores de FP32 no se reparte de forma uniforme sobre los 256 niveles de INT8, así que un cast directo pierde mucha información. Wu et al. (2020) muestran que con calibración posterior al entrenamiento, que ajusta factores de escala por capa usando datos representativos, o con entrenamiento consciente de la cuantización, la exactitud se mantiene dentro del 1 % de la versión en punto flotante en todas las redes que estudiaron, incluidas MobileNet y BERT-Large.

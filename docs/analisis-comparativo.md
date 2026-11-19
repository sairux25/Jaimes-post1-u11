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

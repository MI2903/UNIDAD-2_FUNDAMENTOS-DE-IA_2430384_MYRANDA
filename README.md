# Visualizador de rutas en grafos

Aplicación de escritorio en Python con PyQt6, QDarkStyle, NetworkX y Matplotlib. Contiene dos versiones del grafo de las fotografías, permite escribir origen y destino, dibuja la ruta y muestra las matrices de adyacencia e incidencia.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
python grafo_visual.py
```

Escribe el número de los nodos de inicio y final y pulsa **Calcular rutas**. El menú **Ruta destacada** selecciona cuál de los resultados se pinta en el grafo. Puedes cambiar entre tema **Claro** y **Oscuro**. Los botones sobre el grafo permiten usar el zoom, mover, restablecer la vista y guardar una imagen PNG o SVG. La explicación del tipo de grafo aparece en la barra de estado.

La ventana abre en **Negativos**, con inicio **0**, final **0**, ruta seleccionada **Bellman-Ford** y tema **Claro**. Al abrir no calcula ni destaca ningún camino; el cálculo empieza al pulsar **Calcular rutas**. El dibujo usa aristas y flechas gruesas, y distribuye automáticamente los pesos cerca de sus aristas para separarlos de otros pesos y nodos. Al cambiar el tamaño de la ventana, las etiquetas se redistribuyen.

El selector **Pesos** muestra inicialmente solo los números de la ruta destacada para que el grafo se lea con facilidad; selecciona **Todos** para ver los pesos de todas las aristas. **Ampliar grafo** da más espacio al dibujo y activa **Todos**. Pulsa **Volver a la interfaz** o `Esc` para regresar a los controles y resultados; se restablece la selección de pesos anterior.

Al cambiar al grafo **Positivos** se selecciona **Dijkstra** y se muestran los pesos de todas las aristas. Al pulsar **Calcular rutas**, se anima la exploración: cada paso muestra el nodo cuyo menor costo acaba de confirmarse y el camino hasta él. Después se dibuja la ruta final tramo a tramo. **Animar Dijkstra** permite repetirla. En el grafo **Negativos** se selecciona **Bellman-Ford** y la animación queda desactivada porque Dijkstra no admite pesos negativos.

## Capturas automáticas de Dijkstra

Ejecuta `python generar_capturas_dijkstra.py` para crear las dos series solicitadas: **24 → 8** en `capturas_nodo_inicial_24` y **8 → 24** en `capturas_nodo_inicial_8`. Se captura el inicio, cada nodo confirmado y **cada arista examinada**, tanto si mejora la distancia como si se descarta. La última imagen se llama `ruta_definitiva_ORIGEN_a_DESTINO.png`. Todas las imágenes muestran el peso de **todas** las aristas y una tabla con la distancia provisional y el antecesor de cada nodo, como en el ejemplo de clase. `resumen.txt` registra todas las decisiones y el costo final. Para otros nodos usa `python generar_capturas_dijkstra.py ORIGEN DESTINO`. La generación se hace en el grafo positivo y conserva las posiciones del visualizador; las aristas tienen un contorno blanco para distinguir sus cruces.

## Capturas automáticas de Bellman-Ford

Ejecuta `python generar_capturas_bellman.py` para crear **8 → 24** en `capturas_nodo_inicial_8_bellman` y **24 → 8** en `capturas_nodo_inicial_24_bellman`. Las series incluyen la inicialización, cada relajación de arco en cada pasada (también las que no mejoran la distancia), un resumen al final de cada pasada, la verificación de ciclos negativos y la imagen final `ruta_definitiva_ORIGEN_a_DESTINO.png`. Cada captura conserva todos los pesos y muestra las distancias y los antecesores actuales. `indice.html` permite recorrer las imágenes con las flechas del teclado y `resumen.txt` registra las decisiones. Para otros nodos usa `python generar_capturas_bellman.py ORIGEN DESTINO`.

Las dos series se generan sobre el grafo positivo no dirigido; Bellman-Ford evalúa cada arista en ambos sentidos. En el grafo negativo dirigido no hay camino de 8 a 24, por lo que no podría obtenerse una ruta definitiva para ese par.

## Convenciones

- **Positivos:** grafo no dirigido. Los algoritmos Dijkstra, Bellman-Ford y Floyd-Warshall están disponibles.
- **Negativos:** grafo dirigido de izquierda a derecha; las conexiones verticales van de arriba abajo. Esto evita que cada arista negativa genere por sí sola un ciclo negativo al poder recorrerla de ida y vuelta. Dijkstra no es válido con pesos negativos, por lo que se muestra como no disponible. Bellman-Ford y Floyd-Warshall sí se calculan.
- La matriz de adyacencia muestra el menor peso si hay dos aristas entre el mismo par de nodos. `∞` significa que no hay conexión. La diagonal vale 0.
- La matriz de incidencia tiene una columna por arista, incluidas las paralelas. En positivos, `1` marca ambos extremos. En negativos, `-1` indica el origen y `1` el destino. El peso aparece en el encabezado de cada columna.

Los pesos fueron transcritos de fotografías, donde algunas etiquetas se superponen o no son completamente legibles. Se concentran en `ARISTAS_POSITIVAS` y `CAMBIOS_NEGATIVOS` dentro de `grafo_visual.py` para que puedas ajustarlos al enunciado original. En particular, revisa las conexiones `10–1`, `12–16` y `66–13`.

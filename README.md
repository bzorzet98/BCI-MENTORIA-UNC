# Mentoría BCI: EEG e Imaginación Motora para la Rehabilitación Post-Stroke

¡Bienvenidos al repositorio central de la mentoría! Este entorno contiene el código base para el desarrollo de los pipelines de procesamiento de señales y modelos de IA.

---
## Opción 1: Configuración en la Nube (Google Colaboratory + Google Drive)

Si preferís trabajar desde la nube sin instalar nada en tu computadora, podés clonar este repositorio directamente en tu Google Drive usando Colab:

1. Entrá a [Google Colab](https://colab.research.google.com/).
2. Creá una nueva notebook en blanco o subí/abrí el archivo `tutorial_inicial.ipynb`.
3. Ejecutá el bloque de código de inicialización (Celda 1). Este bloque te pedirá permiso para conectar tu Google Drive y **clonará automáticamente el repositorio de GitHub** en tu unidad (`Mi Unidad/bci-mentoria-unc`).
4. **Subir los Datos:** Lo único que deberás hacer a mano es entrar a tu Google Drive desde el navegador, ir a la carpeta creada `bci-mentoria-unc/data/` y arrastrar allí los archivos `.npy` de las bases de datos (`Cho2017` o `Dreyer2023`) ya preprocesadas.

## Opción 2: Configuración Local (Trabajar en tu PC)

Si vas a clonar el repositorio y trabajar de forma local en tu computadora, seguí estos pasos según tu sistema operativo:

### 1. Clonar el repositorio y acceder
Abrí tu terminal (Linux/Mac) o la Git Bash / CMD (Windows) y ejecutá:
```bash
git clone [https://github.com/tu-usuario/bci-stroke-rehab.git](https://github.com/tu-usuario/bci-stroke-rehab.git)
cd bci-stroke-rehab
```


###  2. Crear y activar un entorno virtual (Recomendado)
 * En Windows (CMD o PowerShell):

```bash
python -m venv bci_mentoria_env
.\bci_mentoria_env\Scripts\activate
```
 * En Linux / macOS:

```bash
python3 -m venv bci_mentoria_env
source bci_mentoria_env/bin/activate
```
###  3. Instalación de librerías obligatorias
Una vez activado el entorno, instalá todas las dependencias requeridas ejecutando:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

(Nota: El archivo instalará numpy, pandas, scipy, matplotlib, seaborn, scikit-learn y mne).

4. Descarga de datos
Crea las carpetas correspondientes dentro de data/ y coloca allí los archivos .npy provistos por el mentor:



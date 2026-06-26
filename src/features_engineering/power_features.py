import numpy as np
from scipy.signal import welch

class SignalProcessor:
    """
    Clase base para el procesamiento básico de señales de EEG y 
    extracción inicial de características.
    """
    
    @staticmethod
    def calcular_potencia_total(matrix_eeg):
        """
        Recibe una matriz de numpy de forma [N, C, T] (Ensayos, Canales, Tiempo)
        o un ensayo individual de forma [C, T].
        Calcula la potencia total promedio (media de los cuadrados de la señal) 
        a lo largo del eje del tiempo (último eje).
        """
        # Eleva al cuadrado y calcula la media sobre el eje del tiempo (axis=-1)
        potencia = np.mean(np.square(matrix_eeg), axis=-1)
        return potencia

    @staticmethod
    def calcular_psd(matrix_eeg, sfreq):
        """
        Calcula la Densidad Espectral de Potencia (PSD) usando el método de Welch.
        Recibe:
            matrix_eeg: array de numpy con la señal (última dimensión debe ser el Tiempo).
            sfreq: Frecuencia de muestreo de la señal (Hz).
        Retorna:
            frecuencias: Array con los bins de frecuencias calculados.
            psd: Array con los valores de potencia espectral.
        """
        # Calculamos la PSD a lo largo del eje del tiempo (axis=-1)
        frecuencias, psd = welch(matrix_eeg, fs=sfreq, axis=-1, nperseg=int(sfreq*2))
        return frecuencias, psd
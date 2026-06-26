import os
import pandas as pd
import mne
from global_config import DATABASES_PATH
import numpy as np
from .base import BaseEEGDataset
from .subject import Subject

"""HAY QUE UNIFICAR CON LA NUEVA LOGICA DE LOS OBJETOS SUBJECTS. EN ESTE CASO DEBEMOS DE REDEFINIR LOS METODOS DE ABSTRACT """


class Dreyer2023A(BaseEEGDataset):
    """
    Motor Imagery dataset A from Dreyer et al. 2023.

    This dataset contains EEG, EMG, and EOG recordings from 60 subjects performing 
    Left/Right hand Motor Imagery (MI) tasks under both calibration (sham feedback) 
    and online (real feedback) conditions.

    References
    ----------
    .. [1] Dreyer, P., Roc, A., Pillette, L. et al. A large EEG database with users'
        profile information for motor imagery brain-computer interface research. 
        Sci Data 10, 580 (2023). https://doi.org/10.1038/s41597-023-02445-z

    Data Structure & Channels
    -------------------------
    *   **Sampling Rate:** 512 Hz
    *   **EEG (27 Channels):** Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, 
        FC4, FC6, CP2, CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3.
    *   **Reference:** Left earlobe.
    *   **Ground:** FPz.
    *   **EMG (2 Channels):** EMGl, EMGr (Wrist extensors/flexors).
    *   **EOG (3 Channels):** Right eye (Vertical: above/below, Horizontal: side).

    Experimental Protocol
    ---------------------
    The experiment consists of a single session divided into resting state baselines, 
    calibration runs (sham feedback), and user training runs (real feedback).

    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | Context                  | Run Type              | Trials / Duration      | Description                               |
    +==========================+=======================+========================+===========================================+
    | **Resting State**        | Eyes Closed (CE)      | 3 minutes              | Baseline recording, subject relaxed.      |
    | (rest)                   | Eyes Open (OE)        | 3 minutes              | Baseline recording, subject relaxed.      |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Acquisition**       | Calibration           | 2 Runs                 | System calibration phase.                 |
    | (motor_imagery_          | (Sham Feedback)       | 40 trials/run          | Feedback is simulated/sham.               |
    | acquisition)             |                       | (20 Left / 20 Right)   |                                           |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Online**            | User Training         | 4 Runs                 | Real-time BCI usage.                      |
    | (motor_imagery_online)   | (Real Feedback)       | 40 trials/run          | Positive feedback (blue bar) based on     |
    |                          |                       | (20 Left / 20 Right)   | classifier confidence.                    |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+

    Trial Structure (MI Tasks)
    --------------------------
    Total trial duration is 8 seconds + inter-trial interval.

    +-------------------+-------------------+---------------------------------------------------------------+
    | Time (s)          | Event             | Description                                                   |
    +===================+===================+===============================================================+
    | **0.0 - 2.0 s**   | Fixation          | Green cross appears on screen.                                |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **2.0 s**         | Audio Cue         | Acoustic signal announces upcoming task.                      |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **3.0 - 4.25 s**  | Visual Cue        | Red arrow points Left or Right (Duration: 1.25s).             |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **4.25 - 8.0 s**  | Feedback          | Continuous feedback (blue bar) provided for 3.75s.            |
    |                   |                   | *Acquisition:* Sham feedback. *Online:* Real positive feedback.|
    +-------------------+-------------------+---------------------------------------------------------------+
    | **8.0 s +**       | Inter-trial       | Black screen for 1.5s to 3.5s (Randomized).                   |
    +-------------------+-------------------+---------------------------------------------------------------+
    """
    def __init__(self, sessions = ["session_1"], subjects=None, data_to_load = None):
        # 1. Define specific configuration for Cho
        path = os.path.join(DATABASES_PATH, 'Dreyer2023', 'Signals', 
                            'DATA A')
        subjects = list(range(1, 61)) if subjects is None else subjects
        event_id = {'init_run': 32769, 
                    'end_run': 32770,
                    'start_trial':768,
                    'cross_aparition':786,
                    'acoustic_signal': 33282,
                    'left_hand': 769,
                    'right_hand': 770,
                    'start_feedback': 781,
                    'end_trial':800}

        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, event_id=event_id, code='Dreyer2023A')
        
        # 3. Database-specific attributes
        self.sfreq = 512
        self.standard_montage = "standard_1020"
        self.ch_names = {"eeg":['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 
                                'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 
                                'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'],
                        "eog":["EOG1","EOG2", "EOG3"],
                        "emg":["EMGl", "EMGr"]}
        
        self.eeg_ch = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 
                'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 
                'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3']
        
        self.eog_ch = ["EOG1","EOG2", "EOG3"]
        self.emg_ch = ["EMGl", "EMGr"]
        self.ch_names = self.eeg_ch + self.eog_ch + self.emg_ch + ['stim']
        self.ch_types = ["eeg"] * len(self.eeg_ch) + ["eog"] * len(self.eog_ch) + ["emg"] * len(self.emg_ch) + ['stim']
        
        self.sessions = sessions if sessions is not None else ["session_1"]
        self.data_availables = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest', 'motor_imagery']
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions_available = ['session_1']
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata(path = os.path.join(DATABASES_PATH, 'Dreyer2023',))
    
    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load
        
    def _load_subjects_metadata(self, path=None):
        # Check if the file exists
        if os.path.exists(os.path.join(path, 'database_information_A.csv')):
            try:
                subjects_metadata = pd.read_csv(os.path.join(path, 'database_information_A.csv'))
            except Exception as e:
                print(f"Error loading subjects metadata: {e}")
                subjects_metadata = None
        else:
            subjects_metadata = None
        return subjects_metadata

    def _create_raw_simple(self, file_path):
        # 1. Read the gdf file
        raw =  mne.io.read_raw_gdf(file_path, eog=None, misc=None, 
                                    stim_channel='auto', exclude=(), 
                                    include=None, preload=True, verbose=None)

        # 2. Create the stim channel
        events_original, events_id_map = mne.events_from_annotations(raw)
        id_to_original_code = {v: int(k) for k, v in events_id_map.items() if k.isdigit()}
        stim_data = np.zeros((1, len(raw)))
        samples = events_original[:, 0]
        
        internal_ids = events_original[:, 2]
        original_codes = [id_to_original_code.get(idx, 0) for idx in internal_ids]
        # This is only to avoid possibles issues:
        corrected_samples = np.clip(samples, 0, len(raw) - 1)
        stim_data[0, corrected_samples] = original_codes

        info_stim = mne.create_info(
            ch_names=["stim"], 
            sfreq=raw.info['sfreq'], 
            ch_types=["stim"]
        )

        # Importante mantener first_samp para evitar desfases
        stim_raw = mne.io.RawArray(stim_data, info_stim, first_samp=raw.first_samp)
        raw.add_channels([stim_raw], force_update_info=True)
        
        # 3. Rename EMG channels
        mapping_names = {'EMGg': 'EMGl', 'EMGd': 'EMGr'}
        raw.rename_channels(mapping_names)
        
        # 4. Set types of channels
        mapping_types = {ch: 'eeg' for ch in self.eeg_ch}
        mapping_types.update({ch: 'eog' for ch in self.eog_ch})
        mapping_types.update({ch: 'emg' for ch in self.emg_ch})
        mapping_types['stim'] = 'stim'
        raw.set_channel_types(mapping_types)
        
        # 5. Reorder channels
        raw.reorder_channels(self.ch_names)
        
        # 6. Set the system
        raw.set_montage(self.standard_montage)
        return raw

    def _obtain_extra_info(self, subject_id=None):
        extra_info = {}
        extra_info['n_imagery_trials_per_run'] = 40
        
        # Añadir info del CSV (Age, Gender, etc.)
        if self.subjects_metadata is not None:
            # Buscamos la fila del sujeto (asumiendo que hay una columna 'subject' o similar)
            sub_row = self.subjects_metadata[self.subjects_metadata['subject_id'] == subject_id]
            if not sub_row.empty:
                extra_info['personal_metadata'] = sub_row.to_dict(orient='records')[0]
        return extra_info
    
    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject: 
        subject_path = os.path.join(self.dataset_path, f"A{subject_id}")
        if not os.path.exists(subject_path):
            print(f"Warning: File not found for subject {subject_id}")
            return None
        subject_data = {}


        for data in self.data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"A{subject_id}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"A{subject_id}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
            elif data == 'motor_imagery':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"A{subject_id}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"A{subject_id}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"A{subject_id}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info(subject_id=subject_id)
        return Subject(subject_id=subject_id, subject_dict=subject_data)
    
    # METODOS VIEJOS
        
    def get_data(self, subjects=None, data_to_load=None, sessions=None):
        data_to_load = data_to_load if data_to_load is not None else self.data_to_load
        
        # Si el usuario pide sesiones específicas en esta función, sobreescribimos temporalmente
        # las sesiones configuradas por defecto en el objeto.
        current_sessions = sessions if sessions is not None else getattr(self, 'sessions', ["session_1"])
        subjects, data_to_load = self._validate_inputs(subjects, data_to_load)
        data_dict = {"session_1":{}} 
        for subject in subjects:
            if subject not in self.subject_list:
                print(f"Sujeto {subject} no está en la lista válida. Omitiendo...")
                continue
            data_dict["session_1"][subject] = self._get_single_subject_data(subject, data_to_load, current_sessions)
                
        return data_dict

    def _get_single_subject_data(self, subject, data_to_load, sessions):
        subject_data = {}
        subject_path = os.path.join(self.dataset_path, f'A{subject}')


        for data in data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"A{subject}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"A{subject}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"A{subject}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info()
        return subject_data    
        
class Dreyer2023B(BaseEEGDataset):
    """
    Motor Imagery dataset B from Dreyer et al. 2023.

    This dataset contains EEG, EMG, and EOG recordings from 21 subjects performing 
    Left/Right hand Motor Imagery (MI) tasks under both calibration (sham feedback) 
    and online (real feedback) conditions.

    References
    ----------
    .. [1] Dreyer, P., Roc, A., Pillette, L. et al. A large EEG database with users'
        profile information for motor imagery brain-computer interface research. 
        Sci Data 10, 580 (2023). https://doi.org/10.1038/s41597-023-02445-z

    Data Structure & Channels
    -------------------------
    *   **Sampling Rate:** 512 Hz
    *   **EEG (27 Channels):** Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, 
        FC4, FC6, CP2, CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3.
    *   **Reference:** Left earlobe.
    *   **Ground:** FPz.
    *   **EMG (2 Channels):** EMGl, EMGr (Wrist extensors/flexors).
    *   **EOG (3 Channels):** Right eye (Vertical: above/below, Horizontal: side).

    Experimental Protocol
    ---------------------
    The experiment consists of a single session divided into resting state baselines, 
    calibration runs (sham feedback), and user training runs (real feedback).

    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | Context                  | Run Type              | Trials / Duration      | Description                               |
    +==========================+=======================+========================+===========================================+
    | **Resting State**        | Eyes Closed (CE)      | 3 minutes              | Baseline recording, subject relaxed.      |
    | (rest)                   | Eyes Open (OE)        | 3 minutes              | Baseline recording, subject relaxed.      |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Acquisition**       | Calibration           | 2 Runs                 | System calibration phase.                 |
    | (motor_imagery_          | (Sham Feedback)       | 40 trials/run          | Feedback is simulated/sham.               |
    | acquisition)             |                       | (20 Left / 20 Right)   |                                           |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Online**            | User Training         | 4 Runs                 | Real-time BCI usage.                      |
    | (motor_imagery_online)   | (Real Feedback)       | 40 trials/run          | Positive feedback (blue bar) based on     |
    |                          |                       | (20 Left / 20 Right)   | classifier confidence.                    |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+

    Trial Structure (MI Tasks)
    --------------------------
    Total trial duration is 8 seconds + inter-trial interval.

    +-------------------+-------------------+---------------------------------------------------------------+
    | Time (s)          | Event             | Description                                                   |
    +===================+===================+===============================================================+
    | **0.0 - 2.0 s**   | Fixation          | Green cross appears on screen.                                |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **2.0 s**         | Audio Cue         | Acoustic signal announces upcoming task.                      |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **3.0 - 4.25 s**  | Visual Cue        | Red arrow points Left or Right (Duration: 1.25s).             |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **4.25 - 8.0 s**  | Feedback          | Continuous feedback (blue bar) provided for 3.75s.            |
    |                   |                   | *Acquisition:* Sham feedback. *Online:* Real positive feedback.|
    +-------------------+-------------------+---------------------------------------------------------------+
    | **8.0 s +**       | Inter-trial       | Black screen for 1.5s to 3.5s (Randomized).                   |
    +-------------------+-------------------+---------------------------------------------------------------+
    """
    def __init__(self, data_to_load = None):
        # 1. Define specific configuration for Cho
        path = os.path.join(DATABASES_PATH, 'Dreyer2023', 'Signals', 
                            'DATA B')
        subjects = list(range(61, 81))
        event_id = {'init_run': 32769, 
                    'end_run': 32770,
                    'start_trial':768,
                    'cross_aparition':786,
                    'acoustic_signal': 33282,
                    'left_hand': 769,
                    'right_hand': 770,
                    'start_feedback': 781,
                    'end_trial':800}

        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, event_id=event_id, code='Dreyer2023B')
        
        # 3. Database-specific attributes
        self.sfreq = 512
        self.standard_montage = "standard_1020"
        self.ch_names = {"eeg":['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 
                                'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 
                                'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'],
                        "eog":["EOG1","EOG2", "EOG3"],
                        "emg":["EMGl", "EMGr"]}
        self.data_availables = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest']
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions_available = ['session_1']
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata(path = os.path.join(DATABASES_PATH, 'Dreyer2023'))

    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load
        
    def _load_subjects_metadata(self, path=None):
        # Check if the file exists
        if os.path.exists(os.path.join(path, 'database_information_B.csv')):
            try:
                subjects_metadata = pd.read_csv(os.path.join(path, 'database_information_B.csv'))
            except Exception as e:
                print(f"Error loading subjects metadata: {e}")
                subjects_metadata = None
        else:
            subjects_metadata = None
        return subjects_metadata
    

    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject: 
        subject_path = os.path.join(self.dataset_path, f's{subject_id:02d}.mat')
        if not os.path.exists(subject_path):
            print(f"Warning: File not found for subject {subject_id}")
            return None
        subject_data = {}


        for data in self.data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"B{subject_id}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"B{subject_id}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"B{subject_id}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info()
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    def _create_raw_simple(self, file_path):
        raw =  mne.io.read_raw_gdf(file_path, eog=None, misc=None, 
                                    stim_channel='auto', exclude=(), 
                                    include=None, preload=False, verbose=None)
        mapping_types = {'EMGg': 'emg', 'EMGd': 'emg', 'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog'}
        raw.set_channel_types(mapping_types)
        # 2. Rename them to match the paper's English terminology
        mapping_names = {'EMGg': 'EMGl', 'EMGd': 'EMGr'}
        raw.rename_channels(mapping_names)
        # 3. Set the system
        raw.set_montage(self.standard_montage)
        return raw

    def _obtain_extra_info(self):
        extra_info = {}
        extra_info['n_imagery_trials_per_run'] = 40
        return extra_info
    
    # Metodos VIEJOS
    
    
    def get_data(self, subjects=None, data_to_load=None, sessions=None):
        data_to_load = data_to_load if data_to_load is not None else self.data_to_load
        
        # Si el usuario pide sesiones específicas en esta función, sobreescribimos temporalmente
        # las sesiones configuradas por defecto en el objeto.
        current_sessions = sessions if sessions is not None else getattr(self, 'sessions', ["session_1"])
        subjects, data_to_load = self._validate_inputs(subjects, data_to_load)
        data_dict = {"session_1":{}} 
        for subject in subjects:
            if subject not in self.subject_list:
                print(f"Sujeto {subject} no está en la lista válida. Omitiendo...")
                continue
            data_dict["session_1"][subject] = self._get_single_subject_data(subject, data_to_load, current_sessions)
                
        return data_dict

    def _get_single_subject_data(self, subject, data_to_load, sessions):
        subject_data = {}
        subject_path = os.path.join(self.dataset_path, f'B{subject}')


        for data in data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"B{subject}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"B{subject}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"B{subject}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info()
        return subject_data

class Dreyer2023C(BaseEEGDataset):
    """
    Motor Imagery dataset c from Dreyer et al. 2023.

    This dataset contains EEG, EMG, and EOG recordings from 6 subjects performing 
    Left/Right hand Motor Imagery (MI) tasks under both calibration (sham feedback) 
    and online (real feedback) conditions.

    References
    ----------
    .. [1] Dreyer, P., Roc, A., Pillette, L. et al. A large EEG database with users'
        profile information for motor imagery brain-computer interface research. 
        Sci Data 10, 580 (2023). https://doi.org/10.1038/s41597-023-02445-z

    Data Structure & Channels
    -------------------------
    *   **Sampling Rate:** 512 Hz
    *   **EEG (27 Channels):** Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, 
        FC4, FC6, CP2, CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3.
    *   **Reference:** Left earlobe.
    *   **Ground:** FPz.
    *   **EMG (2 Channels):** EMGl, EMGr (Wrist extensors/flexors).
    *   **EOG (3 Channels):** Right eye (Vertical: above/below, Horizontal: side).

    Experimental Protocol
    ---------------------
    The experiment consists of a single session divided into resting state baselines, 
    calibration runs (sham feedback), and user training runs (real feedback).

    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | Context                  | Run Type              | Trials / Duration      | Description                               |
    +==========================+=======================+========================+===========================================+
    | **Resting State**        | Eyes Closed (CE)      | 3 minutes              | Baseline recording, subject relaxed.      |
    | (rest)                   | Eyes Open (OE)        | 3 minutes              | Baseline recording, subject relaxed.      |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Acquisition**       | Calibration           | 2 Runs                 | System calibration phase.                 |
    | (motor_imagery_          | (Sham Feedback)       | 40 trials/run          | Feedback is simulated/sham.               |
    | acquisition)             |                       | (20 Left / 20 Right)   |                                           |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+
    | **MI Online**            | User Training         | 4 Runs                 | Real-time BCI usage.                      |
    | (motor_imagery_online)   | (Real Feedback)       | 40 trials/run          | Positive feedback (blue bar) based on     |
    |                          |                       | (20 Left / 20 Right)   | classifier confidence.                    |
    +--------------------------+-----------------------+------------------------+-------------------------------------------+

    Trial Structure (MI Tasks)
    --------------------------
    Total trial duration is 8 seconds + inter-trial interval.

    +-------------------+-------------------+---------------------------------------------------------------+
    | Time (s)          | Event             | Description                                                   |
    +===================+===================+===============================================================+
    | **0.0 - 2.0 s**   | Fixation          | Green cross appears on screen.                                |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **2.0 s**         | Audio Cue         | Acoustic signal announces upcoming task.                      |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **3.0 - 4.25 s**  | Visual Cue        | Red arrow points Left or Right (Duration: 1.25s).             |
    +-------------------+-------------------+---------------------------------------------------------------+
    | **4.25 - 8.0 s**  | Feedback          | Continuous feedback (blue bar) provided for 3.75s.            |
    |                   |                   | *Acquisition:* Sham feedback. *Online:* Real positive feedback.|
    +-------------------+-------------------+---------------------------------------------------------------+
    | **8.0 s +**       | Inter-trial       | Black screen for 1.5s to 3.5s (Randomized).                   |
    +-------------------+-------------------+---------------------------------------------------------------+
    """
    def __init__(self, data_to_load = None):
        # 1. Define specific configuration for Cho
        path = os.path.join(DATABASES_PATH, 'Dreyer2023', 'Signals', 
                            'DATA C')
        subjects = list(range(82, 87))
        event_id = {'init_run': 32769, 
                    'end_run': 32770,
                    'start_trial':768,
                    'cross_aparition':786,
                    'acoustic_signal': 33282,
                    'left_hand': 769,
                    'right_hand': 770,
                    'start_feedback': 781,
                    'end_trial':800}

        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, event_id=event_id, code='Dreyer2023C')
        
        # 3. Database-specific attributes
        self.sfreq = 512
        self.standard_montage = "standard_1020"
        self.ch_names = {"eeg":['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 
                                'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 
                                'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'],
                        "eog":["EOG1","EOG2", "EOG3"],
                        "emg":["EMGl", "EMGr"]}
        self.data_availables = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest']
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions_available = ['session_1']
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata(path = os.path.join(DATABASES_PATH, 'Dreyer2023'))
    
    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load
        
    def _load_subjects_metadata(self, path=None):
        # Check if the file exists
        if os.path.exists(os.path.join(path, 'database_information_C.csv')):
            try:
                subjects_metadata = pd.read_csv(os.path.join(path, 'database_information_C.csv'))
            except Exception as e:
                print(f"Error loading subjects metadata: {e}")
                subjects_metadata = None
        else:
            subjects_metadata = None
        return subjects_metadata
    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject: 
        subject_path = os.path.join(self.dataset_path, f's{subject_id:02d}.mat')
        if not os.path.exists(subject_path):
            print(f"Warning: File not found for subject {subject_id}")
            return None
        subject_data = {}


        for data in self.data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"C{subject_id}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"C{subject_id}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"C{subject_id}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info()
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    def _create_raw_simple(self, file_path):
        raw =  mne.io.read_raw_gdf(file_path, eog=None, misc=None, 
                                    stim_channel='auto', exclude=(), 
                                    include=None, preload=False, verbose=None)
        mapping_types = {'EMGg': 'emg', 'EMGd': 'emg', 'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog'}
        raw.set_channel_types(mapping_types)
        # 2. Rename them to match the paper's English terminology
        mapping_names = {'EMGg': 'EMGl', 'EMGd': 'EMGr'}
        raw.rename_channels(mapping_names)
        # 3. Set the system
        raw.set_montage(self.standard_montage)
        return raw

    def _obtain_extra_info(self):
        extra_info = {}
        extra_info['n_imagery_trials_per_run'] = 40
        return extra_info
        
        
    # METODOS VIEJOS
    
    
    def get_data(self, subjects=None, data_to_load=None, sessions=None):
        data_to_load = data_to_load if data_to_load is not None else self.data_to_load
        
        # Si el usuario pide sesiones específicas en esta función, sobreescribimos temporalmente
        # las sesiones configuradas por defecto en el objeto.
        current_sessions = sessions if sessions is not None else getattr(self, 'sessions', ["session_1"])
        subjects, data_to_load = self._validate_inputs(subjects, data_to_load)
        data_dict = {"session_1":{}} 
        for subject in subjects:
            if subject not in self.subject_list:
                print(f"Sujeto {subject} no está en la lista válida. Omitiendo...")
                continue
            data_dict["session_1"][subject] = self._get_single_subject_data(subject, data_to_load, current_sessions)
                
        return data_dict

    def _get_single_subject_data(self, subject, data_to_load, sessions):
        subject_data = {}
        subject_path = os.path.join(self.dataset_path, f'C{subject}')


        for data in data_to_load:
            subject_data[data] = {}
            
            # Cargamos directamente en session_dict (sin nivel anidado extra)
            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = os.path.join(subject_path, f"C{subject}_R{run}_acquisition.gdf")
                    if os.path.exists(file_path): # Buena práctica por si falta algún archivo
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = os.path.join(subject_path, f"C{subject}_R{run}_onlineT.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                        
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = os.path.join(subject_path, f"C{subject}_{baseline}_baseline.gdf")
                    if os.path.exists(file_path):
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # 5. La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info()
        return subject_data
        
        
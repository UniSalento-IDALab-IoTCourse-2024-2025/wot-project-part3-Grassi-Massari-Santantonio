import pandas as pd
import numpy as np
import joblib
import time
import os
import sys
import argparse
from typing import Dict, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RaspberryDeliveryClassifier:
    """
    Versione ottimizzata per Raspberry Pi con preprocessing integrato
    """
    def __init__(self, model_path: str = "./weight/delivery_model.pkl"):
        self.model = None
        self.label_encoder = None
        self.feature_columns = []
        self.model_path = model_path
        
    def load_model(self):
        """
        Carica il modello pre-addestrato
        """
        try:
            logger.info(f"Caricamento modello da {self.model_path}...")
            start_time = time.time()
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            
            load_time = time.time() - start_time
            logger.info(f"Modello caricato in {load_time:.2f} secondi")
            logger.info(f"Classi disponibili: {list(self.label_encoder.classes_)}")
            logger.info(f"Numero features: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            return False
    
    def preprocess_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Preprocessa il CSV per renderlo compatibile con il classifier
        """
        try:
            logger.info(f"Preprocessing CSV: {csv_path}")
            
            # Legge il CSV originale
            df = pd.read_csv(csv_path)
            
            # Mappa i nomi delle colonne
            column_mapping = {
                'Sample_Index': 'timestamp',
                'Temperature [degC]': 'temperature',
                'Acc_X [mg]': 'acc_x_mg',
                'Acc_Y [mg]': 'acc_y',
                'Acc_Z [mg]': 'acc_z',
                'Gyro_X [dps]': 'gyro_x',
                'Gyro_Y [dps]': 'gyro_y',
                'Gyro_Z [dps]': 'gyro_z'
            }
            
            # Rinomina le colonne se necessario
            df = df.rename(columns=column_mapping)
            
            # Verifica che tutte le colonne richieste siano presenti
            required_columns = ['timestamp', 'temperature', 'acc_x_mg', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Colonne mancanti dopo preprocessing: {missing_columns}")
            
            logger.info(f"CSV preprocessato con successo. Righe: {len(df)}, Colonne: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Errore nel preprocessing del CSV: {e}")
            raise
    
    def extract_features(self, delivery_data: pd.DataFrame) -> Dict:
        """
        Estrae features da una serie temporale (ottimizzato per Raspberry Pi)
        """
        features = {}
        
        # Colonne sensori
        sensor_cols = ['temperature', 'acc_x_mg', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Features statistiche base per ogni sensore
        for col in sensor_cols:
            if col in delivery_data.columns:
                col_data = delivery_data[col].values  # Usa numpy per velocit
                features[f'{col}_mean'] = np.mean(col_data)
                features[f'{col}_std'] = np.std(col_data)
                features[f'{col}_max'] = np.max(col_data)
                features[f'{col}_min'] = np.min(col_data)
                features[f'{col}_median'] = np.median(col_data)
                features[f'{col}_range'] = np.max(col_data) - np.min(col_data)
        
        # Magnitude accelerometro
        if all(col in delivery_data.columns for col in ['acc_x_mg', 'acc_y', 'acc_z']):
            acc_magnitude = np.sqrt(
                delivery_data['acc_x_mg'].values**2 + 
                delivery_data['acc_y'].values**2 + 
                delivery_data['acc_z'].values**2
            )
            features['acc_magnitude_mean'] = np.mean(acc_magnitude)
            features['acc_magnitude_std'] = np.std(acc_magnitude)
            features['acc_magnitude_max'] = np.max(acc_magnitude)
            
        # Magnitude giroscopio
        if all(col in delivery_data.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            gyro_magnitude = np.sqrt(
                delivery_data['gyro_x'].values**2 + 
                delivery_data['gyro_y'].values**2 + 
                delivery_data['gyro_z'].values**2
            )
            features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
            features['gyro_magnitude_std'] = np.std(gyro_magnitude)
            features['gyro_magnitude_max'] = np.max(gyro_magnitude)
            
        # Trend analysis
        timestamps = np.arange(len(delivery_data))
        for col in sensor_cols:
            if col in delivery_data.columns:
                try:
                    # Calcolo manuale della pendenza 
                    x = timestamps
                    y = delivery_data[col].values
                    n = len(x)
                    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                    features[f'{col}_trend'] = slope
                except:
                    features[f'{col}_trend'] = 0
        
        # Features temporali
        features['delivery_duration'] = len(delivery_data)
        features['timestamp_range'] = delivery_data['timestamp'].max() - delivery_data['timestamp'].min()
        
        # Varianza rolling 
        window_size = min(5, len(delivery_data) // 2)
        if window_size > 1:
            for col in ['acc_x_mg', 'acc_y', 'acc_z']:
                if col in delivery_data.columns:
                    # Calcolo rolling variance manuale
                    data = delivery_data[col].values
                    rolling_vars = []
                    for i in range(window_size, len(data)):
                        window = data[i-window_size:i]
                        rolling_vars.append(np.var(window))
                    if rolling_vars:
                        features[f'{col}_rolling_var_mean'] = np.mean(rolling_vars)
                    else:
                        features[f'{col}_rolling_var_mean'] = 0
        
        return features
    
    def predict_delivery(self, delivery_data: pd.DataFrame) -> tuple:
        """
        Predice il risultato per una singola consegna
        """
        if self.model is None:
            raise ValueError("Modello non caricato! Chiamare load_model() prima.")
        
        start_time = time.time()
        
        # Estrazione features
        features = self.extract_features(delivery_data)
        
        # Conversione in array numpy 
        feature_array = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        
        # Predizione
        prediction = self.model.predict(feature_array)[0]
        probability = self.model.predict_proba(feature_array)[0]
        
        # Decodifica label
        result = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probability.max()
        
        inference_time = time.time() - start_time
        
        return result, confidence, inference_time
    
    def predict_from_folder(self, folder_path: str, csv_filename: str = "merged_sensors.csv") -> dict:
        """
        Predice da una cartella contenente il CSV
        """
        try:
            # Costruisce il path completo
            csv_path = os.path.join(folder_path, csv_filename)
            
            # Verifica che il file esista
            if not os.path.exists(csv_path):
                return {"error": f"File {csv_path} non trovato"}
            
            logger.info(f"Elaborazione file: {csv_path}")
            
            # Preprocessa il CSV
            df = self.preprocess_csv(csv_path)
            
            # Effettua la predizione
            result, confidence, inference_time = self.predict_delivery(df)
            
            return {
                "folder": folder_path,
                "csv_file": csv_filename,
                "prediction": result,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(inference_time * 1000, 2),
                "data_points": len(df),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione da cartella: {e}")
            return {
                "folder": folder_path,
                "csv_file": csv_filename,
                "error": str(e),
                "status": "error"
            }
    
    def predict_from_csv(self, csv_path: str, delivery_id: Optional[int] = None) -> dict:
        """
        Predice da un file CSV (mantenuto per compatibilit)
        """
        try:
            logger.info(f"Caricamento dati da {csv_path}...")
            df = self.preprocess_csv(csv_path)
            
            if delivery_id is not None:
                # Filtra per delivery_id specifico
                delivery_data = df[df['delivery_id'] == delivery_id]
                if delivery_data.empty:
                    return {"error": f"Delivery ID {delivery_id} non trovato"}
            else:
                # Usa tutto il dataset come una singola consegna
                delivery_data = df
            
            result, confidence, inference_time = self.predict_delivery(delivery_data)
            
            return {
                "prediction": result,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(inference_time * 1000, 2),
                "data_points": len(delivery_data)
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione: {e}")
            return {"error": str(e)}
    
    def predict_realtime(self, data_buffer: list) -> dict:
        """
        Predice da un buffer di dati in tempo reale
        """
        try:
            # Conversione lista in DataFrame
            df = pd.DataFrame(data_buffer)
            
            result, confidence, inference_time = self.predict_delivery(df)
            
            return {
                "prediction": result,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(inference_time * 1000, 2),
                "data_points": len(data_buffer)
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione real-time: {e}")
            return {"error": str(e)}

def main():
    """
    Funzione principale per utilizzo da linea di comando
    """
    parser = argparse.ArgumentParser(description='Raspberry Pi Delivery Classifier')
    parser.add_argument('folder', help='Cartella contenente il file merged_sensors.csv')
    parser.add_argument('--model', default='./weight/delivery_model.pkl', 
                       help='Percorso del modello (default: ./weight/delivery_model.pkl)')
    parser.add_argument('--csv', default='merged_sensors.csv', 
                       help='Nome del file CSV (default: merged_sensors.csv)')
    parser.add_argument('--output-format', choices=['detailed', 'simple', 'json'], 
                       default='detailed', help='Formato output (default: detailed)')
    
    args = parser.parse_args()
    
    # Verifica che la cartella esista
    if not os.path.exists(args.folder):
        logger.error(f"Cartella non trovata: {args.folder}")
        sys.exit(1)
    
    # Inizializza classifier
    classifier = RaspberryDeliveryClassifier(args.model)
    
    # Carica modello
    if not classifier.load_model():
        logger.error("Impossibile caricare il modello!")
        sys.exit(1)
    
    # Effettua predizione
    result = classifier.predict_from_folder(args.folder, args.csv)
    
    # Output 
    if args.output_format == 'simple':
        # Output command line
        if result.get("status") == "success":
            print(f"PREDICTION:{result['prediction']}")
            print(f"CONFIDENCE:{result['confidence']:.4f}")
            print(f"STATUS:success")
        else:
            print(f"STATUS:error")
            print(f"ERROR:{result.get('error', 'Errore sconosciuto')}")
    elif args.output_format == 'json':
        # Output JSON
        import json
        print(json.dumps(result, indent=2))
    else:
        # Output dettagliato 
        print("\n" + "="*50)
        print("RISULTATO PREDIZIONE")
        print("="*50)
        
        if result.get("status") == "success":
            print(f" Cartella: {result['folder']}")
            print(f"File CSV: {result['csv_file']}")
            print(f" Predizione: {result['prediction']}")
            print(f" Confidenza: {result['confidence']:.4f}")
            print(f" Tempo inferenza: {result['inference_time_ms']:.2f} ms")
            print(f" Punti dati: {result['data_points']}")
            print(f" Status: {result['status']}")
        else:
            print(f" Errore: {result.get('error', 'Errore sconosciuto')}")
            print(f" Cartella: {result.get('folder', 'N/A')}")
            print(f" File CSV: {result.get('csv_file', 'N/A')}")
        
        print("="*50)
    
        # Codice di uscita
    if result.get("status") == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        
        print("Modalit test - usando dati di esempio")
        
       
        classifier = RaspberryDeliveryClassifier("./weight/delivery_model.pkl")
        
        # Carica modello
        if not classifier.load_model():
            logger.error("Impossibile caricare il modello!")
            sys.exit(1)
        
        
        sample_data = [
            {"timestamp": 0, "temperature": 35.0, "acc_x_mg": 20.5, "acc_y": 25.3, "acc_z": 22.1, 
             "gyro_x": 15.2, "gyro_y": -10.3, "gyro_z": 5.8},
            {"timestamp": 1, "temperature": 35.1, "acc_x_mg": 21.2, "acc_y": 26.1, "acc_z": 23.5, 
             "gyro_x": 16.1, "gyro_y": -11.2, "gyro_z": 6.2},
        ]
        
        result = classifier.predict_realtime(sample_data)
        print(f"Predizione real-time: {result}")
    else:
        # Modalit normale con argomenti da linea di comando
        main()

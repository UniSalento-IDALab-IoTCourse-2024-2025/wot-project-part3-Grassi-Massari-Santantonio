import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import linregress
import joblib
import os
from typing import Dict, List, Tuple

class DeliveryClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def extract_features(self, delivery_data: pd.DataFrame) -> Dict:
        """
        Estrae features da una serie temporale di una singola consegna
        """
        features = {}
        
        # Colonne sensori
        sensor_cols = ['temperature', 'acc_x_mg', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Features statistiche base per ogni sensore
        for col in sensor_cols:
            if col in delivery_data.columns:
                features[f'{col}_mean'] = delivery_data[col].mean()
                features[f'{col}_std'] = delivery_data[col].std()
                features[f'{col}_max'] = delivery_data[col].max()
                features[f'{col}_min'] = delivery_data[col].min()
                features[f'{col}_median'] = delivery_data[col].median()
                features[f'{col}_range'] = delivery_data[col].max() - delivery_data[col].min()
        
        # Magnitude accelerometro
        if all(col in delivery_data.columns for col in ['acc_x_mg', 'acc_y', 'acc_z']):
            acc_magnitude = np.sqrt(
                delivery_data['acc_x_mg']**2 + 
                delivery_data['acc_y']**2 + 
                delivery_data['acc_z']**2
            )
            features['acc_magnitude_mean'] = acc_magnitude.mean()
            features['acc_magnitude_std'] = acc_magnitude.std()
            features['acc_magnitude_max'] = acc_magnitude.max()
            
        # Magnitude giroscopio
        if all(col in delivery_data.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            gyro_magnitude = np.sqrt(
                delivery_data['gyro_x']**2 + 
                delivery_data['gyro_y']**2 + 
                delivery_data['gyro_z']**2
            )
            features['gyro_magnitude_mean'] = gyro_magnitude.mean()
            features['gyro_magnitude_std'] = gyro_magnitude.std()
            features['gyro_magnitude_max'] = gyro_magnitude.max()
            
        # regressione lineare (trend) per ogni sensore
        timestamps = range(len(delivery_data))
        for col in sensor_cols:
            if col in delivery_data.columns:
                try:
                    slope, _, _, _, _ = linregress(timestamps, delivery_data[col])
                    features[f'{col}_trend'] = slope
                except:
                    features[f'{col}_trend'] = 0
        
        # Features temporali
        features['delivery_duration'] = len(delivery_data)
        features['timestamp_range'] = delivery_data['timestamp'].max() - delivery_data['timestamp'].min()
        
        # Varianza rolling (instabilità nel tempo)
        window_size = min(5, len(delivery_data) // 2)
        if window_size > 1:
            for col in ['acc_x_mg', 'acc_y', 'acc_z']:
                if col in delivery_data.columns:
                    rolling_var = delivery_data[col].rolling(window=window_size).var()
                    features[f'{col}_rolling_var_mean'] = rolling_var.mean()
        
        return features
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara il dataset raggruppando per delivery_id ed estraendo features
        """
        print("Preparazione dataset...")
        
        # Raggruppamento per delivery_id
        delivery_groups = df.groupby('delivery_id')
        
        features_list = []
        labels_list = []
        
        for delivery_id, group in delivery_groups:
            # Estrazione delle features per questa consegna
            features = self.extract_features(group)
            features_list.append(features)
            
            # Label
            labels_list.append(group['delivery_result'].iloc[0])
        
        # Conversione in DataFrame
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list)
        
        # Gestione NaN (riempimento con 0)
        features_df = features_df.fillna(0)
        
        # Salvataggio dei nomi delle features
        self.feature_columns = features_df.columns.tolist()
        
        print(f"Dataset preparato: {len(features_df)} consegne, {len(self.feature_columns)} features")
        print(f"Distribuzione classi: {labels_series.value_counts()}")
        
        return features_df, labels_series
    
    def train(self, df: pd.DataFrame, cv_folds: int = 5, random_state: int = 42):
        """
        Addestra il modello con stratified k-fold cross validation
        """
        print("Inizio addestramento...")
        
        # Preparazione dataset
        X, y = self.prepare_dataset(df)
        
        # Encoding delle labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Inizializzazione del modello
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1  # Usa tutti i core CPU
        )
        
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        print(f"Cross-validation con {cv_folds} fold...")
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=skf, scoring='accuracy')
        
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Addestramento finale su tutto il dataset
        print("Addestramento finale...")
        self.model.fit(X, y_encoded)
        
        # Predizioni per valutazione
        y_pred = self.model.predict(X)
        
        # Report
        print("\nClassification Report:")
        print(classification_report(y_encoded, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance 
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features più importanti:")
        print(feature_importance.head(10))
        
        return cv_scores
    
    def save_model(self, model_path: str = "delivery_model.pkl"):
        """
        Salva il modello e i componenti necessari
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato!")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        print(f"Modello salvato in: {model_path}")
        
        # Salvataggio info del modello
        info_path = model_path.replace('.pkl', '_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Modello: {type(self.model).__name__}\n")
            f.write(f"Numero features: {len(self.feature_columns)}\n")
            f.write(f"Classi: {list(self.label_encoder.classes_)}\n")
            f.write(f"Features: {self.feature_columns}\n")
        
        print(f"Info modello salvate in: {info_path}")
    
    def load_model(self, model_path: str = "delivery_model.pkl"):
        """
        Carica il modello salvato
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Modello caricato da: {model_path}")
    
    def predict_delivery(self, delivery_data: pd.DataFrame) -> str:
        """
        Predice il risultato per una singola consegna
        """
        if self.model is None:
            raise ValueError("Modello non caricato!")
        
        # Estrazione features
        features = self.extract_features(delivery_data)
        
        # Conversione in DataFrame con le colonne giuste
        features_df = pd.DataFrame([features])
        
        # Verifica delle features siano presenti
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Riordina colonne
        features_df = features_df[self.feature_columns]
        
        # Predizione
        prediction = self.model.predict(features_df)[0]
        probability = self.model.predict_proba(features_df)[0]
        
        # Decodifica label
        result = self.label_encoder.inverse_transform([prediction])[0]
        
        # Confidenza
        confidence = probability.max()
        
        return result, confidence


if __name__ == "__main__":
    # Caricmento dati
    df = pd.read_csv('./Dataset/sensor_data/sensor_data.csv')
    
    # Inizializzazione classifier
    classifier = DeliveryClassifier()
    
    # Addestramento con cross-validation
    cv_scores = classifier.train(df, cv_folds=5)
    
    # Salvataggio 
    classifier.save_model("./weight/delivery_model.pkl")
    
    print("\nAddestramento completato!")
    print(f"Dimensione file modello: {os.path.getsize('./weight/delivery_model.pkl') / 1024:.2f} KB")
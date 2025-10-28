## Descrizione

Questa repository contiene il codice sorgente e i file relativi al modello di Intelligenza Artificiale sviluppato per classificare la "qualità" o lo "stato di salute" di una consegna basandosi su dati sensoriali. Il modello utilizza un approccio di **apprendimento supervisionato**, specificamente una **Random Forest**, addestrata su dati etichettati per predire una delle classi: "low", "medium", o "high".

Il modello è progettato per ricevere in input una serie di misure (feature) provenienti dai sensori durante una consegna e restituire la classe di qualità più probabile.

## Tecnologie Utilizzate

* **Linguaggio:** Python
* **Librerie Principali:**
    * Scikit-learn (per l'implementazione della Random Forest, scaling e encoding)
    * Pandas (per la manipolazione dei dati)
    * Joblib (per salvare/caricare il modello, lo scaler e l'encoder)
    * Flask (per creare un server API REST per l'inferenza - `IA.py`)
    * TensorFlow/Keras (utilizzato per un modello alternativo `.h5` e la conversione in `.tflite`, anche se la descrizione principale si basa su Random Forest)
* **Tecnica di Machine Learning:** Apprendimento Supervisionato (Classificazione)
* **Algoritmo:** Random Forest Classifier (`delivery_model.pkl`)

## Funzionalità Principali

1.  **Addestramento del Modello (`DeliveryTrainer.py`):**
    * Carica i dati dai file CSV presenti nella cartella `Dataset/` (suddivisi per etichetta: `high`, `medium`, `low`).
    * Preprocessa i dati (es. scaling delle feature numeriche tramite `StandardScaler`).
    * Codifica le etichette testuali ("low", "medium", "high") in valori numerici (`LabelEncoder`).
    * Addestra un modello Random Forest Classifier (con 100 alberi decisionali, come specificato in `Documento.docx`).
    * Salva il modello addestrato (`delivery_model.pkl`), lo scaler (`scaler.pkl`) e l'encoder (`label_encoder.pkl`) nella cartella `weight/` per l'uso futuro in inferenza.

2.  **Inferenza (`DeliveryInference.py`, `IA.py`):**
    * Carica il modello pre-addestrato, lo scaler e l'encoder.
    * `DeliveryInference.py` permette di effettuare previsioni su nuovi dati (singole misure o batch).
    * `IA.py` implementa un server web Flask che espone un endpoint API (`/predict`) per ricevere nuove misure (come JSON) e restituire la previsione della classe in tempo reale.

3.  **Conversione TFLite (`TFLiteConverter.py`):**
    * Script per convertire un modello salvato in formato Keras (`.h5`) in formato TensorFlow Lite (`.tflite`), ottimizzato per l'esecuzione su dispositivi mobile o edge (Nota: il modello primario descritto sembra essere `delivery_model.pkl`, basato su scikit-learn).

## Struttura del Progetto

* **`Dataset/`**: Contiene i dati utilizzati per l'addestramento e la validazione.
    * `high/`, `medium/`, `low/`: Sottocartelle con file CSV contenenti dati etichettati.
    * `unlabeled/`: Contiene dati non etichettati (potenzialmente per test o inferenza batch).
    * `sensor_data/`: Potrebbe contenere dati grezzi o aggregati.
    * `dataset-meta.json`: File con metadati sul dataset (opzionale).
* **`weight/`**: Contiene i file del modello addestrato e gli oggetti di preprocessing salvati.
    * `delivery_model.pkl`: Il modello Random Forest addestrato.
    * `scaler.pkl`: L'oggetto `StandardScaler` fittato sui dati di training.
    * `label_encoder.pkl`: L'oggetto `LabelEncoder` fittato sulle etichette.
    * *(Altri file come `.h5`, `.tflite` potrebbero derivare da esperimenti con altri modelli)*
* **`DeliveryTrainer.py`**: Script per l'addestramento del modello.
* **`DeliveryInference.py`**: Script (o modulo) per eseguire l'inferenza utilizzando il modello salvato.
* **`IA.py`**: Script per avviare il server Flask che espone l'API di inferenza.
* **`TFLiteConverter.py`**: Script per la conversione del modello in formato TFLite.
* **`requirements_training.txt`**: Dipendenze Python necessarie per eseguire `DeliveryTrainer.py`.
* **`requirement_inference.txt`**: Dipendenze Python necessarie per eseguire `DeliveryInference.py` e `IA.py`.
* **`Documento.docx`**: Documentazione che spiega i concetti di apprendimento supervisionato e Random Forest applicati al progetto.
* **`.gitignore`**: File per specificare i file e le cartelle da ignorare in Git.

## Installazione ed Esecuzione

### Prerequisiti

* Python 3.x installato.
* `pip` (Python package installer).

### Installazione Dipendenze

È consigliabile creare un ambiente virtuale.

1.  **Per l'addestramento:**
    ```bash
    pip install -r requirements_training.txt
    ```
2.  **Per l'inferenza (compreso il server Flask):**
    ```bash
    pip install -r requirement_inference.txt
    ```

### Esecuzione

1.  **Addestramento:**
    * Assicurarsi che i dati siano correttamente posizionati nella cartella `Dataset/`.
    * Eseguire lo script di training:
        ```bash
        python DeliveryTrainer.py
        ```
    * Questo genererà/aggiornerà i file `.pkl` nella cartella `weight/`.

2.  **Avvio del Server di Inferenza:**
    * Assicurarsi che i file `.pkl` siano presenti nella cartella `weight/`.
    * Eseguire lo script del server Flask:
        ```bash
        python IA.py
        ```
    * Il server sarà in ascolto (di default su `http://0.0.0.0:5000` o simile, controllare l'output dello script) e pronto a ricevere richieste POST all'endpoint `/predict`.

3.  **Inferenza Standalone (opzionale):**
    * Modificare ed eseguire `DeliveryInference.py` per testare previsioni su dati specifici direttamente da script.

## Autori

* **Francesco Grassi**
* **Daniele Massari**
* **Alessio Santantonio**

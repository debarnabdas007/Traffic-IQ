import pandas as pd
import pickle
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'vehicle_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'knn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'data', 'models', 'scaler.pkl')

def train():
    print("--- Traffic-IQ Training Module (v2: With Scaling) ---")
    
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        print("❌ Error: Dataset not found.")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_PATH)
    
    # Clean duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    print(f"Loaded {len(df)} samples (Removed {original_len - len(df)} duplicates).")

    # 2. Features & Target
    X = df[['Area', 'Aspect_Ratio']].values
    y = df['Label'].values

    # 3. SPLIT FIRST (Crucial: Never fit scaler on test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. SCALING (The Fix!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn scale from Train
    X_test_scaled = scaler.transform(X_test)        # Apply same scale to Test

    # 5. Train KNN
    # Increased K to 5 to reduce noise sensitivity
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    print("\nTraining complete.")

    # 6. Evaluate
    print("\n--- Evaluation on Test Set ---")
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Color-coded output
    if acc > 0.85:
        grade = "✅ EXCELLENT"
    elif acc > 0.70:
        grade = "⚠️ ACCEPTABLE"
    else:
        grade = "❌ POOR"
        
    print(f"Accuracy: {acc*100:.2f}%  [{grade}]")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Car', 'Truck', 'Bike'], zero_division=0))

    # 7. Save BOTH Model and Scaler
    # We need the scaler in the final app to scale new video frames!
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
        
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nSaved Model -> {MODEL_PATH}")
    print(f"Saved Scaler -> {SCALER_PATH}")
    print("(You must load BOTH in the main app)")

if __name__ == "__main__":
    train()
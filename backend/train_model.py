import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_nutrition_model():
    # Load your labeled data
    df = pd.read_csv('data/labeled_nutrition.csv')

    # 1. Select Features (Inputs) and Target (Output)
    # We use Age, Gender, Dietary Iron, and Dietary Vit D to predict 'any_deficiency'
    X = df[['RIDAGEYR', 'RIAGENDR', 'DR1TIRON', 'DR1TVD']]
    y = df['any_deficiency']

    # 2. Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize and Train the Random Forest
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save the trained model to a file
    # Ensure you have an 'ml_models' folder or just save it in 'data' for now
    joblib.dump(model, 'data/nutrition_model.pkl')
    print("\nModel saved as 'data/nutrition_model.pkl'")

if __name__ == "__main__":
    train_nutrition_model()
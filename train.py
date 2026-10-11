import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def main():
    # Load dataset
    df = pd.read_csv('dataset1.csv')
    
    # Feature selection
    features = ['Fever', 'Headache', 'Cough', 'Fatigue', 'Body_Pain']
    X = df[features]
    y = df['Disease']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully with Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save trained model
    joblib.dump(model, 'model.pkl')
    print("Model saved as 'model.pkl'")

if __name__ == '__main__':
    main()
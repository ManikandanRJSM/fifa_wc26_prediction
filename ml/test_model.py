from sklearn.metrics import classification_report, accuracy_score
import joblib


if __name__ == "__main__":

    # Load saved model
    loaded_model = joblib.load("fifa_xgb_model.pkl")

    # Predict on test data
    y_pred = loaded_model.predict(X_test)

    print("Actual results   :", list(y_test))
    print("Predicted results:", list(y_pred))
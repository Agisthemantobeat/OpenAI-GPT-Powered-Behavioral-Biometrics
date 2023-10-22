import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keyboard

# Constants for keystroke features
KEY_HOLD = 'hold_time'
KEY_RELEASE = 'release_time'
TYPING_SPEED = 'typing_speed'

# Data collection - capture keystroke dynamics
def collect_keystroke_data(duration=10):
    print("Collecting keystroke data for User...")
    keystrokes = []
    start_time = time.time()

    while time.time() - start_time < duration:
        event = keyboard.read_event(suppress=True)
        timestamp = time.time()
        keystrokes.append((event, timestamp))

    return keystrokes

# Feature extraction - compute key hold time, key release time, and typing speed
def extract_features(keystrokes):
    features = []
    for i in range(1, len(keystrokes)):
        prev_event, prev_timestamp = keystrokes[i - 1]
        event, timestamp = keystrokes[i]

        if event.event_type == keyboard.KEY_DOWN and prev_event.event_type == keyboard.KEY_DOWN:
            hold_time = timestamp - prev_timestamp
            release_time = 0  # Not applicable for key down events
            typing_speed = 0  # Not applicable for key down events
        elif event.event_type == keyboard.KEY_UP and prev_event.event_type == keyboard.KEY_DOWN:
            hold_time = 0  # Not applicable for key up events
            release_time = timestamp - prev_timestamp
            typing_speed = 1 / (timestamp - prev_timestamp)
        else:
            continue

        features.append({
            KEY_HOLD: hold_time,
            KEY_RELEASE: release_time,
            TYPING_SPEED: typing_speed
        })

    return features

# Train a machine learning model
def train_model(features, labels):
    # Extract features and convert to a proper format
    X = np.array([[f[KEY_HOLD], f[KEY_RELEASE], f[TYPING_SPEED]] for f in features])

    # Convert labels to numeric values
    label_to_numeric = {label: i for i, label in enumerate(set(labels))}
    y = np.array([label_to_numeric[label] for label in labels])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple machine learning model (Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

    return model


# Authentication
def authenticate_user(model, keystroke_features):
    features = np.array([[kf[KEY_HOLD], kf[KEY_RELEASE], kf[TYPING_SPEED]] for kf in keystroke_features])
    prediction = model.predict(features)
    # print()
    return prediction[0]



# Main function
def main():
    keystrokes = collect_keystroke_data(duration=10)  # Capture keystrokes for 10 seconds
    features = extract_features(keystrokes)
    # print(features)

    # Assume two users for demonstration
    labels = ['user1', 'user2']
    user_labels = np.random.choice(labels, len(features))  # Simulated user labels
    # print(user_labels)

    model = train_model(features, user_labels)

    # Simulated authentication using the trained model
    # Replace with actual keystroke data from a user
    user_keystroke_features = [{KEY_HOLD: 0, KEY_RELEASE: 0.05, TYPING_SPEED: 9.13}]
    user = authenticate_user(model, user_keystroke_features)
    # print(user)
    print("Authenticated User:", labels[user])

if __name__ == "__main__":
    main()

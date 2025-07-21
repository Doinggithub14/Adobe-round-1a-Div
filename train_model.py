import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('headings_dataset.csv')

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Encode labels (H1, H2, H3 → 0,1,2)
encoder = LabelEncoder()
y = encoder.fit_transform(df['level'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and transformers
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

print("✅ Model, vectorizer, and encoder saved!")
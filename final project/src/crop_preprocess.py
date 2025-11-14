import os, sys, json, joblib
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(ROOT, 'data', 'crop_dataset')
OUT_DIR = os.path.join(ROOT, 'data', 'processed', 'crop')
os.makedirs(OUT_DIR, exist_ok=True)

# find first csv
csvs = glob(os.path.join(INPUT_DIR, '**', '*.csv'), recursive=True)
if not csvs:
    print('No CSV found in', INPUT_DIR); sys.exit(1)
csv_path = csvs[0]
print('Using CSV:', csv_path)

df = pd.read_csv(csv_path)
print('Initial shape:', df.shape)
# guess target column
candidates = ['label','crop','Crop','target','y','Crop_Type','crop_name']
target_col = None
for c in candidates:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # try columns with few unique values
    for c in df.columns:
        if df[c].nunique() < 50 and df[c].dtype == object:
            target_col = c
            break

if target_col is None:
    print('Could not detect target column automatically. Columns are:')
    print(list(df.columns))
    print('Please re-run with the correct file containing a known target column. Exiting.')
    sys.exit(1)

print('Detected target column:', target_col)

# missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]

for c in num_cols:
    med = df[c].median()
    df[c].fillna(med, inplace=True)

for c in cat_cols:
    mode = df[c].mode()
    if not mode.empty:
        df[c].fillna(mode[0], inplace=True)
    else:
        df[c].fillna('NA', inplace=True)

# create features if possible
cols = df.columns
if set(['N','P','K']).issubset(cols):
    df['soil_fertility'] = (df['N'] + df['P'] + df['K'])/3.0
    print('Created soil_fertility')
if set(['temperature','humidity']).issubset(cols):
    df['climate_index'] = df['temperature'] * df['humidity']
    print('Created climate_index')

# drop perfectly duplicated columns or extremely correlated
try:
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print('Dropped highly correlated columns:', to_drop)
except Exception as e:
    print('Correlation step skipped:', e)

# prepare X,y
X = df.drop(columns=[target_col])
y = df[target_col].astype(str)

# encode categorical columns
encoders = {}
for c in X.columns:
    if X[c].dtype == object or X[c].nunique() < 20:
        # small cardinality -> OneHot
        if X[c].nunique() <= 10:
            dummies = pd.get_dummies(X[c], prefix=c)
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
            encoders[c] = {'type':'onehot','cols':dummies.columns.tolist()}
        else:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = {'type':'label','classes': list(le.classes_)}
print('Final feature shape:', X.shape)

# Split dataset: 70/15/15
try:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
except Exception as e:
    # fallback without stratify
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# scale numeric features
scaler = StandardScaler()
num_columns_final = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train[num_columns_final] = scaler.fit_transform(X_train[num_columns_final])
X_val[num_columns_final] = scaler.transform(X_val[num_columns_final])
X_test[num_columns_final] = scaler.transform(X_test[num_columns_final])

# save processed csvs
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
val_df = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUT_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
pd.DataFrame({'feature_columns': X.columns.tolist()}).to_csv(os.path.join(OUT_DIR, 'feature_columns.csv'), index=False)

# save artifacts
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))
joblib.dump(encoders, os.path.join(OUT_DIR, 'encoders.joblib'))
print('Saved processed data to', OUT_DIR)

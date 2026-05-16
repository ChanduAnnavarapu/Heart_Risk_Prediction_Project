from logger_config import setup_logger
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import GroupShuffleSplit,GroupKFold,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

load_dotenv()
LOG_PATH =  Path(os.getenv("PROJECT_ROOT")) / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

def train_model():
    try: 
        load_dotenv()
        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH =  PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
        
        TARGET_COL= os.getenv("TARGET_COL")
        TEST_SIZE= float(os.getenv("TEST_SIZE"))
        RANDOM_STATE= int(os.getenv("RANDOM_STATE"))
        
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(LOG_PATH)
        
        logger.info("Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        x=df.drop(columns=[TARGET_COL])
        y=df[TARGET_COL]
        logger.info("Splitting dataset into train and test sets...")
        
        row_signature=pd.util.hash_pandas_object(x, index=False)
        gss= GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=int(RANDOM_STATE))
        train_idx, test_idx = next(gss.split(x, y, groups=row_signature))
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logger.info(f"Train set shape: {x_train.shape}, Test set shape: {x_test.shape}")
        
        # Define the best hyperparameters for RandomForestClassifier
        best_rfc_param=RandomForestClassifier(
        n_estimators=1119,
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=11,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.6,
        ccp_alpha=0.0017,
        random_state=int(RANDOM_STATE),
        n_jobs=-1
        )
        
        pipeline=Pipeline(
            steps=[
            ("scaler", StandardScaler()),
            ("Model", best_rfc_param)
        ])
        
        #model training
        pipeline.fit(x_train, y_train)
        logger.info("Model training completed.")
        
        y_train_pred=pipeline.predict(x_train)
        y_test_pred=pipeline.predict(x_test)
        
        #model evaluation
        train_accuracy=round(accuracy_score(y_train, y_train_pred)*100, 2)
        test_accuracy=round(accuracy_score(y_test, y_test_pred)*100, 2)
        logger.info(f"Train Accuracy: {train_accuracy}%")
        logger.info(f"Test Accuracy: {test_accuracy}%")
        
        #save train model
        dump(pipeline, MODEL_PATH)
        logger.info(f"Trained model saved at: {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise e


if __name__ == "__main__":
    train_model()
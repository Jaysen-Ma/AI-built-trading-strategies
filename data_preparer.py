import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
from data_store import DataStore

logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, 
                 data_store: DataStore,
                 symbol='BTCUSD',
                 target_lookahead=10,  # in minutes
                 risk_percentage=0.0003,  # 0.03%
                 reward_percentage=0.0006,  # 0.06%
                 lot_size=0.01):
        """
        Initializes the DataPreparer with database connection and trading parameters.

        Parameters:
        - data_store (DataStore): An instance of the DataStore class.
        - symbol (str): The trading symbol, e.g., 'EURUSD'.
        - target_lookahead (int): Number of minutes to look ahead for target creation.
        - risk_percentage (float): Risk per trade (e.g., 0.0003 for 0.03%).
        - reward_percentage (float): Reward per trade (e.g., 0.0006 for 0.06%).
        - lot_size (float): The lot size per trade (e.g., 0.01).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing DataPreparer with parameters: symbol=%s, "
                          "target_lookahead=%s, risk_percentage=%s, reward_percentage=%s, lot_size=%s",
                          symbol, target_lookahead, risk_percentage, reward_percentage, lot_size)
        self.data_store = data_store
        self.symbol = symbol
        self.target_lookahead = target_lookahead
        self.risk_percentage = risk_percentage
        self.reward_percentage = reward_percentage
        self.lot_size = lot_size
        
        # Initialize placeholders
        self.data: pd.DataFrame = pd.DataFrame()
        self.features = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = None
        self.model = None

    def load_data(self):
        """
        Connects to ArcticDB via DataStore and retrieves the data for the specified symbol.
        """
        self.logger.info("Starting to load data from ArcticDB via DataStore.")
        try:
            self.data = self.data_store.retrieve_data(self.symbol).copy()
            if self.data.empty:
                self.logger.error(f"No data found for symbol: {self.symbol}")
                raise ValueError(f"No data found for symbol: {self.symbol}")
            self.logger.debug("Data columns: %s", self.data.columns.tolist())
            self.data.sort_index(inplace=True)
            self.logger.info("Data for %s loaded successfully with %d records.", self.symbol, len(self.data))
        except Exception as e:
            self.logger.error("Error loading data: %s", e)
            raise

    def create_target(self):
        """
        Creates the target variable based on the specified risk/reward ratio and lookahead period.
        The target has three classes:
        - 0: Bullish (price increases by at least reward_percentage within lookahead)
        - 1: Bearish (price decreases by at least reward_percentage within lookahead)
        - 2: Neutral (price does not move beyond the thresholds)
        """
        self.logger.info("Creating target variable.")
        try:
            # Create a copy of the original data to avoid modifying it directly
            df = self.data.copy()
            self.logger.debug("Original data shape: %s", df.shape)
            
            # Calculate the highest and lowest future prices within the lookahead period
            df['future_high'] = df['close'].shift(-self.target_lookahead).rolling(window=self.target_lookahead).max()
            df['future_low'] = df['close'].shift(-self.target_lookahead).rolling(window=self.target_lookahead).min()
            self.logger.debug("Calculated future_high and future_low.")
            
            # Calculate the percentage change from the current close price to the future high and low prices
            df['pct_change_high'] = (df['future_high'] - df['close']) / df['close']
            df['pct_change_low'] = (df['future_low'] - df['close']) / df['close']
            self.logger.debug("Calculated pct_change_high and pct_change_low.")
            
            # Define the target variable based on the percentage changes
            # 0: Bullish (future high >= reward percentage)
            # 1: Bearish (future low <= -reward percentage)
            # 2: Neutral (neither condition met)
            conditions = [
                df['pct_change_high'] >= self.reward_percentage,
                np.abs(df['pct_change_low']) >= self.reward_percentage
            ]
            choices = [0, 1]  # 0: Bullish, 1: Bearish
            df['target'] = np.select(conditions, choices, default=2)
            self.logger.debug("Target variable created with conditions applied.")
            
            # Drop auxiliary columns
            df.drop(['future_high', 'future_low', 'pct_change_high', 'pct_change_low'], axis=1, inplace=True)
            self.logger.debug("Dropped auxiliary columns.")
            
            # Drop rows with NaN values resulting from shifting
            initial_shape = df.shape
            df.dropna(inplace=True)
            self.logger.debug("Dropped NaN rows. Shape before: %s, after: %s", initial_shape, df.shape)
            
            self.data = df
            self.logger.info("Target variable created with classes: 0 (Bullish), 1 (Bearish), 2 (Neutral).")
        except Exception as e:
            self.logger.error("Error creating target variable: %s", e)
            raise

    def select_features(self, feature_list=None):
        """
        Selects the feature columns for model training.

        Parameters:
        - feature_list (list): Optional list of features to include. If None, all technical indicators are used.
        """
        self.logger.info("Selecting features for model training.")
        try:
            if feature_list is None:
                self.features = [
                    'open', 'high', 'low', 'close',
                    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',
                    'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
                    'RSI_14', 'MACD_12_26_9_MACD_12_26_9',
                    # Add other features as necessary
                ]
                self.logger.debug("Default feature list selected: %s", self.features)
            else:
                self.features = feature_list
                self.logger.debug("Custom feature list selected: %s", self.features)
            
            self.logger.info("Selected %d features for model training.", len(self.features))
        except Exception as e:
            self.logger.error("Error selecting features: %s", e)
            raise

    def split_data(self, test_size=0.2):
        """
        Splits the data into training and testing sets without shuffling to maintain temporal order.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        """
        self.logger.info("Splitting data into training and testing sets with test_size=%.2f.", test_size)
        try:
            X = self.data[self.features].dropna()
            y = self.data.loc[X.index, 'target']
            self.logger.debug("Features shape: %s, Target shape: %s", X.shape, y.shape)
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False)
            
            self.logger.info("Data split into training (%d) and testing (%d) sets.",
                             len(self.X_train), len(self.X_test))
        except Exception as e:
            self.logger.error("Error splitting data: %s", e)
            raise

    def normalize_features(self):
        self.logger.info("Normalizing features using StandardScaler.")
        try:
            self.scaler = StandardScaler()
            
            # Fit on training data
            self.scaler.fit(self.X_train[self.features])
            self.logger.debug("Scaler fitted on training data.")
            
            self.X_train[self.features] = self.scaler.transform(self.X_train[self.features])
            self.logger.debug("Training features transformed.")
            
            # Transform test data using the same scaler
            self.X_test[self.features] = self.scaler.transform(self.X_test[self.features])
            self.logger.debug("Testing features transformed.")
            
            self.logger.info("Features normalized successfully.")
        except Exception as e:
            self.logger.error("Error normalizing features: %s", e)
            raise

    def prepare(self):
        """
        Executes the full data preparation pipeline.
        """
        self.logger.info("Starting data preparation pipeline.")
        try:
            self.load_data()
            self.create_target()
            self.select_features()
            self.split_data()
            self.normalize_features()
            self.logger.info("Data preparation completed successfully.")
        except Exception as e:
            self.logger.error("Error during data preparation: %s", e)
            raise

    def get_data(self):
        """
        Returns the prepared data splits.

        Returns:
        - X_train (DataFrame): Training features.
        - X_test (DataFrame): Testing features.
        - y_train (Series): Training targets.
        - y_test (Series): Testing targets.
        """
        self.logger.debug("Retrieving prepared data splits.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_scalers(self, train_scaler_path='scaler_train.pkl'):
        """
        Saves the fitted scaler to disk for future use.

        Parameters:
        - train_scaler_path (str): File path to save the training scaler.
        """
        self.logger.info("Saving scalers to disk.")
        try:
            joblib.dump(self.scaler, train_scaler_path)
            self.logger.info("Scaler saved to %s.", train_scaler_path)
        except Exception as e:
            self.logger.error("Error saving scalers: %s", e)
            raise

    def train_model(self, iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', l2_leaf_reg=3.0, border_count=254, thread_count=-1):
        """
        Trains a CatBoostClassifier on the prepared training data.

        Parameters:
        - iterations (int): Number of trees.
        - depth (int): Depth of the trees.
        - learning_rate (float): Learning rate.
        - loss_function (str): Loss function to use.
        - l2_leaf_reg (float): L2 regularization term on weights.
        - border_count (int): The number of splits for numerical features.
        - thread_count (int): Number of threads to use. -1 means use all available.
        """
        self.logger.info("Training CatBoost model with iterations=%d, depth=%d, learning_rate=%.2f, loss_function=%s, l2_leaf_reg=%.2f, border_count=%d, thread_count=%d.",
                         iterations, depth, learning_rate, loss_function, l2_leaf_reg, border_count, thread_count)
        try:
            self.model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                loss_function=loss_function,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                thread_count=thread_count,
                verbose=100,
                random_seed=42,
                task_type='GPU',  # Use GPU for acceleration
                devices='0'  # Specify GPU device ID, '0' for first GPU (optional if multiple GPUs)
            )
            
            self.logger.debug("CatBoostClassifier initialized.")
            
            self.model.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test))
            self.logger.info("CatBoost model trained successfully.")
        except Exception as e:
            self.logger.error("Error training CatBoost model: %s", e)
            raise

    def evaluate_model(self):
        """
        Evaluates the trained model on the testing set and prints classification metrics.
        """
        self.logger.info("Evaluating the trained model.")
        try:
            if self.model is None:
                self.logger.warning("Model is not trained yet.")
                return

            # Predict on the test set
            y_pred = self.model.predict(self.X_test)
            self.logger.debug("Model predictions: %s", y_pred[:10])  # Log first 10 predictions

            # Display the classification report as a DataFrame
            self.logger.info("Classification Report:")
            report = classification_report(self.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            self.logger.debug("Classification Report DataFrame:\n%s", report_df)

            # You can choose to log the report or handle it as needed
            # For example, save to a file or log specific metrics
            self.logger.info("\n%s", report_df.to_string())

            # Display the confusion matrix using a heatmap
            self.logger.info("Generating confusion matrix.")
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Pred 0', 'Pred 1', 'Pred 2'], yticklabels=['True 0', 'True 1', 'True 2'])
            plt.title("Confusion Matrix")
            plt.ylabel("Actual Label")
            plt.xlabel("Predicted Label")
            plt.savefig('confusion_matrix.png')
            self.logger.info("Confusion matrix saved as confusion_matrix.png.")
            plt.close()

            # Optionally, display accuracy and other relevant metrics separately
            accuracy = report.get('accuracy', 0)
            self.logger.info("Model Accuracy: %.2f%%", accuracy * 100)
        except Exception as e:
            self.logger.error("Error evaluating model: %s", e)
            raise

    def save_model(self, model_path='catboost_model.cbm'):
        """
        Saves the trained CatBoost model to disk.

        Parameters:
        - model_path (str): File path to save the model.
        """
        self.logger.info("Saving CatBoost model to %s.", model_path)
        try:
            if self.model is not None:
                self.model.save_model(model_path)
                self.logger.info("Model saved to %s.", model_path)
            else:
                self.logger.warning("No model to save.")
        except Exception as e:
            self.logger.error("Error saving model: %s", e)
            raise
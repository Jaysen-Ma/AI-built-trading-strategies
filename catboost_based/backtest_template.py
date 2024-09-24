import pandas as pd
import backtrader as bt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
from data_store import DataStore
from data_preparer import DataPreparer
from catboost_strategy import CatBoostStrategy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during testing
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # You can add FileHandler here to log to a file
    ]
)

logger = logging.getLogger(__name__)

class CustomPandasData(bt.feeds.PandasData):
    # Add additional lines (columns from your DataFrame)
    lines = (
        'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 
        'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 
        'RSI_14', 'MACD_12_26_9_MACD_12_26_9',
    )
    
    # Define the default parameters for the additional columns
    params = (
        ('datetime', None),  # Use the DataFrame's DatetimeIndex
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'tick_volume'),  # Map 'volume' to 'tick_volume'
        ('openinterest', None),  # No openinterest column
        ('SMA_10', 'SMA_10'),
        ('SMA_20', 'SMA_20'),
        ('SMA_50', 'SMA_50'),
        ('SMA_100', 'SMA_100'),
        ('EMA_10', 'EMA_10'),
        ('EMA_20', 'EMA_20'),
        ('EMA_50', 'EMA_50'),
        ('EMA_100', 'EMA_100'),
        ('RSI_14', 'RSI_14'),
        ('MACD_12_26_9_MACD_12_26_9', 'MACD_12_26_9_MACD_12_26_9'),
    )

def run_backtest(data_store:DataStore):
    logger.info("Starting backtest process.")
    try:
        # Initialize data preparation
        data_preparer = DataPreparer(
            data_store=data_store,
            symbol='BTCUSD',  # Change to 'EURUSD' if needed
            target_lookahead=10,
            risk_percentage=0.0003,  # 0.03%
            reward_percentage=0.0006,  # 0.06%
            lot_size=0.01
        )
        
        # Prepare the data
        data_preparer.prepare()
        
        # Train and evaluate the model
        data_preparer.train_model(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass'
        )
        data_preparer.evaluate_model()
        
        # Save the model and scalers
        data_preparer.save_model(model_path='catboost_model.cbm')
        data_preparer.save_scalers(train_scaler_path='scaler_train.pkl')
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        logger.debug("Initialized Cerebro engine.")
        
        # Add strategy
        cerebro.addstrategy(CatBoostStrategy, 
                            model_path='catboost_model.cbm',
                            train_scaler_path='scaler_train.pkl',
                            lookahead=10,
                            risk_pct=0.0004,
                            reward_pct=0.00075,
                            lot_size=1,
                            printlog=True)
        logger.debug("Added CatBoostStrategy to Cerebro.")
        
        # Retrieve the raw data from DataStore
        raw_data = data_store.retrieve_data('BTCUSD')  # Change symbol if needed
        logger.debug("Loaded raw data for backtesting.")
        
        # Verify the index
        logger.info("Verifying DataFrame Index.")
        logger.debug("DataFrame Index: %s", raw_data.index)
        logger.debug("Index Type: %s", type(raw_data.index))
        
        # Identify the last date
        last_datetime = raw_data.index.max()
        last_date = last_datetime.date()
        logger.info("Last date in dataset: %s", last_date)
        
        # Define the start and end datetime for the last date
        end_datetime = pd.Timestamp(last_date)
        start_datetime = end_datetime - pd.Timedelta(hours=3) - pd.Timedelta(seconds=1)
        
        # Filter the DataFrame for the last date
        filtered_data = raw_data.loc[start_datetime:end_datetime]
        logger.info("Filtered data for backtest: %s to %s", start_datetime, end_datetime)
        logger.info("Filtered data shape: %s", filtered_data.shape)

        # Optional: Remove duplicates
        if filtered_data.index.duplicated().any():
            logger.warning("Duplicate datetime indices found. Removing duplicates.")
            filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]
            logger.info("Duplicates removed. New data shape: %s", filtered_data.shape)
        
        # Handle missing values if necessary
        if filtered_data.isnull().any().any():
            logger.warning("Missing values found in feature columns. Handling them by dropping.")
            filtered_data.dropna(inplace=True)
            logger.info("Missing values dropped. New data shape: %s", filtered_data.shape)
        
        required_columns = [
            'open', 'high', 'low', 'close',
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',
            'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
            'RSI_14', 'MACD_12_26_9_MACD_12_26_9'
            ]
        missing_columns = [col for col in required_columns if col not in filtered_data.columns]
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            raise ValueError(f"Missing columns in data: {missing_columns}")

        # Clean volume data
        if not np.isfinite(filtered_data['tick_volume']).all():
            logger.error("Volume data contains NaN or Inf values.")
            filtered_data['tick_volume'].replace([np.inf, -np.inf], np.nan, inplace=True)
            filtered_data['tick_volume'].fillna(0, inplace=True)  # Example handling
            logger.info("Volume data cleaned.")

        # Convert to Backtrader-compatible format with fromdate and todate
        data_bt = CustomPandasData(
            dataname=filtered_data,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='tick_volume',  # Map 'volume' to 'tick_volume'
            SMA_10='SMA_10',  # Use the column name as the value
            SMA_20='SMA_20',
            SMA_50='SMA_50',
            SMA_100='SMA_100',
            EMA_10='EMA_10',
            EMA_20='EMA_20',
            EMA_50='EMA_50',
            EMA_100='EMA_100',
            RSI_14='RSI_14',
            MACD_12_26_9_MACD_12_26_9='MACD_12_26_9_MACD_12_26_9',
            openinterest=None,
            fromdate=start_datetime,  # Set fromdate
            todate=end_datetime        # Set todate
        )
        logger.debug("Backtrader data feed created.")
        
        cerebro.adddata(data_bt)
        logger.debug("Data feed added to Cerebro.")
        
        # Set initial cash
        cerebro.broker.setcash(100000.0)
        logger.info("Initial portfolio value set to $100,000.00.")
        
        # Set commission (assuming spread is handled in strategy)
        cerebro.broker.setcommission(commission=0.0002)
        logger.debug("Commission set to 0.0002.")
        
        # Print starting portfolio value
        starting_value = cerebro.broker.getvalue()
        logger.info('Starting Portfolio Value: $%.2f', starting_value)
        
        # Run the backtest
        logger.info("Running the backtest.")
        cerebro.run()
        logger.info("Backtest completed.")
        
        # Print final portfolio value
        final_value = cerebro.broker.getvalue()
        logger.info('Final Portfolio Value: $%.2f', final_value)
        
        # After filtering and cleaning data
        if not np.isfinite(filtered_data).all().all():
            logger.error("Data contains NaN or Inf values. Cleaning data before plotting.")
            filtered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            filtered_data.dropna(inplace=True)
            logger.info("Data cleaned. New data shape: %s", filtered_data.shape)
        else:
            logger.info("Data contains only finite values.")

        # Plot the results
        logger.info("Plotting and saving backtest results.")
        fig = cerebro.plot()[0][0]
        fig.savefig('backtest_results.png')
        logger.info("Backtest results saved as 'backtest_results.png'.")
    except Exception as e:
        logger.error("An error occurred during the backtest: %s", e)
        raise

if __name__ == "__main__":
    run_backtest(DataStore('symbol_specific'))

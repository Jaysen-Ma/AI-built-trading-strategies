import pandas as pd
import backtrader as bt
import joblib
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)

class CatBoostStrategy(bt.Strategy):
    params = (
        ('model_path', 'catboost_model.cbm'),
        ('train_scaler_path', 'scaler_train.pkl'),  # Use the training scaler
        ('lookahead', 10),  # Minutes to look ahead
        ('risk_pct', 0.0003),
        ('reward_pct', 0.0006),
        ('lot_size', 0.01),
        ('printlog', True),
        ('order_interval', 4),  # Interval in minutes between orders
        ('stop_loss', 0.0003),    # 0.1% stop-loss
        ('take_profit', 0.0006),  # 0.2% take-profit
    )

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = CatBoostClassifier()
        self.model.load_model(self.params.model_path)
        self.scaler = joblib.load(self.params.train_scaler_path)
        self.history = []
        self.orders = []  # To track multiple orders
        self.last_order_time = None  # To track the last order time
        self.active_trade = None  # Track active trade

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.logger.info("BUY order completed at price: %.5f", order.executed.price)
            elif order.issell():
                self.logger.info("SELL order completed at price: %.5f", order.executed.price)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            action = 'BUY' if order.isbuy() else 'SELL'
            self.logger.warning(
                f"Order {action} rejected/canceled/margin: "
                f"Type: {order.exectype}, Size: {order.size}, Price: {order.created.price}"
            )

        # Remove the order from the tracking list
        if order in self.orders:
            self.orders.remove(order)

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            self.logger.info('%s %s', dt.isoformat(), txt)

    def next(self):
        current_datetime = self.datas[0].datetime.datetime(0)
        self.logger.info("Current datetime: %s", current_datetime)
        # Append current data to history
        current_data = {
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'SMA_10': self.data.SMA_10[0],
            'SMA_20': self.data.SMA_20[0],
            'SMA_50': self.data.SMA_50[0],
            'SMA_100': self.data.SMA_100[0],
            'EMA_10': self.data.EMA_10[0],
            'EMA_20': self.data.EMA_20[0],
            'EMA_50': self.data.EMA_50[0],
            'EMA_100': self.data.EMA_100[0],
            'RSI_14': self.data.RSI_14[0], 
            'MACD_12_26_9_MACD_12_26_9': self.data.MACD_12_26_9_MACD_12_26_9[0],
        }
        self.history.append(current_data)
        self.logger.debug("Appended new data to history: %s", current_data)

        
        # Create a DataFrame from history
        df = pd.DataFrame(self.history)
        
        # Feature Selection
        features = [
            'open', 'high', 'low', 'close',
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',
            'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
            'RSI_14', 'MACD_12_26_9_MACD_12_26_9'
        ]
    
        
        # Extract the latest feature set as a DataFrame
        latest_features = df[features].iloc[[-1]]
        
        # Check for NaN values in features
        if latest_features.isnull().any().any():
            self.logger.warning("NaN values found in features. Skipping prediction.")
            return
        
        # Normalize features using the training scaler
        try:
            scaled_features = self.scaler.transform(latest_features)
        except Exception as e:
            self.logger.error(f"Error during feature scaling: {e}")
            return
        
        # Predict the class
        try:
            prediction = self.model.predict(scaled_features)[0]
            self.logger.info("Prediction made: %d", prediction)
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return
        
        # Implement trading logic based on prediction
        # Check if enough time has passed since the last order
        if self.last_order_time is None or (current_datetime - self.last_order_time).total_seconds() >= self.params.order_interval * 60:
            if prediction == 0:  # Bullish
                self.logger.debug("Bullish prediction detected.")
                # Create a Market Buy Order without specifying price
                buy_order = self.buy(size=self.params.lot_size, exectype=bt.Order.Market)
                self.current_order = buy_order  # Track the current order for stop loss
                self.last_order_time = current_datetime
                self.log('BUY ORDER CREATED')
                self.logger.info("BUY order created at market price.")
                
            elif prediction == 1:  # Bearish
                self.logger.debug("Bearish prediction detected.")
                # Create a Market Sell Order without specifying price
                sell_order = self.sell(size=self.params.lot_size, exectype=bt.Order.Market)
                self.current_order = sell_order  # Track the current order for stop loss
                self.last_order_time = current_datetime
                self.log('SELL ORDER CREATED')
                self.logger.info("SELL order created at market price.")
            else:
                # Neutral prediction; do nothing
                self.logger.debug("Neutral prediction detected. No action taken.")
        else:
            minutes_since_last_order = (current_datetime - self.last_order_time).total_seconds() / 60
            self.logger.debug(
                "Order interval not reached. Time since last order: %.2f minutes.", 
                minutes_since_last_order
            )
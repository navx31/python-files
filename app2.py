import sys
import os
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtWebEngineWidgets import *
from PyQt6.QtWebEngineCore import *
from PyQt6.QtWebChannel import QWebChannel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict
import threading
import time
import hashlib
import requests
from queue import Queue
import websocket
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Try to import TA-Lib, but provide fallbacks if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not found. Using fallback calculations for technical indicators.")

# ----------------------------------------------------------
# Data Classes & Configuration
# ----------------------------------------------------------

@dataclass
class StockInfo:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    pe_ratio: float
    sector: str
    last_updated: datetime.datetime

@dataclass
class PortfolioHolding:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0

class CacheManager:
    """Manages offline data caching"""
    def __init__(self, cache_dir="stock_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
    def get_cache_key(self, symbol, data_type, period="1d"):
        return f"{symbol}_{data_type}_{period}_{datetime.date.today()}"
    
    def save_data(self, key, data):
        cache_file = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'data': data,
                'timestamp': datetime.datetime.now()
            }, f)
    
    def load_data(self, key, max_age_hours=24):
        cache_file = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                age = (datetime.datetime.now() - cached['timestamp']).total_seconds() / 3600
                if age < max_age_hours:
                    return cached['data']
        return None
    
    def clear_old_cache(self, days=7):
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.getmtime(filepath) < cutoff.timestamp():
                os.remove(filepath)

class TechnicalIndicators:
    """Calculates technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return np.array([np.nan] * len(prices))
        if TALIB_AVAILABLE:
            try:
                return talib.RSI(prices, timeperiod=period)
            except:
                pass
        # Fallback calculation
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return np.array([np.nan] * len(prices))
        if TALIB_AVAILABLE:
            try:
                macd, signal_line, hist = talib.MACD(prices, fastperiod=fast, 
                                                     slowperiod=slow, signalperiod=signal)
                return macd, signal_line, hist
            except:
                pass
        # Fallback calculation
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        if len(prices) < period:
            return np.array([np.nan] * len(prices))
        if TALIB_AVAILABLE:
            try:
                upper, middle, lower = talib.BBANDS(prices, timeperiod=period, 
                                                    nbdevup=std_dev, nbdevdn=std_dev)
                return upper, middle, lower
            except:
                pass
        # Fallback calculation
        sma = TechnicalIndicators.calculate_sma(prices, period)
        std = pd.Series(prices).rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_sma(prices, period):
        if TALIB_AVAILABLE:
            try:
                return talib.SMA(prices, timeperiod=period)
            except:
                pass
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def calculate_ema(prices, period):
        if TALIB_AVAILABLE:
            try:
                return talib.EMA(prices, timeperiod=period)
            except:
                pass
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

# ----------------------------------------------------------
# Real-Time Data Streamer
# ----------------------------------------------------------

class RealTimeStreamer(QThread):
    """Handles real-time data streaming"""
    data_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.symbols = []
        self.running = False
        self.websocket = None
        self.update_interval = 5000  # 5 seconds
        self.cache = CacheManager()
        
    def add_symbol(self, symbol):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol):
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def run(self):
        self.running = True
        while self.running and self.symbols:
            for symbol in self.symbols:
                try:
                    data = self.fetch_real_time_data(symbol)
                    if data:
                        self.data_received.emit(data)
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
            self.msleep(self.update_interval)
    
    def fetch_real_time_data(self, symbol):
        """Fetch real-time data (simulated for demo, replace with actual API)"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                prev_close = info.get('previousClose', 0)
                current_price = latest['Close']
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0
                
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': latest.get('Volume', 0),
                    'timestamp': datetime.datetime.now().isoformat()
                }
        except:
            pass
        return None
    
    def stop(self):
        self.running = False
        self.wait()

# ----------------------------------------------------------
# Portfolio Manager
# ----------------------------------------------------------

class PortfolioManager:
    """Manages portfolio holdings and calculations"""
    
    def __init__(self):
        self.holdings = {}
        self.cash = 100000.00  # Starting cash
        self.transactions = []
        self.total_value = self.cash  # Initialize total_value
        self.load_portfolio()
    
    def add_transaction(self, symbol, action, quantity, price, fee=0):
        """Add a buy/sell transaction"""
        timestamp = datetime.datetime.now()
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,  # 'buy' or 'sell'
            'quantity': quantity,
            'price': price,
            'fee': fee,
            'total': (quantity * price) + fee
        }
        
        self.transactions.append(transaction)
        
        if action == 'buy':
            self.cash -= transaction['total']
            if symbol in self.holdings:
                holding = self.holdings[symbol]
                total_quantity = holding.quantity + quantity
                total_cost = (holding.avg_cost * holding.quantity) + (quantity * price)
                holding.avg_cost = total_cost / total_quantity
                holding.quantity = total_quantity
            else:
                self.holdings[symbol] = PortfolioHolding(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price
                )
        else:  # sell
            if symbol in self.holdings:
                holding = self.holdings[symbol]
                if quantity <= holding.quantity:
                    self.cash += transaction['total']
                    holding.quantity -= quantity
                    if holding.quantity == 0:
                        del self.holdings[symbol]
        
        self.update_portfolio_value()
        self.save_portfolio()
    
    def update_prices(self, price_data):
        """Update current prices for all holdings"""
        for symbol, holding in self.holdings.items():
            if symbol in price_data:
                holding.current_price = price_data[symbol]
                holding.profit_loss = (holding.current_price - holding.avg_cost) * holding.quantity
                holding.profit_loss_percent = ((holding.current_price - holding.avg_cost) / 
                                              holding.avg_cost * 100) if holding.avg_cost > 0 else 0
        
        self.update_portfolio_value()
    
    def update_portfolio_value(self):
        """Calculate total portfolio value"""
        self.total_value = self.cash
        for holding in self.holdings.values():
            self.total_value += holding.current_price * holding.quantity
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        # Ensure portfolio value is updated
        self.update_portfolio_value()
        
        holdings_value = sum(h.current_price * h.quantity for h in self.holdings.values())
        total_invested = sum(h.avg_cost * h.quantity for h in self.holdings.values())
        total_pl = sum(h.profit_loss for h in self.holdings.values())
        total_pl_percent = (total_pl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'cash': self.cash,
            'holdings_value': holdings_value,
            'total_value': self.total_value,
            'total_invested': total_invested,
            'total_pl': total_pl,
            'total_pl_percent': total_pl_percent,
            'holdings_count': len(self.holdings)
        }
    
    def save_portfolio(self):
        """Save portfolio to file"""
        data = {
            'holdings': self.holdings,
            'cash': self.cash,
            'transactions': self.transactions,
            'total_value': self.total_value
        }
        with open('portfolio.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    def load_portfolio(self):
        """Load portfolio from file"""
        try:
            with open('portfolio.pkl', 'rb') as f:
                data = pickle.load(f)
                self.holdings = data.get('holdings', {})
                self.cash = data.get('cash', 100000.00)
                self.transactions = data.get('transactions', [])
                self.total_value = data.get('total_value', self.cash)
                
                # Update current prices and portfolio value
                self.update_portfolio_value()
        except FileNotFoundError:
            self.holdings = {}
            self.cash = 100000.00
            self.transactions = []
            self.total_value = self.cash

# ----------------------------------------------------------
# News Aggregator
# ----------------------------------------------------------

class NewsAggregator:
    """Aggregates financial news from multiple sources"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.sources = [
            ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
            ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
            ("Reuters", "http://feeds.reuters.com/reuters/topNews"),
            ("Bloomberg", "https://www.bloomberg.com/feeds/podcasts/etf_report.xml")
        ]
    
    def fetch_news(self, symbol=None, limit=20):
        """Fetch news for a symbol or general market news"""
        cache_key = f"news_{symbol if symbol else 'general'}"
        cached = self.cache.load_data(cache_key, max_age_hours=1)
        if cached:
            return cached
        
        news_items = []
        
        # For demo, simulate news items
        if symbol:
            # Symbol-specific news
            news_items.extend([
                {
                    'title': f'{symbol} Reports Strong Quarterly Earnings',
                    'source': 'Financial Times',
                    'time': '2 hours ago',
                    'url': f'https://example.com/news/{symbol.lower()}-earnings',
                    'summary': f'{symbol} exceeded analyst expectations with revenue growth of 15%.'
                },
                {
                    'title': f'Analysts Raise Price Target for {symbol}',
                    'source': 'Bloomberg',
                    'time': '5 hours ago',
                    'url': f'https://example.com/news/{symbol.lower()}-target',
                    'summary': f'Multiple analysts have increased their price targets for {symbol}.'
                }
            ])
        
        # General market news
        news_items.extend([
            {
                'title': 'Federal Reserve Holds Interest Rates Steady',
                'source': 'Reuters',
                'time': '3 hours ago',
                'url': 'https://example.com/news/fed-rates',
                'summary': 'The Federal Reserve announced it will maintain current interest rates.'
            },
            {
                'title': 'Tech Stocks Rally on AI Optimism',
                'source': 'CNBC',
                'time': '6 hours ago',
                'url': 'https://example.com/news/tech-rally',
                'summary': 'Technology stocks led the market higher amid renewed AI enthusiasm.'
            }
        ])
        
        self.cache.save_data(cache_key, news_items[:limit])
        return news_items[:limit]

# ----------------------------------------------------------
# Browser Tabs with Enhanced Features
# ----------------------------------------------------------

class StockBrowserTab(QWidget):
    """Enhanced stock browser with technical analysis"""
    
    def __init__(self, symbol=None, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.indicators = ['RSI', 'MACD', 'BBANDS', 'SMA', 'EMA']
        self.selected_indicators = []
        self.cache = CacheManager()
        self.init_ui()
        
        if symbol:
            self.load_stock_data(symbol)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QHBoxLayout()
        
        # Symbol input
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol...")
        if self.symbol:
            self.symbol_input.setText(self.symbol)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.on_load_clicked)
        
        # Time period selector
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
        self.period_combo.setCurrentText("1y")
        
        # Interval selector
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
        self.interval_combo.setCurrentText("1d")
        
        # Indicators selector
        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems(["Select Indicator..."] + self.indicators)
        self.add_indicator_btn = QPushButton("Add")
        self.add_indicator_btn.clicked.connect(self.add_indicator)
        
        # Selected indicators list
        self.selected_list = QListWidget()
        
        control_panel.addWidget(QLabel("Symbol:"))
        control_panel.addWidget(self.symbol_input)
        control_panel.addWidget(self.load_btn)
        control_panel.addWidget(QLabel("Period:"))
        control_panel.addWidget(self.period_combo)
        control_panel.addWidget(QLabel("Interval:"))
        control_panel.addWidget(self.interval_combo)
        control_panel.addWidget(QLabel("Indicators:"))
        control_panel.addWidget(self.indicator_combo)
        control_panel.addWidget(self.add_indicator_btn)
        
        layout.addLayout(control_panel)
        layout.addWidget(QLabel("Selected Indicators:"))
        layout.addWidget(self.selected_list)
        
        # Web view for charts
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Stats panel
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            background: #1a1a1a;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
        """)
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def on_load_clicked(self):
        symbol = self.symbol_input.text().strip().upper()
        if symbol:
            self.symbol = symbol
            self.load_stock_data(symbol)
    
    def load_stock_data(self, symbol):
        """Load stock data with caching"""
        period = self.period_combo.currentText()
        interval = self.interval_combo.currentText()
        
        # Try cache first
        cache_key = f"{symbol}_chart_{period}_{interval}"
        cached_data = self.cache.load_data(cache_key, max_age_hours=24)
        
        if cached_data:
            self.display_stock_data(cached_data)
        else:
            # Fetch new data
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period, interval=interval)
                info = stock.info
                
                if not hist.empty:
                    data = {
                        'hist': hist,
                        'info': info,
                        'period': period,
                        'interval': interval
                    }
                    self.cache.save_data(cache_key, data)
                    self.display_stock_data(data)
                else:
                    QMessageBox.warning(self, "Error", f"No data found for {symbol}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load {symbol}: {str(e)}")
    
    def display_stock_data(self, data):
        """Display stock data with charts"""
        hist = data['hist']
        info = data['info']
        
        # Create interactive chart with Plotly
        fig = self.create_stock_chart(hist, info)
        
        # Update stats
        stats_html = self.generate_stats_html(info, hist)
        self.stats_label.setText(stats_html)
        
        # Display chart
        self.web_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
    
    def create_stock_chart(self, hist, info):
        """Create interactive stock chart with indicators"""
        try:
            # Use Plotly to create chart
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Determine if we need subplots for indicators
            needs_rsi_subplot = 'RSI' in self.selected_indicators
            needs_macd_subplot = 'MACD' in self.selected_indicators
            
            if needs_rsi_subplot or needs_macd_subplot:
                # Create subplots for indicators
                rows = 2  # Price + Volume
                if needs_rsi_subplot:
                    rows += 1
                if needs_macd_subplot:
                    rows += 1
                
                row_heights = [0.6] + [0.2] * (rows - 1)
                subplot_titles = ["Price"]
                
                if needs_rsi_subplot:
                    subplot_titles.append("RSI")
                if needs_macd_subplot:
                    subplot_titles.append("MACD")
                subplot_titles.append("Volume")
                
                fig = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=row_heights,
                    subplot_titles=subplot_titles
                )
                
                # Add price chart
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price',
                    increasing_line_color='#4CAF50',
                    decreasing_line_color='#f44336'
                ), row=1, col=1)
                
                current_row = 2
                
                # Add RSI if selected
                if needs_rsi_subplot and len(hist) >= 14:
                    prices = hist['Close'].values
                    rsi = TechnicalIndicators.calculate_rsi(prices)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=rsi,
                        name='RSI(14)',
                        line=dict(color='purple', width=1)
                    ), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                    current_row += 1
                
                # Add MACD if selected
                if needs_macd_subplot and len(hist) >= 26:
                    prices = hist['Close'].values
                    macd, signal, hist_macd = TechnicalIndicators.calculate_macd(prices)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=macd,
                        name='MACD',
                        line=dict(color='blue', width=1)
                    ), row=current_row, col=1)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=signal,
                        name='Signal',
                        line=dict(color='red', width=1)
                    ), row=current_row, col=1)
                    
                    colors = ['#4CAF50' if h >= 0 else '#f44336' for h in hist_macd]
                    fig.add_trace(go.Bar(
                        x=hist.index,
                        y=hist_macd,
                        name='MACD Hist',
                        marker_color=colors
                    ), row=current_row, col=1)
                    current_row += 1
                
                # Add volume
                colors = ['#4CAF50' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else '#f44336' 
                         for i in range(len(hist))]
                
                fig.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color=colors
                ), row=current_row, col=1)
                
            else:
                # Simple price chart without subplots
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price',
                    increasing_line_color='#4CAF50',
                    decreasing_line_color='#f44336'
                ))
                
                # Add selected indicators
                prices = hist['Close'].values
                
                if 'SMA' in self.selected_indicators and len(prices) >= 20:
                    sma20 = TechnicalIndicators.calculate_sma(prices, 20)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=sma20,
                        name='SMA(20)',
                        line=dict(color='orange', width=1)
                    ))
                
                if 'EMA' in self.selected_indicators and len(prices) >= 12:
                    ema12 = TechnicalIndicators.calculate_ema(prices, 12)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=ema12,
                        name='EMA(12)',
                        line=dict(color='cyan', width=1)
                    ))
                
                if 'BBANDS' in self.selected_indicators and len(prices) >= 20:
                    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(prices)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=upper,
                        name='BB Upper',
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=lower,
                        name='BB Lower',
                        fill='tonexty',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ))
            
            fig.update_layout(
                title=f"{self.symbol} - Stock Analysis",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            # Return a simple figure if Plotly fails
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
            fig.update_layout(title=f"{self.symbol} - Price", template="plotly_dark")
            return fig
    
    def generate_stats_html(self, info, hist):
        """Generate statistics HTML"""
        if hist.empty:
            return "No data available"
        
        latest = hist.iloc[-1]
        prev_close = info.get('previousClose', 0)
        current_price = latest['Close']
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close else 0
        
        stats = f"""
        <b>{self.symbol}</b> | ${current_price:.2f} | 
        <span style='color: {'#4CAF50' if change >= 0 else '#f44336'}'>
        {change:+.2f} ({change_percent:+.2f}%)
        </span> |
        Volume: {latest.get('Volume', 0):,} |
        Market Cap: ${info.get('marketCap', 0):,.0f} |
        PE: {info.get('trailingPE', 'N/A')}
        """
        return stats
    
    def add_indicator(self):
        indicator = self.indicator_combo.currentText()
        if indicator != "Select Indicator..." and indicator not in self.selected_indicators:
            self.selected_indicators.append(indicator)
            self.selected_list.addItem(indicator)
            if self.symbol:
                self.load_stock_data(self.symbol)

class PortfolioTab(QWidget):
    """Portfolio management tab"""
    
    def __init__(self):
        super().__init__()
        self.portfolio = PortfolioManager()
        self.init_ui()
        # Don't update display here, wait for UI to be fully initialized
        QTimer.singleShot(100, self.update_display)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Portfolio summary
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("""
            background: #2a2a2a;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
        """)
        layout.addWidget(self.summary_label)
        
        # Add transaction form
        form_group = QGroupBox("Add Transaction")
        form_layout = QFormLayout()
        
        self.tx_symbol = QLineEdit()
        self.tx_symbol.setPlaceholderText("AAPL")
        
        self.tx_action = QComboBox()
        self.tx_action.addItems(["Buy", "Sell"])
        
        self.tx_quantity = QDoubleSpinBox()
        self.tx_quantity.setRange(0.01, 10000)
        self.tx_quantity.setValue(1)
        
        self.tx_price = QDoubleSpinBox()
        self.tx_price.setRange(0.01, 10000)
        self.tx_price.setValue(150)
        
        self.tx_fee = QDoubleSpinBox()
        self.tx_fee.setRange(0, 1000)
        self.tx_fee.setValue(0)
        
        self.add_tx_btn = QPushButton("Execute")
        self.add_tx_btn.clicked.connect(self.execute_transaction)
        
        form_layout.addRow("Symbol:", self.tx_symbol)
        form_layout.addRow("Action:", self.tx_action)
        form_layout.addRow("Quantity:", self.tx_quantity)
        form_layout.addRow("Price:", self.tx_price)
        form_layout.addRow("Fee:", self.tx_fee)
        form_layout.addRow(self.add_tx_btn)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Holdings table
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(7)
        self.holdings_table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Avg Cost", "Current", "Value", "P/L", "P/L %"
        ])
        self.holdings_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Holdings:"))
        layout.addWidget(self.holdings_table)
        
        # Transactions table
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(6)
        self.transactions_table.setHorizontalHeaderLabels([
            "Date", "Symbol", "Action", "Qty", "Price", "Total"
        ])
        layout.addWidget(QLabel("Transaction History:"))
        layout.addWidget(self.transactions_table)
        
        # Update button
        update_btn = QPushButton("Update Prices")
        update_btn.clicked.connect(self.update_prices)
        layout.addWidget(update_btn)
        
        self.setLayout(layout)
    
    def execute_transaction(self):
        symbol = self.tx_symbol.text().strip().upper()
        action = self.tx_action.currentText().lower()
        quantity = self.tx_quantity.value()
        price = self.tx_price.value()
        fee = self.tx_fee.value()
        
        if not symbol:
            QMessageBox.warning(self, "Error", "Please enter a symbol")
            return
        
        # Validate sell transaction
        if action == 'sell' and symbol not in self.portfolio.holdings:
            QMessageBox.warning(self, "Error", f"You don't own any {symbol}")
            return
        
        if action == 'sell':
            holding = self.portfolio.holdings[symbol]
            if quantity > holding.quantity:
                QMessageBox.warning(self, "Error", 
                                  f"Only {holding.quantity} shares available")
                return
        
        self.portfolio.add_transaction(symbol, action, quantity, price, fee)
        self.update_display()
        
        # Clear form
        self.tx_symbol.clear()
        self.tx_quantity.setValue(1)
        self.tx_price.setValue(150)
        self.tx_fee.setValue(0)
    
    def update_prices(self):
        """Update current prices for all holdings"""
        symbols = list(self.portfolio.holdings.keys())
        if not symbols:
            return
        
        price_data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                price_data[symbol] = price
            except:
                continue
        
        self.portfolio.update_prices(price_data)
        self.update_display()
    
    def update_display(self):
        """Update all display elements"""
        # Update summary
        summary = self.portfolio.get_portfolio_summary()
        summary_text = f"""
        <b>Portfolio Summary</b><br>
        Cash: ${summary['cash']:,.2f} | 
        Holdings: ${summary['holdings_value']:,.2f} | 
        Total: ${summary['total_value']:,.2f}<br>
        Invested: ${summary['total_invested']:,.2f} | 
        P/L: <span style='color: {'#4CAF50' if summary['total_pl'] >= 0 else '#f44336'}'>
        ${summary['total_pl']:,.2f} ({summary['total_pl_percent']:+.2f}%)</span>
        """
        self.summary_label.setText(summary_text)
        
        # Update holdings table
        self.holdings_table.setRowCount(len(self.portfolio.holdings))
        for i, (symbol, holding) in enumerate(self.portfolio.holdings.items()):
            value = holding.current_price * holding.quantity
            pl_color = "#4CAF50" if holding.profit_loss >= 0 else "#f44336"
            
            self.holdings_table.setItem(i, 0, QTableWidgetItem(symbol))
            self.holdings_table.setItem(i, 1, QTableWidgetItem(f"{holding.quantity:.2f}"))
            self.holdings_table.setItem(i, 2, QTableWidgetItem(f"${holding.avg_cost:.2f}"))
            self.holdings_table.setItem(i, 3, QTableWidgetItem(f"${holding.current_price:.2f}"))
            self.holdings_table.setItem(i, 4, QTableWidgetItem(f"${value:,.2f}"))
            
            pl_item = QTableWidgetItem(f"${holding.profit_loss:,.2f}")
            pl_item.setForeground(QColor(pl_color))
            self.holdings_table.setItem(i, 5, pl_item)
            
            pl_percent_item = QTableWidgetItem(f"{holding.profit_loss_percent:+.2f}%")
            pl_percent_item.setForeground(QColor(pl_color))
            self.holdings_table.setItem(i, 6, pl_percent_item)
        
        # Update transactions table
        self.transactions_table.setRowCount(len(self.portfolio.transactions))
        for i, tx in enumerate(reversed(self.portfolio.transactions[-20:])):  # Last 20
            self.transactions_table.setItem(i, 0, 
                QTableWidgetItem(tx['timestamp'].strftime("%Y-%m-%d %H:%M")))
            self.transactions_table.setItem(i, 1, QTableWidgetItem(tx['symbol']))
            self.transactions_table.setItem(i, 2, QTableWidgetItem(tx['action'].upper()))
            self.transactions_table.setItem(i, 3, QTableWidgetItem(f"{tx['quantity']:.2f}"))
            self.transactions_table.setItem(i, 4, QTableWidgetItem(f"${tx['price']:.2f}"))
            self.transactions_table.setItem(i, 5, QTableWidgetItem(f"${tx['total']:.2f}"))

class NewsTab(QWidget):
    """News aggregator tab"""
    
    def __init__(self):
        super().__init__()
        self.news_aggregator = NewsAggregator()
        self.current_symbol = None
        self.init_ui()
        self.load_general_news()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QHBoxLayout()
        
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol for specific news...")
        
        self.load_news_btn = QPushButton("Load Symbol News")
        self.load_news_btn.clicked.connect(self.load_symbol_news)
        
        self.general_news_btn = QPushButton("General News")
        self.general_news_btn.clicked.connect(self.load_general_news)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_news)
        
        control_panel.addWidget(QLabel("Symbol:"))
        control_panel.addWidget(self.symbol_input)
        control_panel.addWidget(self.load_news_btn)
        control_panel.addWidget(self.general_news_btn)
        control_panel.addWidget(self.refresh_btn)
        control_panel.addStretch()
        
        layout.addLayout(control_panel)
        
        # News list
        self.news_list = QListWidget()
        self.news_list.setSpacing(5)
        self.news_list.itemClicked.connect(self.on_news_item_clicked)
        layout.addWidget(self.news_list)
        
        self.setLayout(layout)
    
    def load_general_news(self):
        self.current_symbol = None
        self.symbol_input.clear()
        news_items = self.news_aggregator.fetch_news(limit=20)
        self.display_news(news_items)
    
    def load_symbol_news(self):
        symbol = self.symbol_input.text().strip().upper()
        if symbol:
            self.current_symbol = symbol
            news_items = self.news_aggregator.fetch_news(symbol, limit=20)
            self.display_news(news_items)
    
    def refresh_news(self):
        if self.current_symbol:
            self.load_symbol_news()
        else:
            self.load_general_news()
    
    def display_news(self, news_items):
        self.news_list.clear()
        
        for item in news_items:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            title = QLabel(f"<b>{item['title']}</b>")
            title.setWordWrap(True)
            title.setStyleSheet("color: white;")
            
            meta = QLabel(f"{item['source']} • {item['time']}")
            meta.setStyleSheet("color: #888; font-size: 11px;")
            
            summary = QLabel(item['summary'])
            summary.setWordWrap(True)
            summary.setStyleSheet("color: #ccc; font-size: 12px;")
            
            layout.addWidget(title)
            layout.addWidget(meta)
            layout.addWidget(summary)
            layout.setContentsMargins(10, 10, 10, 10)
            
            list_item = QListWidgetItem()
            list_item.setSizeHint(widget.sizeHint())
            list_item.setData(Qt.ItemDataRole.UserRole, item['url'])
            
            self.news_list.addItem(list_item)
            self.news_list.setItemWidget(list_item, widget)
    
    def on_news_item_clicked(self, item):
        url = item.data(Qt.ItemDataRole.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))

# ----------------------------------------------------------
# Main Application with All Features
# ----------------------------------------------------------

class QuantumBrowser(QMainWindow):
    """Main application with all enhanced features"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.cache_manager = CacheManager()
        self.streamer = RealTimeStreamer()
        self.streamer.data_received.connect(self.on_realtime_data)
        
        # Setup UI
        self.setWindowTitle("🚀 Quantum Browser - Advanced Stock & Web")
        self.setGeometry(100, 100, 1600, 900)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        self.init_ui()
        
        # Start real-time streaming after UI is initialized
        QTimer.singleShot(1000, self.start_realtime_streaming)
        
        # Load cached data
        self.load_cached_watchlist()
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        self.setPalette(dark_palette)
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create sidebar
        sidebar = self.create_sidebar()
        splitter.addWidget(sidebar)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        
        # Add default tabs
        self.add_stock_browser_tab()
        self.add_portfolio_tab()
        self.add_news_tab()
        
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([300, 1300])
        
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create offline indicator
        self.offline_label = QLabel("⚡ Online")
        self.offline_label.setStyleSheet("color: #4CAF50; padding: 2px 10px;")
        self.status_bar.addPermanentWidget(self.offline_label)
    
    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Navigation buttons
        nav_actions = [
            ("← Back", "Back", self.navigate_back),
            ("→ Forward", "Forward", self.navigate_forward),
            ("↻ Refresh", "Refresh", self.refresh_page),
            ("🏠 Home", "Home", self.go_home),
            ("➕ New Tab", "New Tab", self.add_stock_browser_tab)
        ]
        
        for text, tooltip, slot in nav_actions:
            action = QAction(text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(slot)
            toolbar.addAction(action)
        
        toolbar.addSeparator()
        
        # URL bar
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("Enter URL or stock symbol...")
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        toolbar.addWidget(QLabel("URL:"))
        toolbar.addWidget(self.url_bar)
        
        # Stock search
        self.stock_search = QLineEdit()
        self.stock_search.setPlaceholderText("Search stock...")
        self.stock_search.returnPressed.connect(self.search_stock)
        toolbar.addWidget(QLabel("Stock:"))
        toolbar.addWidget(self.stock_search)
        
        toolbar.addSeparator()
        
        # Offline mode toggle
        self.offline_toggle = QAction("🌐 Online", self)
        self.offline_toggle.setCheckable(True)
        self.offline_toggle.toggled.connect(self.toggle_offline_mode)
        toolbar.addAction(self.offline_toggle)
    
    def create_sidebar(self):
        """Create sidebar with watchlist and quick actions"""
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("""
            QWidget {
                background: #1a1a1a;
                color: white;
            }
            QListWidget {
                background: #2a2a2a;
                border: none;
                font-family: monospace;
            }
            QPushButton {
                background: #3a3a3a;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
            QGroupBox {
                color: white;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSpinBox, QLineEdit {
                background: #2a2a2a;
                color: white;
                border: 1px solid #444;
                padding: 3px;
                border-radius: 3px;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        
        # Watchlist section
        watchlist_group = QGroupBox("⭐ Watchlist")
        watchlist_layout = QVBoxLayout()
        
        self.watchlist = QListWidget()
        self.watchlist.itemDoubleClicked.connect(self.open_watchlist_stock)
        watchlist_layout.addWidget(self.watchlist)
        
        # Add to watchlist
        add_layout = QHBoxLayout()
        self.watchlist_input = QLineEdit()
        self.watchlist_input.setPlaceholderText("Add symbol...")
        add_layout.addWidget(self.watchlist_input)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_to_watchlist)
        add_layout.addWidget(add_btn)
        
        watchlist_layout.addLayout(add_layout)
        watchlist_group.setLayout(watchlist_layout)
        layout.addWidget(watchlist_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        
        actions = [
            ("📈 Market Overview", self.show_market_overview),
            ("💰 Top Gainers", lambda: self.show_top_stocks("gainers")),
            ("📉 Top Losers", lambda: self.show_top_stocks("losers")),
            ("📊 Screener", self.run_screener),
            ("📰 Financial News", self.add_news_tab),
            ("💼 My Portfolio", self.add_portfolio_tab)
        ]
        
        for text, slot in actions:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            actions_layout.addWidget(btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Real-time updates
        update_group = QGroupBox("Real-time Updates")
        update_layout = QVBoxLayout()
        
        self.update_interval = QSpinBox()
        self.update_interval.setRange(5, 300)
        self.update_interval.setValue(30)
        self.update_interval.setSuffix(" seconds")
        self.update_interval.valueChanged.connect(self.update_streaming_interval)
        
        update_layout.addWidget(QLabel("Update Interval:"))
        update_layout.addWidget(self.update_interval)
        
        self.realtime_toggle = QCheckBox("Enable Real-time")
        self.realtime_toggle.setChecked(True)
        self.realtime_toggle.toggled.connect(self.toggle_realtime)
        update_layout.addWidget(self.realtime_toggle)
        
        update_group.setLayout(update_layout)
        layout.addWidget(update_group)
        
        layout.addStretch()
        
        return sidebar
    
    def add_stock_browser_tab(self, symbol=None):
        """Add a new stock browser tab"""
        tab = StockBrowserTab(symbol)
        tab_name = f"📈 {symbol if symbol else 'Stock Browser'}"
        index = self.tab_widget.addTab(tab, tab_name)
        self.tab_widget.setCurrentIndex(index)
        return tab
    
    def add_portfolio_tab(self):
        """Add portfolio tab"""
        tab = PortfolioTab()
        index = self.tab_widget.addTab(tab, "💰 Portfolio")
        self.tab_widget.setCurrentIndex(index)
        return tab
    
    def add_news_tab(self):
        """Add news tab"""
        tab = NewsTab()
        index = self.tab_widget.addTab(tab, "📰 News")
        self.tab_widget.setCurrentIndex(index)
        return tab
    
    def close_tab(self, index):
        """Close a tab"""
        if self.tab_widget.count() > 1:
            self.tab_widget.removeTab(index)
    
    # ----------------------------------------------------------
    # Real-time Streaming Methods
    # ----------------------------------------------------------
    
    def start_realtime_streaming(self):
        """Start real-time data streaming"""
        if not self.streamer.isRunning():
            self.streamer.start()
    
    def toggle_realtime(self, enabled):
        """Toggle real-time updates"""
        if enabled:
            self.start_realtime_streaming()
        else:
            self.streamer.stop()
    
    def update_streaming_interval(self, interval):
        """Update streaming interval"""
        self.streamer.update_interval = interval * 1000  # Convert to milliseconds
    
    def on_realtime_data(self, data):
        """Handle incoming real-time data"""
        symbol = data['symbol']
        
        # Update watchlist if symbol is in it
        for i in range(self.watchlist.count()):
            item = self.watchlist.item(i)
            if symbol in item.text():
                price = data['price']
                change = data['change_percent']
                color = "#4CAF50" if change >= 0 else "#f44336"
                item.setText(f"{symbol}: ${price:.2f} ({change:+.2f}%)")
                item.setForeground(QColor(color))
                break
        
        # Update current tab if it's showing this symbol
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, StockBrowserTab) and current_widget.symbol == symbol:
            current_widget.load_stock_data(symbol)
    
    # ----------------------------------------------------------
    # Watchlist Methods
    # ----------------------------------------------------------
    
    def add_to_watchlist(self):
        """Add symbol to watchlist"""
        symbol = self.watchlist_input.text().strip().upper()
        if symbol:
            # Check if already in watchlist
            for i in range(self.watchlist.count()):
                if symbol in self.watchlist.item(i).text():
                    return
            
            # Add to list
            item = QListWidgetItem(f"{symbol}: Loading...")
            self.watchlist.addItem(item)
            self.watchlist_input.clear()
            
            # Add to real-time streamer
            self.streamer.add_symbol(symbol)
            
            # Save watchlist
            self.save_watchlist()
            
            # Fetch initial data
            threading.Thread(target=self.update_watchlist_item, args=(symbol,)).start()
    
    def update_watchlist_item(self, symbol):
        """Update watchlist item with current data"""
        try:
            # Try cache first for offline mode
            cache_key = f"{symbol}_watchlist"
            cached = self.cache_manager.load_data(cache_key, max_age_hours=1)
            
            if cached and self.offline_toggle.isChecked():
                price, change = cached
            else:
                # Fetch live data
                stock = yf.Ticker(symbol)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                prev_close = info.get('previousClose', 0)
                change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                
                # Cache the data
                self.cache_manager.save_data(cache_key, (current_price, change))
            
            # Update UI
            QMetaObject.invokeMethod(self, "update_watchlist_ui",
                                   Qt.ConnectionType.QueuedConnection,
                                   Q_ARG(str, symbol),
                                   Q_ARG(float, current_price),
                                   Q_ARG(float, change))
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
    
    def update_watchlist_ui(self, symbol, price, change):
        """Update watchlist UI thread-safely"""
        for i in range(self.watchlist.count()):
            item = self.watchlist.item(i)
            if symbol in item.text():
                color = "#4CAF50" if change >= 0 else "#f44336"
                item.setText(f"{symbol}: ${price:.2f} ({change:+.2f}%)")
                item.setForeground(QColor(color))
                break
    
    def open_watchlist_stock(self, item):
        """Open watchlist stock in new tab"""
        symbol = item.text().split(":")[0].strip()
        self.add_stock_browser_tab(symbol)
    
    def save_watchlist(self):
        """Save watchlist to cache"""
        symbols = []
        for i in range(self.watchlist.count()):
            text = self.watchlist.item(i).text()
            symbol = text.split(":")[0].strip()
            symbols.append(symbol)
        
        self.cache_manager.save_data("watchlist", symbols)
    
    def load_cached_watchlist(self):
        """Load watchlist from cache"""
        symbols = self.cache_manager.load_data("watchlist", max_age_hours=24)
        if symbols:
            for symbol in symbols:
                item = QListWidgetItem(f"{symbol}: Loading...")
                self.watchlist.addItem(item)
                self.streamer.add_symbol(symbol)
                
                # Update in background
                threading.Thread(target=self.update_watchlist_item, args=(symbol,)).start()
    
    # ----------------------------------------------------------
    # Navigation Methods
    # ----------------------------------------------------------
    
    def navigate_to_url(self):
        """Navigate to URL or search stock"""
        text = self.url_bar.text().strip()
        if not text:
            return
        
        # Check if it's a stock symbol
        if '.' not in text and '/' not in text and len(text) <= 10:
            self.search_stock()
        else:
            # It's a URL
            if not text.startswith(('http://', 'https://')):
                text = 'https://' + text
            
            # Open in new tab
            tab = StockBrowserTab()
            tab.web_view.setUrl(QUrl(text))
            index = self.tab_widget.addTab(tab, "Web")
            self.tab_widget.setCurrentIndex(index)
    
    def search_stock(self):
        """Search for a stock"""
        symbol = self.stock_search.text().strip().upper()
        if symbol:
            self.add_stock_browser_tab(symbol)
            self.stock_search.clear()
    
    def navigate_back(self):
        """Navigate back in current tab"""
        current = self.tab_widget.currentWidget()
        if isinstance(current, StockBrowserTab) and hasattr(current, 'web_view'):
            current.web_view.back()
    
    def navigate_forward(self):
        """Navigate forward in current tab"""
        current = self.tab_widget.currentWidget()
        if isinstance(current, StockBrowserTab) and hasattr(current, 'web_view'):
            current.web_view.forward()
    
    def refresh_page(self):
        """Refresh current page"""
        current = self.tab_widget.currentWidget()
        if isinstance(current, StockBrowserTab) and hasattr(current, 'web_view'):
            current.web_view.reload()
        elif isinstance(current, StockBrowserTab):
            current.load_stock_data(current.symbol)
    
    def go_home(self):
        """Go to home page"""
        self.add_stock_browser_tab()
    
    # ----------------------------------------------------------
    # Market Data Methods
    # ----------------------------------------------------------
    
    def show_market_overview(self):
        """Show market overview"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { background: #0a0a0a; color: white; font-family: Arial; padding: 20px; }
                .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
                .card { background: #1a1a1a; padding: 20px; border-radius: 10px; }
                .gain { color: #4CAF50; }
                .loss { color: #f44336; }
            </style>
        </head>
        <body>
            <h1>📊 Market Overview</h1>
            <div class='grid'>
                <div class='card'><b>S&P 500</b><br>4,567.89 <span class='gain'>↑ 0.85%</span></div>
                <div class='card'><b>NASDAQ</b><br>14,234.56 <span class='gain'>↑ 1.23%</span></div>
                <div class='card'><b>DOW</b><br>35,678.90 <span class='gain'>↑ 0.45%</span></div>
            </div>
            <h2>Market Trends</h2>
            <p>Technology sector leading gains, Energy sector under pressure.</p>
        </body>
        </html>
        """
        
        tab = StockBrowserTab()
        tab.web_view.setHtml(html)
        index = self.tab_widget.addTab(tab, "Market Overview")
        self.tab_widget.setCurrentIndex(index)
    
    def show_top_stocks(self, category):
        """Show top gainers/losers"""
        # This would be populated with real data in production
        if category == "gainers":
            stocks = [("AAPL", "+2.5%"), ("TSLA", "+4.2%"), ("NVDA", "+3.8%")]
            title = "Top Gainers"
        else:
            stocks = [("MSFT", "-1.2%"), ("GOOGL", "-0.8%"), ("AMZN", "-1.5%")]
            title = "Top Losers"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ background: #0a0a0a; color: white; font-family: Arial; padding: 20px; }}
                .stock {{ padding: 10px; margin: 5px 0; background: #1a1a1a; border-radius: 5px; }}
                .gain {{ color: #4CAF50; }}
                .loss {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {"".join(f'<div class="stock"><b>{s[0]}</b> <span class="{"gain" if "+" in s[1] else "loss"}">{s[1]}</span></div>' for s in stocks)}
        </body>
        </html>
        """
        
        tab = StockBrowserTab()
        tab.web_view.setHtml(html)
        index = self.tab_widget.addTab(tab, title)
        self.tab_widget.setCurrentIndex(index)
    
    def run_screener(self):
        """Run stock screener"""
        # This would run actual screening logic in production
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { background: #0a0a0a; color: white; font-family: Arial; padding: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
                th { background: #1a1a1a; }
            </style>
        </head>
        <body>
            <h1>🔍 Stock Screener Results</h1>
            <table>
                <tr><th>Symbol</th><th>Price</th><th>Change</th><th>PE</th><th>Sector</th></tr>
                <tr><td>AAPL</td><td>$175.00</td><td style="color:#4CAF50">+2.5%</td><td>28.5</td><td>Technology</td></tr>
                <tr><td>MSFT</td><td>$330.00</td><td style="color:#f44336">-1.2%</td><td>32.1</td><td>Technology</td></tr>
                <tr><td>JNJ</td><td>$155.00</td><td style="color:#4CAF50">+0.8%</td><td>22.3</td><td>Healthcare</td></tr>
            </table>
        </body>
        </html>
        """
        
        tab = StockBrowserTab()
        tab.web_view.setHtml(html)
        index = self.tab_widget.addTab(tab, "Screener")
        self.tab_widget.setCurrentIndex(index)
    
    # ----------------------------------------------------------
    # Offline Mode Methods
    # ----------------------------------------------------------
    
    def toggle_offline_mode(self, enabled):
        """Toggle offline mode"""
        if enabled:
            self.offline_label.setText("📴 Offline")
            self.offline_label.setStyleSheet("color: #f44336; padding: 2px 10px;")
            self.offline_toggle.setText("📴 Offline")
            
            # Stop real-time streaming
            self.realtime_toggle.setChecked(False)
            self.streamer.stop()
        else:
            self.offline_label.setText("⚡ Online")
            self.offline_label.setStyleSheet("color: #4CAF50; padding: 2px 10px;")
            self.offline_toggle.setText("🌐 Online")
            
            # Resume real-time streaming if enabled
            if self.realtime_toggle.isChecked():
                self.start_realtime_streaming()
    
    # ----------------------------------------------------------
    # Event Handlers
    # ----------------------------------------------------------
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop streaming
        self.streamer.stop()
        
        # Save watchlist
        self.save_watchlist()
        
        # Clear old cache
        self.cache_manager.clear_old_cache(days=7)
        
        event.accept()

# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application icon and name
    app.setApplicationName("Quantum Browser")
    app.setApplicationDisplayName("Quantum Browser - Advanced Stock & Web")
    
    # Create and show main window
    window = QuantumBrowser()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
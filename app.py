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
from dataclasses import dataclass
from typing import Optional, List
import asyncio
import threading

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

class BrowserTab(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().profile().downloadRequested.connect(self.on_download_requested)
        self.loadFinished.connect(self.on_load_finished)
        
    def on_download_requested(self, download):
        download.accept()
        
    def on_load_finished(self, ok):
        if ok:
            self.page().runJavaScript("""
                document.addEventListener('contextmenu', function(e) {
                    e.preventDefault();
                });
            """)

# ----------------------------------------------------------
# Main Window with Advanced Features
# ----------------------------------------------------------

class AdvancedStockBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Browser - Advanced Stock & Web Browser")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon("browser_icon.png"))
        
        self.current_tabs = {}
        self.tab_count = 0
        self.history = []
        self.bookmarks = []
        self.stock_watchlist = []
        
        self.init_ui()
        self.load_settings()
        self.show_welcome_screen()
        
    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main splitter (sidebar + content)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create sidebar
        self.sidebar = self.create_sidebar()
        self.main_splitter.addWidget(self.sidebar)
        
        # Create tab widget for browser
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.main_splitter.addWidget(self.tab_widget)
        
        main_layout.addWidget(self.main_splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create context menu
        self.create_context_menu()
        
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Navigation buttons
        self.back_btn = QAction(QIcon("←"), "Back", self)
        self.back_btn.triggered.connect(self.navigate_back)
        toolbar.addAction(self.back_btn)
        
        self.forward_btn = QAction(QIcon("→"), "Forward", self)
        self.forward_btn.triggered.connect(self.navigate_forward)
        toolbar.addAction(self.forward_btn)
        
        self.refresh_btn = QAction(QIcon("↻"), "Refresh", self)
        self.refresh_btn.triggered.connect(self.refresh_page)
        toolbar.addAction(self.refresh_btn)
        
        toolbar.addSeparator()
        
        # Home button
        self.home_btn = QAction(QIcon("🏠"), "Home", self)
        self.home_btn.triggered.connect(self.go_home)
        toolbar.addAction(self.home_btn)
        
        # New tab button
        self.new_tab_btn = QAction(QIcon("➕"), "New Tab", self)
        self.new_tab_btn.triggered.connect(self.create_new_tab)
        toolbar.addAction(self.new_tab_btn)
        
        toolbar.addSeparator()
        
        # URL Bar
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        self.url_bar.setPlaceholderText("Enter URL or stock symbol...")
        toolbar.addWidget(QLabel("URL:"))
        toolbar.addWidget(self.url_bar)
        
        # Go button
        self.go_btn = QAction(QIcon("🚀"), "Go", self)
        self.go_btn.triggered.connect(self.navigate_to_url)
        toolbar.addAction(self.go_btn)
        
        toolbar.addSeparator()
        
        # Stock search
        self.stock_search = QLineEdit()
        self.stock_search.setPlaceholderText("Search stock (AAPL, TSLA)...")
        self.stock_search.returnPressed.connect(self.search_stock)
        toolbar.addWidget(QLabel("Stock:"))
        toolbar.addWidget(self.stock_search)
        
        # Stock search button
        self.stock_btn = QAction(QIcon("📈"), "Stock", self)
        self.stock_btn.triggered.connect(self.search_stock)
        toolbar.addAction(self.stock_btn)
        
        toolbar.addSeparator()
        
        # Bookmarks
        self.bookmark_btn = QAction(QIcon("⭐"), "Bookmark", self)
        self.bookmark_btn.triggered.connect(self.add_bookmark)
        toolbar.addAction(self.bookmark_btn)
        
        # Downloads
        self.downloads_btn = QAction(QIcon("📥"), "Downloads", self)
        self.downloads_btn.triggered.connect(self.show_downloads)
        toolbar.addAction(self.downloads_btn)
        
    def create_sidebar(self):
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
            }
            QPushButton {
                background: #3a3a3a;
                color: white;
                padding: 10px;
                border: none;
                text-align: left;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        
        # Sidebar tabs
        sidebar_tabs = QTabWidget()
        
        # Dashboard Tab
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)
        
        # Market Summary
        market_group = QGroupBox("Market Summary")
        market_layout = QVBoxLayout()
        
        self.market_status = QLabel("Market: Open")
        self.market_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        market_layout.addWidget(self.market_status)
        
        self.sp500_label = QLabel("S&P 500: 4,500.00 (+0.5%)")
        market_layout.addWidget(self.sp500_label)
        
        self.nasdaq_label = QLabel("NASDAQ: 14,000.00 (+0.8%)")
        market_layout.addWidget(self.nasdaq_label)
        
        self.dow_label = QLabel("DOW: 35,000.00 (+0.3%)")
        market_layout.addWidget(self.dow_label)
        
        market_group.setLayout(market_layout)
        dashboard_layout.addWidget(market_group)
        
        # Quick Links
        links_group = QGroupBox("Quick Links")
        links_layout = QVBoxLayout()
        
        links = [
            ("Bloomberg", "https://www.bloomberg.com"),
            ("Yahoo Finance", "https://finance.yahoo.com"),
            ("MarketWatch", "https://www.marketwatch.com"),
            ("Investing.com", "https://www.investing.com"),
            ("CNBC", "https://www.cnbc.com"),
            ("Reuters", "https://www.reuters.com/finance")
        ]
        
        for name, url in links:
            btn = QPushButton(f"🌐 {name}")
            btn.clicked.connect(lambda checked, u=url: self.load_url(u))
            links_layout.addWidget(btn)
        
        links_group.setLayout(links_layout)
        dashboard_layout.addWidget(links_group)
        
        # Watchlist
        watchlist_group = QGroupBox("Watchlist")
        watchlist_layout = QVBoxLayout()
        
        self.watchlist_widget = QListWidget()
        self.watchlist_widget.itemDoubleClicked.connect(self.on_watchlist_item_clicked)
        watchlist_layout.addWidget(self.watchlist_widget)
        
        watchlist_input_layout = QHBoxLayout()
        self.watchlist_input = QLineEdit()
        self.watchlist_input.setPlaceholderText("Add symbol...")
        watchlist_input_layout.addWidget(self.watchlist_input)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_to_watchlist)
        watchlist_input_layout.addWidget(add_btn)
        
        watchlist_layout.addLayout(watchlist_input_layout)
        watchlist_group.setLayout(watchlist_layout)
        dashboard_layout.addWidget(watchlist_group)
        
        dashboard_layout.addStretch()
        sidebar_tabs.addTab(dashboard_widget, "📊 Dashboard")
        
        # History Tab
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(clear_history_btn)
        
        sidebar_tabs.addTab(history_widget, "📜 History")
        
        # Bookmarks Tab
        bookmarks_widget = QWidget()
        bookmarks_layout = QVBoxLayout(bookmarks_widget)
        
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.itemDoubleClicked.connect(self.on_bookmark_clicked)
        bookmarks_layout.addWidget(self.bookmarks_list)
        
        add_bookmark_layout = QHBoxLayout()
        self.bookmark_name_input = QLineEdit()
        self.bookmark_name_input.setPlaceholderText("Bookmark name")
        add_bookmark_layout.addWidget(self.bookmark_name_input)
        
        add_bookmark_btn = QPushButton("Add Current")
        add_bookmark_btn.clicked.connect(self.add_current_bookmark)
        add_bookmark_layout.addWidget(add_bookmark_btn)
        
        bookmarks_layout.addLayout(add_bookmark_layout)
        sidebar_tabs.addTab(bookmarks_widget, "⭐ Bookmarks")
        
        # Screener Tab
        screener_widget = QWidget()
        screener_layout = QVBoxLayout(screener_widget)
        
        screener_filters = QGroupBox("Filters")
        filter_layout = QFormLayout()
        
        self.filter_sector = QComboBox()
        self.filter_sector.addItems(["All", "Technology", "Finance", "Healthcare", "Energy", "Consumer"])
        filter_layout.addRow("Sector:", self.filter_sector)
        
        self.filter_price_min = QLineEdit()
        self.filter_price_min.setPlaceholderText("Min")
        filter_layout.addRow("Price Range:", self.filter_price_min)
        
        self.filter_price_max = QLineEdit()
        self.filter_price_max.setPlaceholderText("Max")
        filter_layout.addRow("", self.filter_price_max)
        
        self.filter_pe_min = QLineEdit()
        self.filter_pe_min.setPlaceholderText("Min PE")
        filter_layout.addRow("PE Ratio:", self.filter_pe_min)
        
        screener_filters.setLayout(filter_layout)
        screener_layout.addWidget(screener_filters)
        
        screener_results = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.screener_table = QTableWidget(0, 5)
        self.screener_table.setHorizontalHeaderLabels(["Symbol", "Price", "Change", "PE", "Sector"])
        self.screener_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.screener_table)
        
        run_btn = QPushButton("Run Screener")
        run_btn.clicked.connect(self.run_screener)
        results_layout.addWidget(run_btn)
        
        screener_results.setLayout(results_layout)
        screener_layout.addWidget(screener_results)
        
        sidebar_tabs.addTab(screener_widget, "🔍 Screener")
        
        layout.addWidget(sidebar_tabs)
        
        return sidebar
    
    def create_context_menu(self):
        self.context_menu = QMenu(self)
        
        back_action = QAction("Back", self)
        back_action.triggered.connect(self.navigate_back)
        self.context_menu.addAction(back_action)
        
        forward_action = QAction("Forward", self)
        forward_action.triggered.connect(self.navigate_forward)
        self.context_menu.addAction(forward_action)
        
        self.context_menu.addSeparator()
        
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_page)
        self.context_menu.addAction(refresh_action)
        
        self.context_menu.addSeparator()
        
        bookmark_action = QAction("Bookmark This Page", self)
        bookmark_action.triggered.connect(self.add_bookmark)
        self.context_menu.addAction(bookmark_action)
        
        inspect_action = QAction("Inspect Element", self)
        inspect_action.triggered.connect(self.inspect_element)
        self.context_menu.addAction(inspect_action)
    
    # ----------------------------------------------------------
    # Core Browser Functions
    # ----------------------------------------------------------
    
    def create_new_tab(self, url=None):
        new_tab = BrowserTab(self)
        new_tab.setUrl(QUrl(url if url else "https://www.google.com"))
        
        index = self.tab_widget.addTab(new_tab, "New Tab")
        self.tab_widget.setCurrentIndex(index)
        
        self.current_tabs[index] = new_tab
        
        # Connect signals
        new_tab.urlChanged.connect(lambda u: self.update_url_bar(u, index))
        new_tab.titleChanged.connect(lambda t: self.update_tab_title(t, index))
        new_tab.loadProgress.connect(lambda p: self.update_progress(p, index))
        
        return new_tab
    
    def close_tab(self, index):
        if self.tab_widget.count() > 1:
            widget = self.tab_widget.widget(index)
            if widget:
                widget.deleteLater()
                del self.current_tabs[index]
            self.tab_widget.removeTab(index)
        else:
            self.close()
    
    def tab_changed(self, index):
        if index >= 0:
            current_browser = self.tab_widget.widget(index)
            if current_browser:
                self.url_bar.setText(current_browser.url().toString())
    
    def navigate_to_url(self):
        url = self.url_bar.text().strip()
        if not url:
            return
        
        # Check if it's a stock symbol
        if not url.startswith(('http://', 'https://', 'file://')):
            if '.' not in url and '/' not in url and len(url) <= 10:
                self.show_stock_analysis(url.upper())
                return
            else:
                url = 'https://' + url if '.' in url else f'https://www.google.com/search?q={url}'
        
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.setUrl(QUrl(url))
    
    def navigate_back(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.back()
    
    def navigate_forward(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.forward()
    
    def refresh_page(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.reload()
    
    def go_home(self):
        self.create_new_tab("https://www.google.com")
    
    def update_url_bar(self, url, tab_index):
        if tab_index == self.tab_widget.currentIndex():
            self.url_bar.setText(url.toString())
            self.add_to_history(url.toString())
    
    def update_tab_title(self, title, tab_index):
        if title:
            self.tab_widget.setTabText(tab_index, title[:20])
    
    def update_progress(self, progress, tab_index):
        if tab_index == self.tab_widget.currentIndex():
            if progress < 100:
                self.status_bar.showMessage(f"Loading... {progress}%")
            else:
                self.status_bar.showMessage("Ready")
    
    # ----------------------------------------------------------
    # Stock Market Functions
    # ----------------------------------------------------------
    
    def search_stock(self):
        symbol = self.stock_search.text().strip().upper()
        if symbol:
            self.show_stock_analysis(symbol)
    
    def show_stock_analysis(self, symbol):
        """Create advanced stock analysis page"""
        html_content = self.generate_stock_html(symbol)
        
        # Create new tab for stock analysis
        new_tab = BrowserTab(self)
        new_tab.setHtml(html_content, QUrl("file:///"))
        
        index = self.tab_widget.addTab(new_tab, f"📈 {symbol}")
        self.tab_widget.setCurrentIndex(index)
        self.current_tabs[index] = new_tab
    
    def generate_stock_html(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y", interval="1d")
            
            if hist.empty:
                return f"<h1>No data for {symbol}</h1>"
            
            # Calculate statistics
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev_close = info.get('previousClose', 0)
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            # Generate chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close',
                line=dict(color='#4CAF50', width=2)
            ))
            
            # Add moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f'{symbol} - Stock Price',
                template='plotly_dark',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
            
            # Generate HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{symbol} Analysis</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ 
                        background: #0a0a0a; 
                        color: white; 
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .stat-card {{
                        background: #1a1a1a;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid #4CAF50;
                    }}
                    .price {{
                        font-size: 2.5em;
                        font-weight: bold;
                    }}
                    .change {{ color: {'#4CAF50' if change >= 0 else '#f44336'}; }}
                    .chart-container {{ 
                        background: #1a1a1a; 
                        padding: 20px; 
                        border-radius: 10px;
                        margin: 20px 0;
                    }}
                    .data-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    .data-table th, .data-table td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #333;
                    }}
                    .data-table th {{
                        background: #2a2a2a;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{info.get('longName', symbol)} ({symbol})</h1>
                        <div class="price">${current_price:.2f}</div>
                        <div class="change">{change:+.2f} ({change_percent:+.2f}%)</div>
                        <p>{info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>Market Cap</h3>
                            <p>${info.get('marketCap', 0):,.0f}</p>
                        </div>
                        <div class="stat-card">
                            <h3>P/E Ratio</h3>
                            <p>{info.get('trailingPE', 'N/A')}</p>
                        </div>
                        <div class="stat-card">
                            <h3>Volume</h3>
                            <p>{info.get('volume', 0):,.0f}</p>
                        </div>
                        <div class="stat-card">
                            <h3>52W High/Low</h3>
                            <p>${info.get('fiftyTwoWeekHigh', 0):.2f} / ${info.get('fiftyTwoWeekLow', 0):.2f}</p>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <div id="chart"></div>
                    </div>
                    
                    <table class="data-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Open</td>
                            <td>${info.get('open', 0):.2f}</td>
                            <td>Day's Range</td>
                            <td>${info.get('dayLow', 0):.2f} - ${info.get('dayHigh', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Previous Close</td>
                            <td>${prev_close:.2f}</td>
                            <td>Beta</td>
                            <td>{info.get('beta', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>EPS</td>
                            <td>{info.get('trailingEps', 'N/A')}</td>
                            <td>Dividend Yield</td>
                            <td>{info.get('dividendYield', 0)*100 if info.get('dividendYield') else 0:.2f}%</td>
                        </tr>
                    </table>
                </div>
                
                <script>
                    var chartData = {chart_html.split('<div')[1].split('</script>')[0] if '<div' in chart_html else ''};
                    document.getElementById('chart').innerHTML = chartData;
                </script>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            return f"<h1>Error loading {symbol}</h1><p>{str(e)}</p>"
    
    def add_to_watchlist(self):
        symbol = self.watchlist_input.text().strip().upper()
        if symbol and symbol not in self.stock_watchlist:
            self.stock_watchlist.append(symbol)
            self.watchlist_widget.addItem(f"{symbol} - Loading...")
            self.watchlist_input.clear()
            
            # Fetch stock data in background
            threading.Thread(target=self.update_watchlist_item, args=(symbol, len(self.stock_watchlist)-1)).start()
    
    def update_watchlist_item(self, symbol, index):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev_close = info.get('previousClose', 0)
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            QMetaObject.invokeMethod(self, "update_watchlist_ui", 
                                   Qt.ConnectionType.QueuedConnection,
                                   Q_ARG(int, index),
                                   Q_ARG(str, symbol),
                                   Q_ARG(float, current_price),
                                   Q_ARG(float, change_percent))
        except:
            pass
    
    def update_watchlist_ui(self, index, symbol, price, change_percent):
        color = "#4CAF50" if change_percent >= 0 else "#f44336"
        self.watchlist_widget.item(index).setText(
            f"{symbol}: ${price:.2f} ({change_percent:+.2f}%)")
        self.watchlist_widget.item(index).setForeground(QColor(color))
    
    def on_watchlist_item_clicked(self, item):
        symbol = item.text().split(":")[0].strip()
        self.show_stock_analysis(symbol)
    
    def run_screener(self):
        # This is a simplified screener - in production, you'd use a real API
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        
        self.screener_table.setRowCount(0)
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                current_price = info.get('currentPrice', 0)
                prev_close = info.get('previousClose', 0)
                change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                pe = info.get('trailingPE', 0)
                sector = info.get('sector', 'N/A')
                
                row = self.screener_table.rowCount()
                self.screener_table.insertRow(row)
                
                self.screener_table.setItem(row, 0, QTableWidgetItem(symbol))
                self.screener_table.setItem(row, 1, QTableWidgetItem(f"${current_price:.2f}"))
                self.screener_table.setItem(row, 2, QTableWidgetItem(f"{change:+.2f}%"))
                self.screener_table.setItem(row, 3, QTableWidgetItem(f"{pe:.1f}" if pe else "N/A"))
                self.screener_table.setItem(row, 4, QTableWidgetItem(sector))
                
            except:
                pass
    
    # ----------------------------------------------------------
    # Bookmark & History Functions
    # ----------------------------------------------------------
    
    def add_bookmark(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            url = current_browser.url().toString()
            title = current_browser.title()
            self.add_bookmark_item(title, url)
    
    def add_current_bookmark(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            url = current_browser.url().toString()
            name = self.bookmark_name_input.text().strip() or current_browser.title()
            self.add_bookmark_item(name, url)
            self.bookmark_name_input.clear()
    
    def add_bookmark_item(self, name, url):
        bookmark = f"{name} | {url}"
        if bookmark not in self.bookmarks:
            self.bookmarks.append(bookmark)
            self.bookmarks_list.addItem(bookmark)
            self.save_settings()
    
    def on_bookmark_clicked(self, item):
        url = item.text().split(" | ")[-1]
        self.load_url(url)
    
    def add_to_history(self, url):
        if url and url not in self.history[-10:]:  # Keep last 10
            self.history.append(url)
            self.history_list.insertItem(0, url)
            if self.history_list.count() > 50:
                self.history_list.takeItem(50)
    
    def clear_history(self):
        self.history.clear()
        self.history_list.clear()
    
    # ----------------------------------------------------------
    # Utility Functions
    # ----------------------------------------------------------
    
    def load_url(self, url):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.setUrl(QUrl(url))
        else:
            self.create_new_tab(url)
    
    def show_welcome_screen(self):
        welcome_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .welcome-container {
                    text-align: center;
                    padding: 50px;
                    background: rgba(0,0,0,0.3);
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                }
                h1 {
                    font-size: 3em;
                    margin-bottom: 20px;
                }
                .features {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin: 30px 0;
                }
                .feature {
                    background: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 10px;
                }
            </style>
        </head>
        <body>
            <div class="welcome-container">
                <h1>🚀 Quantum Browser</h1>
                <p>Advanced Web & Stock Market Browser</p>
                
                <div class="features">
                    <div class="feature">🌐 Multi-tab Browsing</div>
                    <div class="feature">📈 Real-time Stock Data</div>
                    <div class="feature">⭐ Watchlist Management</div>
                    <div class="feature">🔍 Stock Screener</div>
                    <div class="feature">📊 Interactive Charts</div>
                    <div class="feature">💾 Bookmarks & History</div>
                </div>
                
                <p>Enter a URL or stock symbol to get started!</p>
            </div>
        </body>
        </html>
        """
        
        new_tab = BrowserTab(self)
        new_tab.setHtml(welcome_html, QUrl("file:///"))
        
        index = self.tab_widget.addTab(new_tab, "Welcome")
        self.tab_widget.setCurrentIndex(index)
        self.current_tabs[index] = new_tab
    
    def show_downloads(self):
        QMessageBox.information(self, "Downloads", "Download manager coming soon!")
    
    def inspect_element(self):
        current_browser = self.tab_widget.currentWidget()
        if current_browser:
            current_browser.page().triggerAction(QWebEnginePage.WebAction.InspectElement)
    
    def load_settings(self):
        # In production, load from file
        pass
    
    def save_settings(self):
        # In production, save to file
        pass
    
    def closeEvent(self, event):
        # Save settings on close
        self.save_settings()
        event.accept()

# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark theme
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
    app.setPalette(dark_palette)
    
    window = AdvancedStockBrowser()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
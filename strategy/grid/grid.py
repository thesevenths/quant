import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import backtrader as bt
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class OptimizedGridStrategy(bt.Strategy):
    """
    核心特性：
    1. 智能价格区间设置 (85%-115%)
    2. 动态区间调整机制
    3. 趋势适应能力
    4. 完善的风险控制
    5. 手续费优化
    """
    params = (
        ('initial_capital', 100000),
        ('commission', 0.001),           # 0.1%手续费
        ('grid_adjustment_threshold', 0.05),  # 5%价格偏离触发区间调整
        ('max_position_per_grid', 0.05),    # 单网格最大仓位5%
        ('trend_follow_ratio', 0.1),        # 10%资金用于趋势跟踪
        ('stop_loss_ratio', 0.15),          # 15%总资金止损
        ('rebalance_days', 7),              # 每7天强制重平衡
        ('min_grid_spacing_ratio', 0.01),   # 最小网格间距1%
        ('max_grid_spacing_ratio', 0.03),   # 最大网格间距3%
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_datetime = self.datas[0].datetime
        
        # 初始化网格参数
        self.current_price = self.data_close[0]
        self.lower_bound = self.current_price * 0.85
        self.upper_bound = self.current_price * 1.15
        self.num_grids = 18
        
        self.grid_levels = self._calculate_grid_levels()
        self.grid_positions = {i: 0.0 for i in range(len(self.grid_levels) - 1)}
        self.grid_orders = {i: None for i in range(len(self.grid_levels) - 1)}
        
        # 跟踪变量
        self.order_count = 0
        self.trade_count = 0
        self.total_profit = 0
        self.highest_value = self.params.initial_capital
        self.lowest_value = self.params.initial_capital
        self.last_rebalance_date = self.data_datetime.date(0)
        self.start_date = self.data_datetime.date(0)
        
        # 趋势跟踪变量
        self.trend_position = 0
        self.trend_entry_price = 0
        
        # 记录每日净值
        self.daily_values = []
        self.daily_dates = []
        
        print("=== 优化版网格交易策略初始化 ===")
        print(f"初始价格: ${self.current_price:,.2f}")
        print(f"初始区间: ${self.lower_bound:,.2f} - ${self.upper_bound:,.2f}")
        print(f"网格数量: {self.num_grids}")
        print(f"网格间距: ${(self.upper_bound-self.lower_bound)/self.num_grids:,.2f}")
        print(f"趋势资金比例: {self.params.trend_follow_ratio*100:.1f}%")
        print(f"止损比例: {self.params.stop_loss_ratio*100:.1f}%")
    
    def _calculate_grid_levels(self):
        """计算网格级别"""
        return np.linspace(self.lower_bound, self.upper_bound, self.num_grids + 1)
    
    def _should_adjust_boundaries(self):
        """检查是否需要调整边界"""
        current_price = self.data_close[0]
        current_date = self.data_datetime.date(0)
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        
        # 定期重平衡
        if days_since_rebalance >= self.params.rebalance_days:
            return True
        
        # 价格突破边界5%
        below_lower = current_price < self.lower_bound * (1 - self.params.grid_adjustment_threshold)
        above_upper = current_price > self.upper_bound * (1 + self.params.grid_adjustment_threshold)
        
        return below_lower or above_upper
    
    def _adjust_grid_boundaries(self):
        """动态调整网格边界"""
        current_price = self.data_close[0]
        current_date = self.data_datetime.date(0)
        old_lower, old_upper = self.lower_bound, self.upper_bound
        
        if current_price < self.lower_bound * (1 - self.params.grid_adjustment_threshold):
            # 价格大幅下跌
            self.lower_bound = current_price * 0.90
            self.upper_bound = current_price * 1.20
            adjustment_type = "大幅下跌"
        elif current_price > self.upper_bound * (1 + self.params.grid_adjustment_threshold):
            # 价格大幅上涨
            self.lower_bound = current_price * 0.80
            self.upper_bound = current_price * 1.30
            adjustment_type = "大幅上涨"
        else:
            # 定期重新校准
            self.lower_bound = current_price * 0.85
            self.upper_bound = current_price * 1.15
            adjustment_type = "定期校准"
        
        # 重新计算网格数量
        price_range = self.upper_bound - self.lower_bound
        current_price_avg = (self.lower_bound + self.upper_bound) / 2
        
        # 根据波动率调整网格密度
        if len(self.data.close) >= 24:
            recent_prices = np.array(self.data.close.get(size=24))
            volatility = np.std(np.diff(np.log(recent_prices))) if len(recent_prices) > 1 else 0.01
            
            if volatility > 0.02:  # 高波动
                target_spacing = current_price_avg * 0.025
            elif volatility < 0.005:  # 低波动
                target_spacing = current_price_avg * 0.012
            else:  # 中等波动
                target_spacing = current_price_avg * 0.018
            
            self.num_grids = max(12, min(25, int(price_range / target_spacing)))
        else:
            self.num_grids = 18
        
        # 重新计算网格
        self.grid_levels = self._calculate_grid_levels()
        self.grid_positions = {i: 0.0 for i in range(len(self.grid_levels) - 1)}
        self.grid_orders = {i: None for i in range(len(self.grid_levels) - 1)}
        
        # 取消所有未完成订单
        for order in self.broker.get_orders_open():
            self.broker.cancel(order)
        
        print(f"\n🔄 网格区间调整 ({adjustment_type}) - {current_date}")
        print(f"  价格: ${current_price:,.2f}")
        print(f"  旧区间: ${old_lower:,.2f} - ${old_upper:,.2f} ({len(self.grid_levels)-1} grids)")
        print(f"  新区间: ${self.lower_bound:,.2f} - ${self.upper_bound:,.2f} ({self.num_grids} grids)")
        print(f"  新间距: ${(self.upper_bound-self.lower_bound)/self.num_grids:,.2f}")
        print(f"  预计日交易次数: {self._estimate_daily_trades():.1f}")
        
        self.last_rebalance_date = current_date
    
    def _estimate_daily_trades(self):
        """估算每日交易次数"""
        if len(self.data.close) < 24:
            return 10.0
        
        recent_prices = np.array(self.data.close.get(size=24))
        price_range = np.max(recent_prices) - np.min(recent_prices)
        avg_grid_spacing = (self.upper_bound - self.lower_bound) / self.num_grids
        
        if avg_grid_spacing == 0:
            return 5.0
        
        estimated_trades = price_range / avg_grid_spacing * 2  # 每次穿越网格算2次交易
        return min(15.0, max(3.0, estimated_trades))  # 限制在3-15次/天
    
    def _execute_grid_trades(self):
        """执行网格交易"""
        current_price = self.data_close[0]
        current_date = self.data_datetime.date(0)
        
        # 可用于网格交易的资金（扣除趋势跟踪部分）
        grid_capital = self.broker.getcash() * (1 - self.params.trend_follow_ratio)
        cash_per_grid = grid_capital / max(1, self.num_grids)
        
        for i in range(len(self.grid_levels) - 1):
            grid_lower = self.grid_levels[i]
            grid_upper = self.grid_levels[i + 1]
            grid_mid = (grid_lower + grid_upper) / 2
            
            position_size = self.grid_positions[i]
            
            # 买入条件：价格低于中线且没有持仓
            if current_price < grid_mid and position_size == 0:
                max_size = cash_per_grid / current_price
                size = min(max_size, self.params.max_position_per_grid * self.broker.getvalue() / current_price)
                
                if size > 0.0001 and self.broker.getcash() >= size * current_price * 1.01:
                    # 使用限价单，避免滑点
                    order_price = current_price * 0.998
                    self.buy(size=size, exectype=bt.Order.Limit, price=order_price)
                    self.grid_positions[i] = size
                    self.order_count += 1
            
            # 卖出条件：价格高于中线且有持仓
            elif current_price > grid_mid and position_size > 0:
                if self.broker.getposition(self.data).size >= position_size:
                    order_price = current_price * 1.002
                    self.sell(size=position_size, exectype=bt.Order.Limit, price=order_price)
                    self.grid_positions[i] = 0
                    self.order_count += 1
                    self.trade_count += 1
                    
                    # 计算利润
                    buy_value = cash_per_grid  # 简化计算
                    sell_value = position_size * current_price
                    profit = sell_value - buy_value
                    self.total_profit += profit
    
    def _execute_trend_following(self):
        """执行趋势跟踪交易"""
        if self.params.trend_follow_ratio <= 0:
            return
        
        current_price = self.data_close[0]
        trend_capital = self.broker.getvalue() * self.params.trend_follow_ratio
        
        # 20小时均线趋势判断
        if len(self.data.close) >= 20:
            ma20 = sum(self.data.close.get(size=20)) / 20
            
            current_position = self.broker.getposition(self.data).size
            
            # 上升趋势：价格在20小时均线上方
            if current_price > ma20 * 1.01 and self.trend_position <= 0:
                # 清空空头，建立多头
                if self.trend_position < 0:
                    self.close()
                    self.trend_position = 0
                
                size = trend_capital / current_price * 0.8
                if size > 0.0001 and self.broker.getcash() >= size * current_price * 1.01:
                    self.buy(size=size)
                    self.trend_position = size
                    self.trend_entry_price = current_price
            
            # 下降趋势：价格在20小时均线下方
            elif current_price < ma20 * 0.99 and self.trend_position >= 0:
                # 清多头，建立空头
                if self.trend_position > 0:
                    self.close()
                    self.trend_position = 0
                
                size = trend_capital / current_price * 0.8
                if size > 0.0001:
                    self.sell(size=size)
                    self.trend_position = -size
                    self.trend_entry_price = current_price
    
    def _check_stop_loss(self):
        """检查止损条件"""
        current_value = self.broker.getvalue()
        current_date = self.data_datetime.date(0)
        
        if current_value > self.highest_value:
            self.highest_value = current_value
        
        if current_value < self.lowest_value:
            self.lowest_value = current_value
        
        max_drawdown = (self.highest_value - current_value) / self.highest_value
        
        if max_drawdown > self.params.stop_loss_ratio:
            print(f"\n🚨 触发止损! ({current_date})")
            print(f"  当前价值: ${current_value:,.2f}")
            print(f"  最高价值: ${self.highest_value:,.2f}")
            print(f"  最大回撤: {max_drawdown:.2%}")
            
            # 清空所有仓位
            for i in range(len(self.grid_positions)):
                if self.grid_positions[i] > 0:
                    self.sell(size=self.grid_positions[i])
                    self.grid_positions[i] = 0
            
            if self.trend_position != 0:
                self.close()
                self.trend_position = 0
            
            # 重置网格
            self.lower_bound = current_value * 0.85 / (self.num_grids * 1.0)
            self.upper_bound = current_value * 1.15 / (self.num_grids * 1.0)
            self.grid_levels = self._calculate_grid_levels()
    
    def next(self):
        """主逻辑"""
        current_date = self.data_datetime.date(0)
        
        # 每日记录净值
        if not self.daily_dates or self.daily_dates[-1] != current_date:
            self.daily_values.append(self.broker.getvalue())
            self.daily_dates.append(current_date)
        
        # 检查止损
        self._check_stop_loss()
        
        # 检查是否需要调整网格
        if self._should_adjust_boundaries():
            self._adjust_grid_boundaries()
        
        # 执行网格交易
        self._execute_grid_trades()
        
        # 执行趋势跟踪
        self._execute_trend_following()
    
    def stop(self):
        """策略结束时的统计"""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.params.initial_capital - 1) * 100
        max_drawdown = (1 - self.lowest_value / self.highest_value) * 100
        
        print(f'\n\n=== 回测结果总结 ===')
        print(f'回测期间: {self.start_date} 到 {self.data_datetime.date(-1)}')
        print(f'初始资金: ${self.params.initial_capital:,.2f}')
        print(f'最终价值: ${final_value:,.2f}')
        print(f'总收益率: {total_return:.2f}%')
        print(f'总订单数: {self.order_count}')
        print(f'完成交易数: {self.trade_count}')
        print(f'网格总利润: ${self.total_profit:,.2f}')
        print(f'最大回撤: {max_drawdown:.2f}%')
        print(f'最高价值: ${self.highest_value:,.2f}')
        print(f'最低价值: ${self.lowest_value:,.2f}')
        
        # 计算年化收益率
        total_days = (self.data_datetime.date(-1) - self.start_date).days
        if total_days > 0:
            annualized_return = ((1 + total_return/100) ** (365/total_days) - 1) * 100
            print(f'年化收益率: {annualized_return:.2f}%')
        
        # 保存策略参数和结果
        strategy_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backtest_period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.data_datetime.date(-1).strftime('%Y-%m-%d'),
                'total_days': total_days
            },
            'performance': {
                'initial_capital': self.params.initial_capital,
                'final_value': float(final_value),
                'total_return_percent': float(total_return),
                'annualized_return_percent': float(annualized_return) if total_days > 0 else 0,
                'max_drawdown_percent': float(max_drawdown),
                'total_orders': self.order_count,
                'completed_trades': self.trade_count,
                'grid_profit': float(self.total_profit)
            },
            'final_grid_params': {
                'lower_bound': float(self.lower_bound),
                'upper_bound': float(self.upper_bound),
                'num_grids': self.num_grids,
                'grid_levels': [float(level) for level in self.grid_levels],
                'trend_position': float(self.trend_position)
            },
            'daily_nav': {
                'dates': [date.strftime('%Y-%m-%d') for date in self.daily_dates],
                'values': [float(value) for value in self.daily_values]
            }
        }
        
        # 保存JSON结果
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(script_dir, 'grid_strategy_results.json')
        with open(json_file, 'w') as f:
            json.dump(strategy_results, f, indent=2)
        print(f'\n💾 策略结果已保存至: {json_file}')
        
        # 生成性能图表
        self._generate_performance_chart(strategy_results)
    
    def _generate_performance_chart(self, results):
        """生成性能图表"""
        try:
            plt.figure(figsize=(15, 12))
            
            # 1. 净值曲线
            plt.subplot(3, 1, 1)
            dates = [datetime.strptime(date, '%Y-%m-%d') for date in results['daily_nav']['dates']]
            values = results['daily_nav']['values']
            initial_value = results['performance']['initial_capital']
            
            plt.plot(dates, values, 'b-', linewidth=2, label='网格策略')
            plt.axhline(y=initial_value, color='k', linestyle='--', alpha=0.3, label='初始资金')
            
            # 计算买入持有策略
            if hasattr(self.datas[0], 'close'):
                buy_hold_values = [initial_value * (v / self.data.close[0]) for v in self.data.close.get(size=len(dates))]
                plt.plot(dates, buy_hold_values, 'r--', linewidth=2, alpha=0.7, label='买入持有')
            
            plt.title('净值曲线对比', fontsize=14, fontweight='bold')
            plt.ylabel('净值 ($)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            # 2. 网格分布
            plt.subplot(3, 1, 2)
            grid_levels = results['final_grid_params']['grid_levels']
            current_price = self.data.close[0] if len(self.data.close) > 0 else grid_levels[len(grid_levels)//2]
            
            plt.axhline(y=current_price, color='blue', linewidth=2, label=f'当前价格: ${current_price:,.2f}')
            for level in grid_levels:
                plt.axhline(y=level, color='gray', alpha=0.5, linestyle='--')
            
            plt.axhspan(grid_levels[0], grid_levels[-1], alpha=0.2, color='green', label='网格区间')
            
            plt.title('网格分布', fontsize=14, fontweight='bold')
            plt.ylabel('价格 ($)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # 3. 月度收益
            plt.subplot(3, 1, 3)
            monthly_returns = []
            monthly_dates = []
            
            for i in range(1, len(values)):
                if i % 30 == 0:  # 大约每月
                    monthly_return = (values[i] / values[i-30] - 1) * 100 if i >= 30 else 0
                    monthly_returns.append(monthly_return)
                    monthly_dates.append(dates[i])
            
            if monthly_returns:
                plt.bar(monthly_dates, monthly_returns, color=['red' if x < 0 else 'green' for x in monthly_returns])
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                plt.title('月度收益率', fontsize=14, fontweight='bold')
                plt.ylabel('收益率 (%)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            plt.tight_layout()
            
            # 保存图表
            script_dir = os.path.dirname(os.path.abspath(__file__))
            chart_file = os.path.join(script_dir, 'grid_strategy_performance.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f'📊 性能图表已保存至: {chart_file}')
            plt.close()
            
        except Exception as e:
            print(f'⚠️  图表生成失败: {e}')

def load_data():
    """加载比特币数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'btc_usdt_2y_1h.csv')
    
    if not os.path.exists(csv_file):
        print(f'❌ 文件 {csv_file} 不存在！')
        print('💡 请确保CSV文件包含以下列: datetime, open, high, low, close, volume')
        print('💡 文件应放在与脚本相同的目录下')
        raise FileNotFoundError(f"文件 {csv_file} 不存在！")
    
    print(f'📈 加载数据文件: {csv_file}')
    df = pd.read_csv(csv_file)
    
    # 确保datetime列存在
    if 'datetime' not in df.columns:
        raise ValueError('CSV文件必须包含datetime列')
    
    # 转换datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f'✅ 数据加载成功！')
    print(f'   记录数: {len(df)}')
    print(f'   时间范围: {df["datetime"].iloc[0]} 到 {df["datetime"].iloc[-1]}')
    print(f'   价格范围: ${df["low"].min():,.2f} - ${df["high"].max():,.2f}')
    print(f'   平均成交量: {df["volume"].mean():,.0f}')
    
    return df

def calculate_initial_grid_params(df, lookback_days=180):
    """计算初始网格参数"""
    print(f'\n🔧 计算初始网格参数 (回看{lookback_days}天)...')
    
    # 获取当前价格
    current_price = df['close'].iloc[-1]
    print(f'   当前价格: ${current_price:,.2f}')
    
    # 获取回看数据
    required_hours = lookback_days * 24
    if len(df) < required_hours:
        print(f'   ⚠️  数据不足{lookback_days}天，使用全部{len(df)//24}天数据')
        lookback_data = df.copy()
    else:
        lookback_data = df.iloc[-required_hours:].copy()
    
    # 计算历史波动率
    lookback_data['log_return'] = np.log(lookback_data['close'] / lookback_data['close'].shift(1))
    volatility = lookback_data['log_return'].std() * np.sqrt(24)  # 小时波动率年化
    
    print(f'   历史波动率: {volatility:.2%}')
    
    # 设置价格区间（85%-115%）
    lower_bound = current_price * 0.85
    upper_bound = current_price * 1.15
    
    # 根据波动率微调
    if volatility > 0.05:  # 高波动
        lower_bound = current_price * 0.80
        upper_bound = current_price * 1.20
        print('   💥 高波动市场，扩大区间至80%-120%')
    elif volatility < 0.015:  # 低波动
        lower_bound = current_price * 0.90
        upper_bound = current_price * 1.10
        print('   📉 低波动市场，收窄区间至90%-110%')
    
    # 计算网格间距
    price_range = upper_bound - lower_bound
    avg_price = (lower_bound + upper_bound) / 2
    
    # 根据波动率设置网格密度
    if volatility > 0.05:
        target_spacing_ratio = 0.025  # 2.5%
    elif volatility < 0.015:
        target_spacing_ratio = 0.010  # 1.0%
    else:
        target_spacing_ratio = 0.018  # 1.8%
    
    grid_spacing = avg_price * target_spacing_ratio
    num_grids = max(12, min(25, int(price_range / grid_spacing)))
    
    print(f'   优化后区间: ${lower_bound:,.2f} - ${upper_bound:,.2f}')
    print(f'   网格数量: {num_grids}')
    print(f'   网格间距: ${grid_spacing:,.2f} ({grid_spacing/avg_price:.2%})')
    print(f'   预计日交易次数: {price_range/grid_spacing*2:.1f}')
    
    return {
        'current_price': float(current_price),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'num_grids': num_grids,
        'grid_spacing': float(grid_spacing),
        'volatility': float(volatility),
        'lookback_days': lookback_days
    }

def run_backtest(df, initial_params):
    """运行回测"""
    print(f'\n🚀 开始回测...')
    
    # 准备回测数据（使用最近1年数据）
    backtest_days = 365
    required_hours = backtest_days * 24
    
    if len(df) < required_hours:
        print(f'   ⚠️  数据不足{backtest_days}天，使用全部{len(df)//24}天数据进行回测')
        backtest_data = df.copy()
    else:
        backtest_data = df.iloc[-required_hours:].copy()
    
    print(f'   回测期间: {backtest_data["datetime"].iloc[0]} 到 {backtest_data["datetime"].iloc[-1]}')
    print(f'   回测天数: {len(backtest_data)//24}')
    
    # 创建回测引擎
    cerebro = bt.Cerebro()
    
    # 添加数据
    data = bt.feeds.PandasData(
        dataname=backtest_data,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(OptimizedGridStrategy)
    
    # 设置初始资金
    initial_capital = 100000
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    print(f'   初始资金: ${initial_capital:,.2f}')
    print(f'   手续费: {0.001*100:.2f}%')
    
    results = cerebro.run()
    strategy = results[0]
    
    # 打印分析器结果
    print(f'\n📈 性能指标:')
    sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', None)
    if sharpe is not None:
        print(f'   夏普比率: {sharpe:.2f}')
    
    drawdown = strategy.analyzers.drawdown.get_analysis()
    print(f'   最大回撤: {drawdown.max.drawdown:.2f}%')
    
    returns = strategy.analyzers.returns.get_analysis()
    total_return = returns.get('rtot', 0) * 100
    print(f'   总收益率: {total_return:.2f}%')
    
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    if hasattr(trade_analyzer, 'total'):
        total_trades = trade_analyzer.total.total
        won_trades = trade_analyzer.won.total if hasattr(trade_analyzer.won, 'total') else 0
        win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0
        print(f'   总交易次数: {total_trades}')
        print(f'   胜率: {win_rate:.2f}%')
    
    # 与买入持有对比
    buy_hold_return = (backtest_data['close'].iloc[-1] / backtest_data['close'].iloc[0] - 1) * 100
    print(f'   买入持有收益率: {buy_hold_return:.2f}%')
    print(f'   超额收益: {total_return - buy_hold_return:.2f}%')
    
    # 绘制图表
    print(f'\n📊 生成回测图表...')
    fig = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=True, 
                      title='比特币网格交易策略回测', figsize=(15, 10))[0][0]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backtest_chart = os.path.join(script_dir, 'backtest_chart.png')
    fig.savefig(backtest_chart, dpi=300, bbox_inches='tight')
    print(f'   回测图表已保存至: {backtest_chart}')
    plt.close()
    
    return strategy

def main():
    """主函数"""
    print("=" * 60)
    print("₿ 优化版比特币网格交易策略")
    print(f"🕒 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df = load_data()
        
        # 2. 计算初始网格参数
        initial_params = calculate_initial_grid_params(df, lookback_days=180)
        
        # 3. 运行回测
        strategy = run_backtest(df, initial_params)
        
        if strategy:
            print("\n" + "=" * 60)
            print("🎉 网格交易策略回测完成！")
            print("💡 关键建议:")
            print("   • 使用优化后的参数进行实盘")
            print("   • 从小资金开始测试（建议$1000-$5000）")
            print("   • 每周检查一次策略表现")
            print("   • 市场剧烈波动时手动监控")
            print("   • 保持15%的止损纪律")
            print("=" * 60)
            
            # 生成实盘参数建议
            final_params = {
                'current_price': float(strategy.data.close[0]),
                'lower_bound': float(strategy.lower_bound),
                'upper_bound': float(strategy.upper_bound),
                'num_grids': strategy.num_grids,
                'grid_levels': [float(level) for level in strategy.grid_levels],
                'stop_loss_ratio': strategy.params.stop_loss_ratio,
                'trend_follow_ratio': strategy.params.trend_follow_ratio,
                'rebalance_days': strategy.params.rebalance_days,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            params_file = os.path.join(script_dir, 'live_trading_params.json')
            with open(params_file, 'w') as f:
                json.dump(final_params, f, indent=2)
            print(f'\n⚙️  实盘参数已保存至: {params_file}')
            
            print('\n📋 实盘参数摘要:')
            print(f'   价格区间: ${final_params["lower_bound"]:,.2f} - ${final_params["upper_bound"]:,.2f}')
            print(f'   网格数量: {final_params["num_grids"]}')
            print(f'   重新平衡: 每{final_params["rebalance_days"]}天')
            print(f'   止损比例: {final_params["stop_loss_ratio"]*100:.1f}%')
            print(f'   趋势资金: {final_params["trend_follow_ratio"]*100:.1f}%')
    
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 排错建议:")
        print("   1. 检查CSV文件格式和路径")
        print("   2. 确保文件包含datetime, open, high, low, close, volume列")
        print("   3. 安装所需依赖: pip install backtrader pandas numpy matplotlib")
        print("   4. 检查Python版本（建议3.8+）")

if __name__ == "__main__":
    # 设置matplotlib使用Agg后端（无GUI）
    import matplotlib
    matplotlib.use('Agg')
    
    # 运行主函数
    main()
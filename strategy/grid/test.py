import backtrader as bt
import numpy as np

class GridStrategy(bt.Strategy):
    params = (
        ('lower_bound', 75078),
        ('upper_bound', 101576),
        ('num_grids', 18),
        ('initial_capital', 100000),
        ('commission', 0.001),  # 0.1%手续费
    )
    
    def __init__(self):
        self.grid_levels = np.linspace(
            self.p.lower_bound, 
            self.p.upper_bound, 
            self.p.num_grids + 1
        )
        self.grid_positions = {level: 0 for level in self.grid_levels}
        self.order_count = 0
        self.total_profit = 0
        
    def next(self):
        current_price = self.data.close[0]
        
        # 检查是否需要调整区间
        self._check_dynamic_adjustment(current_price)
        
        # 网格交易逻辑
        for i in range(len(self.grid_levels) - 1):
            lower = self.grid_levels[i]
            upper = self.grid_levels[i + 1]
            
            if lower <= current_price < upper:
                # 价格在当前网格内
                self._execute_grid_trades(i, current_price)
    
    def _check_dynamic_adjustment(self, current_price):
        """动态调整价格区间"""
        if current_price < self.p.lower_bound * 0.95:
            print(f"📉 价格跌破下界，调整区间: {current_price:.2f}")
            self.p.lower_bound = current_price * 0.9
            self.p.upper_bound = current_price * 1.2
            self._recalculate_grid_levels()
        elif current_price > self.p.upper_bound * 1.05:
            print(f"📈 价格突破上界，调整区间: {current_price:.2f}")
            self.p.lower_bound = current_price * 0.8
            self.p.upper_bound = current_price * 1.3
            self._recalculate_grid_levels()
    
    def _recalculate_grid_levels(self):
        """重新计算网格级别"""
        self.grid_levels = np.linspace(
            self.p.lower_bound, 
            self.p.upper_bound, 
            self.p.num_grids + 1
        )
        print(f"🔄 新网格区间: ${self.p.lower_bound:,.2f} - ${self.p.upper_bound:,.2f}")
        print(f"📊 网格数量: {self.p.num_grids}, 间距: ${(self.p.upper_bound-self.p.lower_bound)/self.p.num_grids:,.2f}")
    
    def _execute_grid_trades(self, grid_index, current_price):
        """执行网格交易"""
        # 简化版：只展示核心逻辑
        cash_per_grid = self.p.initial_capital / self.p.num_grids
        
        # 买入信号：价格从上方向下穿过网格中线
        if current_price < (self.grid_levels[grid_index] + self.grid_levels[grid_index + 1]) / 2:
            if self.grid_positions[self.grid_levels[grid_index]] == 0:
                size = cash_per_grid / current_price
                self.buy(size=size)
                self.grid_positions[self.grid_levels[grid_index]] = 1
                self.order_count += 1
        
        # 卖出信号：价格从下方向上穿过网格中线
        elif current_price > (self.grid_levels[grid_index] + self.grid_levels[grid_index + 1]) / 2:
            if self.grid_positions[self.grid_levels[grid_index]] == 1:
                size = cash_per_grid / self.grid_levels[grid_index]
                self.sell(size=size)
                self.grid_positions[self.grid_levels[grid_index]] = 0
                self.order_count += 1
    
    def stop(self):
        print(f'=== 回测结果 ===')
        print(f'总订单数: {self.order_count}')
        print(f'最终价值: ${self.broker.getvalue():,.2f}')
        print(f'收益率: {(self.broker.getvalue()/self.p.initial_capital-1)*100:.2f}%')

# 回测运行
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # 加载数据
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(GridStrategy)
    
    # 设置初始资金
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    # 运行回测
    results = cerebro.run()
    
    # 绘制结果
    cerebro.plot(style='candlestick')
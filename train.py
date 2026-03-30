from src.pipeline import train_all


if __name__ == "__main__":
    result = train_all()
    print("训练完成：")
    print("日度指标:", result.daily_metrics)
    print("月度指标:", result.monthly_metrics)
    print("年度指标:", result.yearly_metrics)

import pandas as pd
import numpy as np
from pathlib import Path

def add_macro_features():
    print("Loading quality checked dataset...")
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = base_dir / "data" / "quality_checked" / "retail_prices_quality.csv"
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    print("Generating proxy historical petrol prices...")
    def get_petrol_price(year, month):
        base_2015 = 60.0
        base_2024 = 102.0
        yearly_increase = (base_2024 - base_2015) / 9.0
        year_base = base_2015 + ((year - 2015) * yearly_increase)
        np.random.seed(year * 100 + month)
        noise = np.random.uniform(-2.0, 2.0)
        if year == 2022 and month in [3, 4, 5, 6, 7]:
            noise += np.random.uniform(5.0, 10.0)
        return round(year_base + noise, 2)
        
    df['petrol_price'] = df.apply(lambda row: get_petrol_price(row['year'], row['month']), axis=1)
    
    print("Generating proxy historical geopolitical tension scores...")
    def get_tension_score(year, month):
        np.random.seed(year * 200 + month)
        score = np.random.uniform(20.0, 40.0)
        if year == 2020 and month in [3, 4, 5, 6, 7, 8]:
            score += np.random.uniform(40.0, 50.0)
        if year == 2022 and month in [2, 3, 4, 5, 6]:
            score += np.random.uniform(45.0, 55.0)
        if year in [2023, 2024] and month in [10, 11, 12, 1, 2]:
            score += np.random.uniform(20.0, 30.0)
        return round(min(score, 100.0), 1)

    df['geopolitical_tension'] = df.apply(lambda row: get_tension_score(row['year'], row['month']), axis=1)
    
    print("Saving updated dataset...")
    df.to_csv(data_path, index=False)
    print("Done! Added `petrol_price` and `geopolitical_tension` columns.")

if __name__ == "__main__":
    add_macro_features()

from pathlib import Path
import pandas as pd

app_dir = Path(__file__).parent
data_dir = app_dir / "data"
df = pd.read_csv(data_dir / "train.csv")

df.info()

df.head()
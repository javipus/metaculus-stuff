import pandas as pd

urls = {
  "pfizer" : "https://data.cdc.gov/resource/saz5-9hgg.csv",
  "moderna" : "https://data.cdc.gov/resource/b7pe-5nws.csv",
  "jj" : "https://data.cdc.gov/resource/b7pe-5nws.csv",
  }

data = {k: pd.read_csv(v) for k, v in urls.items()}
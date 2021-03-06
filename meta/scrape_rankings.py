import urllib, json, time
from bs4 import BeautifulSoup
import pandas as pd

# h/t https://stackoverflow.com/questions/14167352/beautifulsoup-html-csv#14167916
def get_rankings():
    url = "https://www.metaculus.com/rankings/"

    with urllib.request.urlopen(url) as f:
        soup = BeautifulSoup(f.read(), "lxml")
    
    table = soup.find('table', attrs={ "class" : "rankings-table"})
    rows = []
    
    for k, row in enumerate(table.find_all('tr')):
        rows.append([val.text.strip() for val in row.find_all('td' if k>0 else 'th')])
        
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.Score.astype(int, copy=False)
    df.set_index('Rank', inplace=True)

    return df

if __name__=='__main__':
    df = get_rankings()
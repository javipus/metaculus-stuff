import urllib, json, time, re
from bs4 import BeautifulSoup
import pandas as pd

# These two only use the API so they're not really "scraping"
def get_questions(dump_file='db/questions.json', verbose=True):

    url = "https://www.metaculus.com/api2/questions/"

    questions = {}
    fields = [
        'page_url',
        'author',
        'title',
        'title_short',
        'status',
        'resolution',
        'created_time',
        'publish_time',
        'close_time',
        'resolve_time',
        'possibilities',
        'can_use_powers',
        'last_activity_time',
        'activity',
        'comment_count',
        'votes',
        'metaculus_prediction',
        'community_prediction',
        'number_of_predictions',
        'prediction_histogram',
        'prediction_timeseries',
        ]


    while True:
        if verbose: print(f"Scraping questions from {url}")

        with urllib.request.urlopen(url) as f:
            data = json.load(f)
        
        for question in data['results']:
            questions[question['id']] = dict((k, question.get(k, None)) for k in fields)
        
        if data['next'] is not None:
            url = data['next']
            #time.sleep(lognorm(loc=.5, s=1).rvs())
        else:
            break

    if verbose: print(f"Done! Writing to {dump_file}")

    with open(dump_file, 'w') as f:
        json.dump(questions, f)

    if verbose: print("Done!\n\n")

def get_users(dump_file='db/users.json', verbose=True):
    url = "https://www.metaculus.com/api2/users/"

    users = {}
    fields = [
        'username',
        'level',
        'date_joined',
        'supporter_level',
        'formerly_known_as',
        ]

    while True:
        if verbose: print(f"Scraping users from {url}")

        with urllib.request.urlopen(url) as f:
            data = json.load(f)
        
        for user in data['results']:
            users[user['id']] = dict((k, user[k]) for k in fields)
        
        if data['next'] is not None:
            url = data['next']
            # time.sleep(lognorm(loc=.5, s=1).rvs())
        else:
            break

    if verbose: print(f"Done! Writing to {dump_file}")

    with open(dump_file, 'w') as f:
        json.dump(users, f)

    if verbose: print("Done!\n\n")

# h/t https://stackoverflow.com/questions/14167352/beautifulsoup-html-csv#14167916
def scrape_rankings(dump_file='db/top_users.csv'):

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

    if dump_file is not None:
        df.to_csv(dump_file)
    else:
        return df

def scrape_stats(users_file='db/users.json', stats_file='db/stats.json', verbose=True):
    stats = safe_json_load(stats_file)
    users = safe_json_load(users_file)
    
    for i, user in enumerate(users):
        if verbose: print(f"Fetching user {user}... {i+1} of {len(users)}")
        if user in stats:
            if verbose: print('Got them already, skipping!')
            continue
        try:
            stats[user] = scrape_user_stats(user)
        except Exception as e:
            print('-- WARNING --')
            print(e)
            print(' -- ')

    if verbose: print(f"Done! Writing to {dump_file}")

    with open(stats_file, 'w') as f:
        json.dump(stats, f)

    if verbose: print("Done!\n\n")

def scrape_user_stats(user, return_soup=False):
    url = f"https://www.metaculus.com/accounts/profile/{user}/"

    with urllib.request.urlopen(url) as f:
        soup = BeautifulSoup(f.read(), "lxml")

    preds = get_preds(soup)
    comms = get_comms(soup)
    auth = get_auth(soup)
    ach = get_achievements(soup)

    ret = {
        'predictions': preds,
        'comments': comms,
        'authored': auth,
        'achievements': ach,
    }

    if return_soup:
        return ret, soup
    else:
        return ret

def get_preds(soup):
    rgx = r"(\d+) predictions on (\d+) questions \((\d+) resolved\)"
    tag = soup.find('span', text='predictions')
    if tag:
        raw = tag.next.next.next.text
        return dict(zip(('predictions', 'questions_predicted', 'resolved'), map(int, re.search(rgx, raw).groups())))
    return

def get_comms(soup):
    rgx = r"(\d+) comments on (\d+) questions"
    tag = soup.find('span', text='comments')
    if tag:
        raw = tag.next.next.next.text
        return dict(zip(('comments', 'questions_commented'), map(int, re.search(rgx, raw).groups())))
    return

def get_auth(soup):
    tag = soup.findAll('a', text=re.compile('.* active questions'))
    
    if not tag:
        return
    else:
        assert len(tag) == 1
    
    raw = tag[0].text
    return int(raw.split()[0])

def get_achievements(soup):
    rgx = r".*profileAchievements.*"
    script = soup.findAll('script', text=re.compile(rgx))
    assert len(script) == 1
    raw = script[0].string

    for line in raw.split(';\n'):
        if re.match(rgx, line):
            return json.loads(line.split('=')[-1])
    return

def safe_json_load(fpath, return_empty=True, mode='r'):
    try:
        with open(fpath, mode) as f:
            ret = json.load(f)
    except OSError:
        if not return_empty:
            raise
        ret = {}
    return ret

if __name__=='__main__':

    # get_questions()
    # get_users()

    scrape_rankings()
    scrape_stats()
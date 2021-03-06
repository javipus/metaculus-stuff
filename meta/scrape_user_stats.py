import urllib, re, json, time
from bs4 import BeautifulSoup
from scipy.stats import lognorm
# TODO logging

def get_stats(user, return_soup=False):
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

    stats = safe_json_load('stats.json')
    users = safe_json_load('users.json')
    
    for i, user in enumerate(users):
        print(f"Fetching user {user}... {i+1} of {len(users)}")
        if user in stats:
            print('Got them already, skipping!')
            continue
        try:
            stats[user] = get_stats(user)
        except Exception as e:
            print('-- WARNING --')
            print(e)
            print(' -- ')
        
        time.sleep(lognorm(loc=.5, s=1).rvs())
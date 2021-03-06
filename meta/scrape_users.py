import urllib, json, time
from scipy.stats import lognorm
# TODO logging

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
    print(url)

    with urllib.request.urlopen(url) as f:
        data = json.load(f)
    
    for user in data['results']:
        users[user['id']] = dict((k, user[k]) for k in fields)
    
    if data['next'] is not None:
        url = data['next']
        time.sleep(lognorm(loc=.5, s=1).rvs())
    else:
        break
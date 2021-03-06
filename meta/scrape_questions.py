import urllib, json, time
from scipy.stats import lognorm
# TODO logging

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
    print(url)

    with urllib.request.urlopen(url) as f:
        data = json.load(f)
    
    for question in data['results']:
        questions[question['id']] = dict((k, question.get(k, None)) for k in fields)
    
    if data['next'] is not None:
        url = data['next']
        #time.sleep(lognorm(loc=.5, s=1).rvs())
    else:
        break

with open('questions.json', 'w') as f:
    json.dump(questions, f)

print(sum([q['number_of_predictions'] for q in questions.values()]))
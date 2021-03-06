1. Will voter turnout for the US 2020 presidential election be higher than 2016? Went with my answer to the other q for which I used 538's prediction. The economist model does not output turnout so I didn't use it.
    - Just took Nate Silver's numbers; this yields `1 - norm(q50, sigma).cdf(55.5 + 2)` where `sigma = (q90-q50) / norm(0,1).ppf(.9)` and `q50 = 154 / VAP`, `q90 = 165 / VAP`, `VAP = 257.6`.
    - I screwed up and thought q90 was actually q75 so my first answer (65%) was wrong. I submitted the right one in the final question of the form but they'll choose one at random to score this question - dammit

1. Will the S&P 500 close higher in 2020 than 2019? Look at option prices:
    - 75% from ergo's notebook

1. Democratic majority in the US Senate?
    - 77%, `mean(mean(forecast.fivethirtyeight, forecast.economist), forecast.gj_superforecasters)`, with 538 at 78% (classic, not deluxe); economist at 74%; and superforecasters at 78%.

1. Will there be a terrorist attack in an OECD founding member state causing more than 3 deaths between November 3rd 2020 and up until March 1st 2021?
    - 78% = mean(74%, 82%) and `82% = 1 - poisson(52 / (10 * 4)).pmf(0)` as there've been 52 attacks over the last 10 years = 40 quarters. The 78% comes from adjusting that 52 down by a 0.8 factor to account for covid-related loss of mobility, people in public places (making attacks less attractive)

1. For the month of January 2021, will Consumer Confidence in the United States return to optimism?
    - Fitted a normal distribution to data from August (minimum). Yes, N=2 (3 data points, 2 in diff). Then sampled X_i iid and calculated the empirical probability that `X_i + X_j + X_k > 100 - y[-1]`. It was 43%.

1. By 1 January 2021, will the rate of new confirmed deaths from COVID-19 in Sweden be higher than Denmark?
    - Just base rate: since the beginning of the pandemic, deaths in Sweden have been higher than in Denmark 85% of the time

1. Will there be at least 10 fatalites caused in post-election political violence in the United States?
    - Deferred to the 75% from [this study's](https://drive.google.com/file/d/1CbkMoNE7eftNwHl5yC71MDmeXJTvzfk9/view) CDF.

1. Will either Joe Biden or Donald Trump concede in the 2020 US presidential elections by November 17th?
    - See [this guesstimate](https://www.getguesstimate.com/models/17096)

1. By December 25th, 2020, will cumulative confirmed cases of COVID-19 in the United States exceed 12 million?
    - see `covidhub.py`

1. Will a SARS-CoV-2 vaccine candidate that has demonstrated an efficacy rate >75% in a n>500 RCT be administered to 10M people by March 1st 2021?
    - just copied the community CDF: 33% - couldn't add superforecaster's predictions bc they use different specification (25M, no RCT wording, different time-frames)

1. By 1 March 2021, will Israel and Saudi Arabia announce a peace or normalization agreement?
    - 70%

1. Will North Korea launch an intercontinental ballistic missile between November 3rd 2020 and March 1st 2021?
I am mostly going off base rates here.

In the last decade, which coincides with the time KJU has been in power, there have been 3 tests. That's 0.1 every 4 months, which is the time left until question deadline.

However, no tests have been conducted since 2018 because the Korean government agreed to a moratorium that they will no longer be observing. This together with the likely change in US administration following the election makes me think they will ramp up their military efforts, since Biden may be less isolationist than Trump. So I'm increasing the base rate by 50% to arrive at my final 15% forecast.

1. Will polling in the US presidential election miss the true results by 3% percentage points or more?
    - I have no strong priors so I'm going full frequentism (sorry!). I did `1-np.diff([norm(*norm.fit(xs)).cdf(r) for r in (-3,3)])[0]` with `xs` being the difference between RCP's average margin and actual margin (see `rcp.csv`) since 2004 (only 4 datapoints) and got Pr[MAE > 3%] = 9%, so I'm going with that

1. On January 1st, 2021, will Americans' opposition to Black Lives Matter be higher than 40%? It's currently at 39% and has been stable for a while. Confidence interval in the graph overlaps with 40%. Let's assume this:
This number is currently at 39%, so almost within polling error of 41%. However, the sample size for this panel is pretty big (>100k) so I consider the probability of the question resolving positive because of sampling noise to be vanishingly small.

Moreover, the number seems to have stabilized post-Geroge Floyd protests and I don't expect it to jump around in the absence of similar events. My base rate for protests of that magnitude is about 2 per year based on the history of the BLM movement. However, a drawn-out ballot count after the election makes protests far more likely, so I'm multiplying this base rate by 2. This yields my 20% estimate.

Crucially, I am assuming that _any_ protests would increase the number of people opposing BLM. I do this because I believe polarization makes it more likely that current undecideds would stop being so and split themselves ~evenly between pro and against camps.
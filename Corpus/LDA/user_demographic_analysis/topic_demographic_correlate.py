import scipy.stats
import numpy as np
from sklearn import preprocessing

topic_names = ["Effects/prohibition vis-a-vis alcohol/tobacco",
"UNCLEAR",
"Hemp: legality and uses",
"Party politics and ideology",
"Quantities and limits",
"Legal status and common use",
"Govt. power and ind. rights/freedoms",
"Border, organized crime & Latin influence",
"State vs. federal legalization",
"Media campaigns & portrayals",
"enforcement vis-a-vis violent crimes",
"Legal market & economic forces",
"Addiction potential & gateway status",
"Reasoning and arguments",
"State-level legal. timelines",
"Police car searches",
"Medical marijuana effects and access",
"Cannabis types and use methods",
"Marijuana use and the workplace",
"FDA schedules",
"UNCLEAR",
"Continuity of US drug & foreign policy",
"Age and familial relations",
"Marijuana and finances",
"User stereotypes and life outcomes",
"Private interests & the prison industry",
"UNCLEAR",
"Legalization across US and the world",
"Police house searches & seizure",
"Legal procedures",
"Emotional and life impact",
"Reddit moderation",
"Everyday enforcement encounters",
"UNCLEAR",
"UNCLEAR",
"Drug testing",
"Judgments of character",
"Imprisonment over marijuana",
"Electoral politics & parties",
"UNCLEAR",
"Local/state regulations",
"Health and opinion research",
"DUI effects & enforcement",
"Racial/minority disparities",
"Federal Court Processes",
"Smoking methods, health effects and bans",
"UNCLEAR",
"Enforcement & observance",
"Gun versus marijuana regulations",
"Expletives-laden discourse"]


correlations = open("correlations.txt", "a")

for topic in range(50):
    if topic_names[topic] != "UNCLEAR":
        file_name = str(topic) + "_results.txt"
        f = open(file_name, "r")
        rel = []
        pol = []
        perc = []
        lines = f.readlines()
        for x in lines:
            x = x.split(" ")
            d = x[2]
            rel.append(float(d))
            pol.append(float(x[4]))
            perc.append(float(x[0]))
        x_array = np.array(perc)
        normalized = preprocessing.normalize([x_array])
        cor_rel = scipy.stats.pearsonr(rel, normalized[0])
        cor_pol = scipy.stats.pearsonr(pol, normalized[0])

        name = topic_names[topic]

        res = str(cor_rel[0]) + "   " +  "  " + str(cor_pol[0]) +  "  "  + name + "\n"
        correlations.write(res)
        f.close()
correlations.close()

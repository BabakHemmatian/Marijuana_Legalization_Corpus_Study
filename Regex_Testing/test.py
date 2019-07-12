import re

sentence = ">He is not cheating.It`s not really a question of whether the other guy is cheating. This is simply a matter of sucking the fun out of the game. If a guy TR`s a dude for his best player, giving up crap in return, good for that guy. But if a team becomes absolutely unstoppable, because someone is stupid, that takes the fun away for me, personally. Almost like Jacksonville fans right now. Why even watch the game, Sunday? The `Fantasy Dream Team` is playing them. It`s almost not even fair. Sure, Denver *signed* all of those players, and everything was done within the rulebook. No one is saying Denver is cheating by having that ridiculous offense, but it`s not going to be an entertaining game at *all* to watch. The difference being, I paid money to be in the fantasy league. I don`t have financial investment in the betterment of the Jaguars.I don`t like having to forfeit my money just because someone else was a moron. Vetoes being abolished would change the game, in my opinion, for the worst. Should the `four vote` thing be the `default` veto? No. But should vetoes be a thing of the past? Absolutely not.First, you need to be in a league with a commissioner whose judgment you trust, and who will listen to the league members cases before making a ruling.In my league, the four-vote `veto` is currently in place. But the twist is, once the four votes are accumulated, the trade is in limbo, and goes to a group chat. The commissioner and anyone who wants to debate the validity of the trade just hash it out with words. This year, a guy traded Steven Jackson (*after* his injury) and Lamar Miller for Jamaal Charles and Eddie Lacy (right after his concussion, so he was still injured). This got voted off for debate, because even at 100%, Jackson isn`t close to Charles` level. But we debated it, and we were dealing with a guy who thought Lacy would turn out well, and he happened to be a gigantic Falcons fan. So, he was happy bringing in those guys. It ended up being pushed through and processed, even though many of us weren`t comfortable with it, but what are you gonna do?The fact is, without any sort of veto process whatsoever, there couldn`t have even been a talk. And that could just be damaging. The threat of `get too crazy and this isn`t going to happen` has to be there, or you`ll introduce collusion. What would stop me from talking to my friend and saying `Give me your two starting WRs, I`ll give you my 3rd RB and 3rd WR, and we`ll split the pot at the end of the year?` Sure, it`s collusion, but no one would be able to prove it if they didn`t hear us talking, and didn`t see me hand him money. Vetoes are essential, if nothing else, in minimal fashion.**EDIT:** sorry, just realized how much I typed, here. **TL;DR:** Without a veto in some form, money leagues can implode"
marijuana = []
legality = []

with open("marijuana_7.txt",'r') as f:
    for line in f:
        marijuana.append(re.compile(line.lower().strip()))

with open("legality_7.txt",'r') as f:
    for line in f:
        legality.append(re.compile(line.lower().strip()))

for exp in marijuana:
    if not exp.search(sentence) is None:
        print(exp)
for exp in legality:
    if not exp.search(sentence) is None:
        print(exp)


from utils import personal_tag_list
import pandas as pd

import random

def generate_personality_tags(seed = 42):
    rng = random.Random(seed)
    tag_len = random.randint(1, 10)
    selected_tags = rng.sample(personal_tag_list, tag_len)
    return ",".join(selected_tags)

data = pd.DataFrame({
    "user_id": range(2000),
    "personality_tags": [generate_personality_tags(idx) for idx in range(2000)],
    "hobby_tags": [""] * 2000,
    "game_tags": [""] * 2000
})

def calculate_bm25_score(tf, doc_len):
    k1 = 2
    b = 0.8
    return (k1+1) * 1.0 * tf / (k1 * (1-b + b * doc_len*1.0/6.0) + tf)

for tag in personal_tag_list:
    data[tag + "_score"] = data["personality_tags"].apply(lambda x:
        calculate_bm25_score(int(tag in x), len(x.split(","))))
data.to_csv("data/candidate_demo.csv", index=False)

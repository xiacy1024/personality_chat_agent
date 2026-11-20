import json

import pandas as pd
data = pd.read_csv("data/candidate_demo.csv")

from utils import *
from agent.personality.dimension_score import personality_dimention_node
from agent.personality.embedded_tag import get_emb_sim
def calculate_bm25_score(tf, doc_len):
    k1 = 2
    b = 0.8
    return (k1+1) * 1.0 * tf / (k1 * (1-b + b * doc_len*1.0/6.0) + tf)

def calculate_score(row, preference_json):
    personality_tags, hobby_tags, game_tags = row["personality_tags"], row["hobby_tags"], row["game_tags"]
    hobby_tag_score = len(set(hobby_tags.split(",").intersection(preference_json.get("hobby_prefered", []))))
    game_tag_score = len(set(game_tags.split(",").intersection(preference_json.get("game_prefered", []))))

    print(row["personality_tags"])
    return 0.0

def get_dim_tag_score_dic(prefered_list, not_prefered_list):
    personality_preference_dic = {
        "personality_prefered": prefered_list,
        "personality_not_prefered": not_prefered_list
    }
    print("用户性格偏好：", personality_preference_dic)
    personality_dimension_dic = personality_dimention_node(personality_preference_dic)
    print("期望性格维度分值：", personality_dimension_dic)
    personality_dimension_dic = {key: personality_dimension_dic[key]
         for key in personality_dimension_dic if personality_dimension_dic[key] > 0
    }
    dim_map_dic = {
        '外向': extra_dim_dic, '内向': intra_dim_dic, '计划性': judge_dim_dic, '开放性': open_dim_dic, '宜人性': agreeable_dic, '情绪稳定性': stable_dic}

    dim_tag_score_dic = {}
    for dim_key in personality_dimension_dic:
        dim_score = personality_dimension_dic[dim_key]
        for tag_key in dim_map_dic[dim_key]:
            dim_tag_score_dic[tag_key] = dim_tag_score_dic.get(tag_key, 0) + dim_score
    return dim_tag_score_dic

def get_similar_tags(preference_list):
    similar_tag_list = []
    for desc in preference_list:
        similar_tag_list.extend(get_emb_sim(desc, TAG_SIMILARITY_CHECK))
    return list(set(similar_tag_list))

def get_personality_score(row, dim_tag_score_dic, similar_tag_list):
    dim_score = 0
    tag_similarity_score = 0
    for tag_key in dim_tag_score_dic.keys():
        dim_score += row[tag_key + "_score"] * dim_tag_score_dic[tag_key]
    for tag_key in similar_tag_list:
        tag_similarity_score += row[tag_key + "_score"]
    return dim_score * PERSONALITY_DIM_WEIGHT + tag_similarity_score * PERSONALITY_TAG_SIMILARITY_WEIGHT

def main():
    with open("data/preference.json", "r", encoding='utf-8') as f:
        preference_json = json.load(f)

    dim_tag_score_dic = get_dim_tag_score_dic(preference_json.get("personality_prefered", []),
                          preference_json.get("personality_not_prefered", []))
    similar_tag_list = get_similar_tags(preference_json.get("personality_prefered", []))

    print("期望性格维度分值：", dim_tag_score_dic)
    print("sbert匹配的所有性格标签：%s" % ",".join(similar_tag_list))

    def get_intersect_cnt(candidate_tag_str, preference_list):
        if candidate_tag_str != candidate_tag_str:
            return 0
        return len(set(candidate_tag_str.split(",")).intersection(preference_list))

    data["personality_score"] = data.apply(lambda row:
            get_personality_score(row, dim_tag_score_dic, similar_tag_list), axis = 1)
    data["hobby_score"] = data["hobby_tags"].apply(
        lambda x: get_intersect_cnt(x, preference_json.get("hobby_prefered", [])))
    if data["hobby_score"].max() != 0.0:
        data["hobby_score"] = data["hobby_score"]*1.0/(data["hobby_score"].max())
    data["game_score"] = data["game_tags"].apply(
        lambda x: get_intersect_cnt(x, preference_json.get("game_score", [])))
    if data["game_score"].max() != 0.0:
        data["game_score"] = data["game_score"] * 1.0 / (data["game_score"].max())
    data["score"] = data["personality_score"] * PERSONALITY_WEIGHT + data["hobby_score"] * HOBBY_WEIGHT + data["game_score"] * GAME_WEIGHT

    top_personal_tags = data.sort_values("score", ascending=False).head(10)["personality_tags"]
    print("\n最终top10用户的性格标签：")
    for tag in top_personal_tags:
        print(tag)

if __name__ == '__main__':
    main()
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('./agent/personality/shibing624/text2vec-base-chinese')
model = SentenceTransformer('shibing624/text2vec-base-chinese')

from utils import personal_tag_list
from agent.personality.api_extract import personality_tag_list_node

personality_tag_embed_dic = {}
for tag in personal_tag_list:
    personality_tag_embed_dic[tag] = model.encode([tag])

def get_emb_sim(description, check = True):
    sim_tag_dic = {}
    desc_embedding = model.encode([description])
    for tag in personal_tag_list:
        similarity = util.cos_sim(desc_embedding, personality_tag_embed_dic[tag])
        # if similarity > 0.55:
        sim_tag_dic[tag] = similarity
    sim_tag_items = list(sim_tag_dic.items())
    sim_tag_items.sort(key=lambda x: x[1], reverse=True)
    output_list = [ item[0] for item in sim_tag_items[:5] if item[1]>0.55]
    # print(description, sim_tag_items)
    if check:
        api_tag_list = personality_tag_list_node(description)
        print("用户性格描述：%s" % description)
        print("api返回结果：%s" % ",".join(api_tag_list))
        print("embedding比对结果：%s" % ",".join(output_list))
        if api_tag_list:
            dcg = sum([sim_tag_dic[tag] for tag in api_tag_list]).numpy().tolist()[0][0]
            idcg = sum([item[1] for item in sim_tag_items[:len(api_tag_list)]]).numpy().tolist()[0][0]
            ndcg = dcg * 1.0 / idcg
            print("ndcg: %.2f" % ndcg)
    return output_list

# print(embed_dic)


from agent.chat import chat_node
from agent.category_extract import category_extract_node
from agent.extract_info import run_extract_workflow

def main():
    chat_record = []
    info_questioned_dic = {"性格": False, "爱好": False, "游戏": False, "职业": False}
    info_answered_dic = {"性格": False, "爱好": False, "游戏": False, "职业": False}
    while True:
        last_category_list = []
        chat_record = chat_record[-20:]
        last_category_dic = category_extract_node(last_chat_list=chat_record[-2:])
        print(last_category_dic)

        for category_key in last_category_dic.keys():
            info_questioned_dic[category_key] = True
            if last_category_dic[category_key]:
                info_answered_dic[category_key] = True
                last_category_list.append(category_key)

        if last_category_list:
            run_extract_workflow(chat_record[-2:], last_category_list, "user_001")

        question = chat_node(
            hist_chat_list = chat_record,
            info_questioned_dic = info_questioned_dic,
            info_answered_dic = info_answered_dic
        )
        print("小牵：", question)
        answer = input("你：")
        chat_record.extend([{"role":"小牵", "content":question}, {"role":"用户", "content":answer}])

if __name__ == '__main__':
    main()


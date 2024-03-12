import ai_model_to_train

mining_qa = {
    "What is mining?": "Mining is the process of extracting valuable minerals or other geological materials from the earth.",
    "How is coal mined?": "Coal is typically mined through two main methods: surface mining and underground mining.",
    "What safety measures are taken in mining?": "Mining operations prioritize safety by providing proper training, equipment, and implementing safety protocols.",
    "Tell me about a famous mine.": "The Kimberly Diamond Mine in South Africa, also known as the Big Hole, is one of the world's most famous diamond mines.",
    "What are the environmental impacts of mining?": "Mining can have significant environmental impacts, including habitat destruction, water pollution, and soil erosion.",
    "What is mineral exploration?": "Mineral exploration is the process of searching for valuable minerals or resources in a specific area.",
    "What are the mining laws in India?": "In India, mining is regulated by various laws, including the Mines and Minerals (Development and Regulation) Act, 1957, and the Mineral (Auction) Rules, 2015.",
    "How are mining leases granted in India?": "Mining leases in India are typically granted through auctions, and the process is governed by the Mineral (Auction) Rules, 2015.",
    "What are the environmental regulations for mining in India?": "Mining in India must comply with various environmental regulations, including the Environment Impact Assessment (EIA) process and the Forest Conservation Act.",
}


questions = ai_model_to_train.question
answer = ai_model_to_train.answer
possible_yes = ['y','Y','yes','Yes']
possible_no = ['N','n','no','No']
def bingbing():
    Qinp = input("What would you like to search today?: \n")
    # the inp is input which will be changed to an input data type when the testing is completed.
    #Qinp = ('What is the MMRD Act?')
    while True:
        if Qinp in questions:
            index_A = questions.index(Qinp)
            print("Question asked:\n",'\n',Qinp,'\n')
            print('Result:\n')
            print(answer[index_A])
            satis = input("Would you like to search for anything else ?: ")
            if satis in possible_yes:
                continue
            elif satis in possible_no:
                print("Thank you, Have a pleasant hour")
                break
            else:
                print("malfunction in code")
                break
        else:
            resp = input("You asked:\n",'\n',Qinp,'\n'*2,'Are you sure if this is the right question? Y = yes; N = no: ')
            if resp in possible_yes:
                print("I am sorry, i can't help you with that question yet....")
                break
            elif resp in possible_no:
                print("alright, please input ur question query again")
                continue
            else:
                print("malfunction in code")


bingbing()
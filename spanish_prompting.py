import os
from glob import glob
from pathlib import Path
import torch
import json


# This script is built on top of talkative-llm wrapper and refers to usage.py script in the talkative-llm repo. More information in the talkative-llm repository repository: https://github.com/minnesotanlp/talkative-llm
from  talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)


CONFIG_DIR = Path(__file__).parent.parent

''' Uncomment for baseline prompting
head_prompt="Translate the sentence into Spanish." 
file_path = "./data/json/spanish_baseline_prompt_output.json"
'''

''' Uncomment for step-by-step prompting 
head_prompt="Translate the sentence into Spanish following these steps:\nStep 1. Identify the idiom.\nStep 2. Find an equivalent idiom in the target language. If there is no equivalent idiom, translate the idiom according to its meaning.\nStep 3. Give a translation of the sentence including that idiom."
file_path = "./data/json/spanish_step_by_step_prompt_output.json"
'''

'''Uncommoentfor CoT few shot prompting
head_prompt = "User: Cuando aprendo espanol en 'The Spanish Academy' los examenes son pan comido. \nAssistant: Step 1. The idiom in the sentence is 'pan comido'\nStep 2. The idiom translates to 'a piece of cake' \nStep 3. Full sentence translates to 'When I learn Spanish at the Spanish Academy exams are a piece of cake.', User: Siempre que intento concentrarme en la clase de matemáticas, termino estando en las nubes.\nAssistant: Step 1. The idiom in the sentence is 'estar en las nubes.'\nStep 2. The idiom translates to 'have one's head in the clouds,' which means to be distracted or daydreaming.\nStep 3. Full sentence translates to 'Whenever I try to concentrate in math class, I end up having my head in the clouds.'"
file_path = "./data/json/spanish_cot_prompt_output.json"
'''

sentences = [
  "It is up to Castro and Roque to show what they are made of and get in the bandwagon.",
"Despite knowing it was almost impossible to win, Peter was still fighting for a lost cause.",
"After several failed attempts, John decided to throw away the towel and leave the French course.",
"I lost my job, but I got an even better chance. It is true that when a door is closed, a window opens.",
"Better don't look for trouble researching that, remember that curiosity killed the cat.",
"He got that job not because of talent, but because he has a plug in the company.",
"You should be grateful to your boss and not criticize him so much; don't bite the hand that feeds you.",
"It's fascinating  that the mechanic has his car broken down. As it is said, the shoemaker’s son always goes barefoot.",
"I decided to stay in my current job; a devil you know is better than the one you don't.",
"Children always go crazy playing when the teacher leaves the classroom. When the cat's away, the mouse will play.",
"She always has ideas out of the ordinary; At school she was considered a freak.",
"The restaurant we went to for dinner in the boonies, it took us almost an hour to get there.",
"After selling his company, he's now loaded and can buy whatever he wants.",
"Carlos's old car finally kicked the bucket in the middle of the trip.",
"John is as good at math as his father was; the apple doesn't fall far from the tree.",
"By buying the train tickets in advance, we saved money and secured our trip, killing two birds with one stone.",
"I'd rather not know what happened at that party because, as the saying goes, what the eye doesn’t see, the heart doesn’t grieve over.",
"That watch looks very expensive, but remember that all that glitters is not gold.",
"Although he was a nobody in his village, he moved to the city and managed to become famous.",
"After talking to my friend about my problems, I felt better; it's true that two in distress makes sorry less.",
"Juan decided to come out of the closet and tell his parents about his sexual orientation.",
"When she forgot about our anniversary, it was the straw that broke the camel's back and I decided to end the relationship.",
"She loves classical music and he loves heavy rock, but there's nothing written about tastes.",
"He was happy as a clam when he got the news that he had passed the test.",
"It seemed like a simple and unpretentious place, but the food was incredibly good. Undoubtedly, looks can be deceiving.",
"I like to try foods from different cultures because variety is the spice of life.",
"He became a big fish in the industry after his invention revolutionized the market.",
"When you try to open that jar, remember brain over brawn.",
"I always like to speak plainly, to call a spade a spade, so we all know where we stand.",
"Thank you for helping me with the move. Remember, you scratch my back I’ll scratch yours.",
"He decided to invest in a new business, thinking that nothing ventured, nothing gained.",
"Don't worry about his threats, barking dogs never bite.",
"I always carry an umbrella in my bag, because it is better to prevent than to cure.",
"The kids were being a clown in the playground and couldn't stop laughing.",
"Even though I got fired, I found a better job, so every cloud has a silver lining.",
"No matter what method you use to study, all roads lead to Rome.",
"It's best not to deal with him when he's angry because he's a bad apple.",
"At the party, Juan told an icebreaker joke and everyone relaxed.",
"Look at those corrupt politicians together; birds of a feather flock together.",
"You've finally cleaned out your room, better late than never.",
"She's not only my wife, she's also my better half.",
"I screwed up when I forgot our wedding anniversary.",
"He changed teams when his team started to lose; he's a real flip-flopper.",
"I don't need to explain to you why, a word to the wise is enough.",
"I went out for a moment, and when I came back my brother had taken my place on the sofa. You snooze, you loose.",
"At the celebration he got drunk; he couldn't even speak.",
"My uncle is always doing weird things, he's crazy.",
"Don't worry about tomorrow's exam, it's going to be a piece of cake.",
"I'm going to take an umbrella just in case, it looks like it might rain.",
"John is always sucking up to the boss to get what he wants.",
"When her colleague got sick, Maria lended his hand to help finish the project.",
"In that big group, I feel worthless, no one notices if I'm there or not.",
"That watch cost me an arm and a leg, but it's worth every penny.",
"My brother calls me every now and then; he is very needy.",
"Don't complicate the situation by making things more complicated than necessary, it's quite simple.",
"There were very few of us at the meeting, almost no one attended.",
"The hotel was in the boonies, we had to walk a lot to get to the center.",
"Since the beginning of winter, it feels like hell, I have not stopped getting sick.",
"That new car is too cool, everyone turns to look at it.",
"He made a mountain out of a molehill trying to mediate those family problems.",
"Although everyone advises him otherwise, he is still stubborn about not selling the house.",
"That politician is a thief, he's always involved in corruption scandals.",
"Your phone is ancient!",
"Novice soldiers are often thrown under the bus in large conflicts.",
"Don't talk to the boss today, he’s in a bad mood because sales are down.",
"After so many months in the gym, he's thin as a rail.",
"With all that rain, I came home and I soaked to the bone.",
"He doesn't give a damn what other people think of his style.",
"Holey Moley! Did you see that? The car almost crashed!",
"I'm fed up with your complaints.",
"Just thinking about the paella my grandmother makes makes my mouth water.",
"With all these debts, Its up to my neck.",
"If you need help with the move, I can lend you a hand.",
"He got completely immersed into the project and hadn't left his office in days.",
"I was so nervous during the interview that I couldn't get my foot down.",
"Although she received bad news, she looked on the bright side.",
"She can talk a lot, she never runs out of topics of conversation.",
"Today I woke up on the wrong side of the bed and everything went wrong for me.",
"I was so worried that I didn't sleep a wink all night.",
"When he was told he had won the prize, he was over the moon.",
"Since he lost his job, he can't seem to catch a break; you know, when it rains, it pours.",
"We ended our relationship amicably; it was nice while it lasted.",
"It's not the kind of car I'd buy, but to each his own.",
"They thought they had beaten me, but I won the trial. He who laughs last, laughs best.",
"He's always late for his appointments, he's very slow at doing everything.",
"Love takes two to tango, and if you two like each other, I won't stop you.",
"Don't worry about staying for dinner, there is always room for more.",
"Don't overwhelm yourself by working all night, getting up early doesn’t make the sun rise sooner.",
"Cancel our vacation? No way!",
"When I asked him about the money, he beat around the bush and never gave a clear answer.",
"I would have liked to win, but I lost. That's life.",
"Don't start looking for problems where there aren't any, let sleeping dogs lie.",
"We decided to leave the problems behind and wipe the slate clean in our friendship.",
"After the breakup, he went to the bar to drown his sorrows with alcohol.",
"I want our relationship to move to the next stage, but all in due time.",
"After working twelve hours straight, he was tired.",
"At the party, she tried to hook up with someone with her new look.",
"If you need help with the move, please count me in.",
"When he suggested changing the marketing strategy, he really hit the nail on the head.",
"It started raining just as we left, here we go again.",
"Tuesday is a public holiday, so we're going to have a long weekend and won't get back to work until Wednesday.",
"I went to buy tickets, but they closed just before I arrived and the door was slammed in my face.",
"Despite his constant failures, he always manages to come back: bad ones never die.",
"After working all day without eating, beggars can’t be choosers; this old bread tastes delicious.",
"With this storm, I got home soaked to the bone.",
"He's lost so much weight that he's thin as a rail.",
"After months in the gym, he’s very hot.",
"I don't give a damn.",
"He turned into a tomato when he forgot the lines on the play.",
"Your phone is ancient!",
"Don't worry about that exam; it will be a piece of cake for you.",
"I doubt they're awake yet. They go to sleep early.",
"Since he lost his job, he's had problems all over the place; when it rains it pours.",
"The car salesman tried to rip me off.",
"My son does nothing but complain and rebel; he's definitely at a difficult age.",
"You're very quiet today, has the cat eaten your tongue?",
"That new computer cost me an arm and a leg, but it's state-of-the-art.",
"I'm fed up with your complaints.",
"My aunt talks up a storm; she can spend hours telling stories.",
"I was so nervous about the interview that I didn't sleep a wink all night.",
"Your argument doesn't make any sense, you should organize your ideas better.",
"She doesn't mince words, she always speaks her mind, no matter the consequences.",
"I burned the midnight oil studying for the final physics exam.",
"Those two are joined at the hip; they are always seen together.",
"My friend told me he met Kim Kardashian, but I think he was pulling a leg.",
"Don't drown in a glass of water. Don’t sweat the small stuff.",
"My girlfriend left me because of her ex. - when life makes you lemons, make lemonade.",
"You're overcomplicating the situation, you're making something more complicated than it really is.",
"You must act fast and decisive, remember that opportunities will pass you by.",
"I don't know what decision to make about the new contract; I'm going to sleep on it.",
"It is ironic that the mechanic always has his car broken down, when you’re an expert in something, you don’t apply it to your own life.",
"I lost my job, but then they offered me a better one, just what they say: when a door closes, a window opens.",
"You can promise me anything, but remember that actions speak louder than words.",
"I decided to accept the current job offer rather than wait for a better one: better to focus on what you have instead of what you don’t.",
"He took on more than he could handle when trying to organize a wedding in a month.",
"Don't worry about that breakup; a new person will make you forget the old one.",
"He tried to give me financial advice, but he really should focus on doing what he is good at.",
"I don't like to make a lot of noise when I leave, I prefer to leave without saying goodbye.",
"Roberto's son also had an affair. - an apple doesn’t fall far from the tree.",
"Look at those cheaters playing cards together; birds of a feather flock together.",
"After winning the prize, he rested on his laurels and stopped trying.",
"Doing more tasks than necessary is like pouring water into the sea.",
"Arguing about politics only adds fuel to the fire at family gatherings.",
"You must be missing a screw to think you can run a marathon without training.",
"Politicians tend to beat around the bush when it comes to tackling contentious issues.",
"After seeing the benefits, everyone wanted to jump on the technology investment bandwagon.",
"She's always optimistic, she sees everything in roses.",
"I was happy as a clam after receiving the news of his promotion.",
"He's in a bad mood today, you better not talk to him too much.",
"After the marathon, I was done; I couldn't move a muscle.",
"Do you want to go to the gym? - No. I'm exhausted.",
"By the time he heard about the betrayal, he was extremely angry.",
"He was over the moon when he found out he was going to be a father.",
"Avoid talking to the boss today, he's in a bad mood about the sales results.",
"I was stunned when they announced their engagement.",
"It's not the kind of car I'd buy, but to each their own.",
"When Jorge suggested that the problem was with the software and not the hardware, he really hit the nail on the head.",
"We were at the picnic when it started raining cats and dogs; we had to pack everything up quickly and seek shelter.",
"I screwed up by telling Sara that the party was a surprise; I didn't know it was a secret.",
"By studying Spanish, I'm killing two birds with one stone: improving my resume and preparing for the trip to Mexico.",
"He flew off the handle when he saw that the dog had smashed the couch cushions.",
"The birthday girl's dad spared no expense with this party."]

PROMPTS=[]
for i in range(len(sentences)):
    PROMPTS.append(head_prompt+' '+sentences[i])
    
def cohere_caller():
    config_path = CONFIG_DIR / "talkative-llm" / "configs" / "cohere" / "cohere_llm_example.yaml"
    print(config_path)
    caller = CohereCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    
    
def openai_caller_completion():
    config_path = CONFIG_DIR / 'talkative-llm' / 'configs' / 'openai' / 'openai_completion_example.yaml'
    caller = OpenAICaller(config=config_path)
    results = caller.generate(PROMPTS)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Data written to {file_path} successfully.")
    del caller


def openai_caller_chat():
    config_path = CONFIG_DIR /'talkative-llm' / 'configs' / 'openai' / 'openai_chat_example.yaml'
    caller = OpenAICaller(config=config_path)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
        {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
        {'role': 'user', 'content': 'Where was it played?'}
    ]
    results = caller.generate(inputs=messages)
    del caller



def huggingface_caller(config_path: str):
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def huggingface_caller():
    config_path = os.path.join(CONFIG_DIR, 'huggingface', 'huggingface_llm_example.yaml')
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mpt_caller():
    config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
    caller = MPTCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
if __name__=="__main__":
    
    # cohere_caller()
    openai_caller_completion()
    

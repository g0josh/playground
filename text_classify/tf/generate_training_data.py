import json

KBASE_PATH = '/home/cbarobotics/dev/catkin_ws/src/chip_alive/pyreem/json_interaction_files/knowledge_base_new.json'
OUTPUT_PATH = '/home/cbarobotics/dev/playground/text_classify/tf/training_data.json'

DETERMINED_INTENTS={
    'weather':["Is it going to rain", "what is the weather like", "Should i carry an umbrella", "what is the temperature outside", "Is it sunny", "is it rainy",
        "should we carry a raincoat"],
    'clock':["what time is it","What's the time","what's time", "what is time", "What is the time now", "can you tell me the time", "What time is it", "what time of the day is it", "What day is today", "what day is it"
        "what day of the week is it", "What is todays date", "What month is it", "what year is it", "What is today", "What is the time of the day"],
    'self':['how are you', 'who are you', 'what is your name', 'where are you from', "are you plugged in", "what's your battery status", "What is your battery level"],
    'greet':['how are you', 'hello', 'hello chip', "what's up", 'good morning', 'good afternoon', 'good evening', 'good night', 'thank you chip', 'thank you', 'bye bye chip', 'see you later'],
    'actions':['can i get a  hug', 'i want a hug', 'i want to hug', 'let me hug you', 'can i hug you', 'shake hands chip', 'can i shake hands', 'shake hands chip', 'i want a shake hand',
        'let us take a photo', 'let us take a picture together', "let's take a selfie", 'can i take a selfie with you', 'can i take a selfie chip', 'i want a fist bump', 'fist bump chip',
        'fist bump', 'can i get a fist pump', 'i want to fist bump', 'lets do a fist bump', 'remember me chip', 'i want you to remember me next time', 'remember me', 'remember my face', 'i want to enrol my face',
        'enrol me chip', 'register my face', 'register me', 'i want a high five', 'high five chip', 'hi 5', "let's high five", "i want to high-five"]
}

INTENTS = {
    "direction":["how far is","how far is the", "where is", "where is the", "which way is", "which way is the", "which way to the", "which way to",
        "which direction is", "which direction is the", "help me with the direction to the", "can you help us with the direction to the",
        "which floor is the", "which floor is", "is there a {} here", "where can i find the", "where can i find a", "where can i find some"],
    "meet":["We are here to meet", "I am here to meet {} this time", "I am here to meet", "I would like to meet", "We would like to meet",
        "I have an appointment with", "We have an appointment with", "Can I meet", "Can we meet", "I have a meeting with",
        "We have a meeting with", "I am here to see", "We are here to see", "Can I see", "Can we see", "I would like to see", "I would like to see"],
    "navigate":["Can you help me to the", "Can you help us to", "Please guide me to the", "Please guide us to", "guide me to the", "Can you take me to the",
        "Can you take us to the", "Can you show us the way to", "Can you show me the way to the", "Can we go to the", "Let us go to", "Help me to the", "let's go to the",
        "go to", "go to the", "let's go to", "let us go", "Help me to the {} this time"]
  }

def main():
    global INTENTS, KBASE_PATH, DETERMINED_INTENTS, OUTPUT_PATH
    try:
        with open(KBASE_PATH, 'r') as f:
            kbase_data = json.load(f)
    except Exception as e:
        print "Exception while loading knowledge base - {}".format(e)

    training_data = {'training_data':[]}
    sentences = []
    for sentence in INTENTS['direction']:
        # direction to places and people
        for key in ['known_places', 'known_people']:
            for place_people in kbase_data[key]:
                for phrase in kbase_data[key][place_people]['phrases']:
                    complt_sentence = sentence.replace('{}', phrase) if '{}' in sentence else sentence + ' ' + phrase
                    sentences.append(complt_sentence)
    training_data['training_data'].append({'intent':'direction', 'sentences':sentences})
    sentences = []
    meet_train_data = {'intent':'meet', 'sentences':[]}
    for sentence in INTENTS['meet']:
        # direction to places and people
        for key in ['known_teams', 'known_people']:
            for team_people in kbase_data[key]:
                for phrase in kbase_data[key][team_people]['phrases']:
                    complt_sentence = sentence.replace('{}', phrase) if '{}' in sentence else sentence + ' ' + phrase
                    sentences.append(complt_sentence)
    training_data['training_data'].append({'intent':'meet', 'sentences':sentences})
    sentences = []
    navigate_train_data = {'intent':'navigate', 'sentences':[]}
    for sentence in INTENTS['navigate']:
        # direction to places and people
        for key in ['known_teams', 'known_people']:
            for team_people in kbase_data[key]:
                for phrase in kbase_data[key][team_people]['phrases']:
                    complt_sentence = sentence.replace('{}', phrase) if '{}' in sentence else sentence + ' ' + phrase
                    sentences.append(complt_sentence)
    training_data['training_data'].append({'intent':'navigate', 'sentences':sentences})

    for intent in DETERMINED_INTENTS:
        training_data['training_data'].append({'intent':intent, 'sentences':DETERMINED_INTENTS[intent]})

    try:
        with open(OUTPUT_PATH, 'a') as f:
            f.write(json.dumps(training_data))
    except Exception as e:
        print "Exception while writing data to {} - {}".format(OUTPUT_PATH, e)


if __name__== '__main__':
    main()

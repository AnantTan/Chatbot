# importing modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# importing modules
import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

# loading the json data
with open("intents.json") as file:
    data = json.load(file) # storing the data in a variable

# if the model is already trained
# if the trained file already exists do not train again
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)

# otherwise run it all again
# train the model again
except:
# declaring some blank lists
# these will be used later to talk to the bot
# these lists will store the patterns
    words = []
    labels = []
    docs_x = []
    docs_y = []

# this loop will go over the data and extract the data required
# for each pattern we turn it into a list of words
# add the patterns and its associated tags relatively
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)# patterns to words using the tokenizer
            words.extend(wrds)
            docs_x.append(wrds)# adding the pattern
            docs_y.append(intent["tag"])# adding docs_x reated tags
    
        if intent["tag"] not in labels:
            labels.append(intent["tag"])# if the tag is not labels add it
    
    # using stemmer to find a more general meaning of the voucablory 
    # this will be easier to scale down the model and make it more accurate
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))# sorting the stemmed words
    
    labels = sorted(labels)# labels will be upadted as the stemmed words
    
    # declaring the training and output list 
    training = []
    output = []
    
    # converting a string to number using 0 
    out_empty = [0 for _ in range(len(labels))]
    
    # using enumerate to obtain an indexed list
    for x, doc in enumerate(docs_x):
        bag = []# declaring bag for bag of words
        
        # stemming down the words in doc
        wrds = [stemmer.stem(w.lower()) for w in doc]
    
        # neural networks do not work with strings
        # they need numbers to work with
        # hence if the word is present in the list it will be marked 1
        # if not present it will be marked 0
        # this is because the words present are lost due to stemming
        # we only know the words present in our models vocab
        for w in words:
            if w in wrds:
                bag.append(1)# word present
            else:
                bag.append(0)# word not present
    
        # making a copy of the list 
        output_row = out_empty[:]
        
        # look in the labels list and check tag and set it to 1 in the output_row
        output_row[labels.index(docs_y[x])] = 1
    
        # add the bag of words to training
        training.append(bag)
        
        # add the output row to the list which has all the 1's and 0's
        output.append(output_row)
    
    # creating the array for training and output data
    training = numpy.array(training)
    output = numpy.array(output)
    
    # run it and save the model to the file
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)
    #tf.reset_default_graph()

# the model
# basically the neural network is going to look at our bag of words
# it is then going to give it a class it belongs to (tag)
net = tflearn.input_data(shape=[None, len(training[0])])# defining the input shape for our model
net = tflearn.fully_connected(net, 8)# have 8 neurons for the hidden layer for the input
net = tflearn.fully_connected(net, 8)# another hidden layer with 8 neurons 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")# allows us to get probabilities for each output
net = tflearn.regression(net)

model = tflearn.DNN(net)# train the model

# load the model
# if the model is there, fine
try:
    model.load("model.tflearn")

# if model is not found train it
except:
# Saving the model and printing its epoch
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]# creating blank bag of words list

    s_words = nltk.word_tokenize(s)
    
    # change the elements according to if the word exists or not
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    # generate the bag list properly
    for se in s_words:
        for i, w in enumerate(words):
            
            # if the word we are looking is there in the se
            # make it 1
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)# return the bag of words

# this method will be our customer agent bot
# this method is going to resposible for chatting
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":# if the user enters quit then exit
            break

        # pass in the inout with the bag of words list
        # this will give us back a probability
        results = model.predict([bag_of_words(inp, words)])
        
        # get the maximum probabilty result's index
        # that particular index should be the response
        results_index = numpy.argmax(results)
       
        # this will find the label from the labels list
        tag = labels[results_index]

        # from the intents find the tag 
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']# get responses using the tag

        # get a random response under the tag
        print(random.choice(responses))

# call the method to run the bot
chat()
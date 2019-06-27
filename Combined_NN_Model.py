### import the required modules and functions

from config import *
from reddit_parser import Parser
from ModelEstimation import NNModel
from NN_Utils import *

# NOTE: Don't forget to set NN=True in defaults.py before running this file

### Define the neural network object

nnmodel=NNModel(training_fraction=NN_training_fraction)

### check key hyperparameters for correct data types

nnmodel.NN_param_typecheck()

### Write hyperparameters to file. Performance measures will be written to the
# same file after analyses are performed

Write_Performance()

### call the parsing function

theparser=Parser()
theparser.Parse_Rel_RC_Comments()

### call the function for calculating the percentage of relevant comments

if calculate_perc_rel:
    theparser.Perc_Rel_RC_Comment()

### create training, development and test sets

# NOTE: Always index training set first.
# NOTE: For valid analysis results, maximum vocabulary size and frequency filter
# should not be changed between the creation of sets for LDA and NN analyses

## Determine the comments that will comprise various sets

nnmodel.Define_Sets()

## Read and index the content of comments in each set

for set_key in nnmodel.set_key_list:
    nnmodel.Index_Set(set_key)

## if performing sentiment pre-training, load comment sentiments from file
if special_doi == False and pretrained == False:
    nnmodel.Get_Sentiment(path)
elif special_doi == True:
    nnmodel.Get_Human_Ratings(path): #TODO: Define this function. It should already somehow incorporated into the ModelEstimator

### Sentiment Modeling/DOI Classification Neural Networks

## create the computation graph

nnmodel.Setup_Comp_Graph()

## create the session and initialize the variables

config = tf.ConfigProto(device_count = {'GPU': 0}) # Use only CPU (due to overly large matrices)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
if pretrained == False:
    state = sess.run(initialState)

### Train and Test the Neural Network ###

## create a list of comment lengths

for set_key in set_key_list:
    for i,x in enumerate(indexes[set_key]):
        lengths[set_key].append(len(indexes[set_key][i]))
    Max[set_key] = max(lengths[set_key]) # maximum length of a comment in this set
Max_l = max(Max['train'],Max['dev'],Max['test']) # max length of a comment in the whole dataset

## initialize vectors to store set accuracies or perplexities

if special_doi == True:
    accuracy = {key: [] for key in set_key_list}
    for set_key in set_key_list:
        accuracy[set_key] = np.empty(epochs)
else:
    perplexity = {key: [] for key in set_key_list}
    for set_key in set_key_list:
        perplexity[set_key] = np.empty(epochs)

### Train and test for the determined number of epochs

print("Number of epochs: "+str(epochs))
print("Number of epochs: "+str(epochs),file=perf)

for k in range(epochs): # for each epoch

    # timer
    print("Started epoch "+str(k+1)+" at "+time.strftime('%l:%M%p'))

    for set_key in set_key_list: # for each set

        if special_doi == True: # if classifying
            TotalCorr = 0 # reset number of correctly classified examples
        else: # if modeling language
            Epoch_Loss = 0 # reset the loss

        # initialize vectors for feeding data and desired output
        inputs = np.zeros([batchSz,Max_l])

        if special_doi == True:
            answers = np.zeros([batchSz,3],dtype=np.int32)
        else:
            answers = np.zeros([batchSz,Max_l])
            loss_weights = np.zeros([batchSz,Max_l])

        # batch counters
        j = 0 # batch comment counter
        p = 0 # batch counter

        for i in range(len(indexes[set_key])): # for each comment in the set
            inputs[j,:lengths[set_key][i]] = indexes[set_key][i]

            if special_doi == True:
                answers[j,:] = vote[set_key][i]
            else:
                answers[j,:lengths[set_key][i]-1] = indexes[set_key][i][1:]
                loss_weights[j,:lengths[set_key][i]] = 1

            j += 1 # update batch comment counter
            if j == batchSz - 1: # if the current batch is filled

                if special_doi == True: # if classifying
                    if set_key == 'train':
                        # train on the examples
                        _,outputs,next,_,Corr = sess.run([train,output,nextState,loss,numCorrect],feed_dict={inpt:inputs,answr:answers,DOutRate:keepP})
                    else:
                        # test on development or test set
                        _,Corr = sess.run([loss,numCorrect],feed_dict={inpt:inputs,answr:answers,DOutRate:1})
                else: # if doing language modeling
                    if set_key == 'train':
                        # train on the examples
                        _,outputs,next,Batch_Loss = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:keepP})
                    else:
                        # test on development or test set
                        Batch_Loss = sess.run(loss,feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:1})

                j = 0 # reset batch comment counter
                p += 1 # update batch counter

                # reset the input/label containers
                inputs = np.zeros([batchSz,Max_l])
                if special_doi == True:
                    answers = np.zeros([batchSz,3],dtype=np.int32)
                else:
                    answers = np.zeros([batchSz,Max_l])
                    loss_weights = np.zeros([batchSz,Max_l])

                # update the GRU state
                state = next # update the GRU state

                # update total number of correctly classified examples or total loss based on the processed batch
                if special_doi == True:
                    TotalCorr += Corr
                else:
                    Epoch_Loss += Batch_Loss

            # during language modeling training, every 10000 comments or at the end of training, save the weights
            if special_doi == False:
                if set_key == 'train' and ((i+1) % 10000 == 0 or i == len(indexes['train']) - 1):

                    # retrieve learned weights
                    embeddings,weights1,weights2,weights3,biases1,biases2,biases3 = sess.run([E,W1,W2,W3,b1,b2,b3])
                    embeddings = np.asarray(embeddings)
                    outputs = np.asarray(outputs)
                    weights1 = np.asarray(weights1)
                    weights2 = np.asarray(weights2)
                    weights3 = np.asarray(weights3)
                    biases1 = np.asarray(biases1)
                    biases2 = np.asarray(biases2)
                    biases3 = np.asarray(biases3)
                    # define a list of the retrieved variables
                    weights = ["embeddings","state","weights1","weights2","weights3","biases1","biases2","biases3"]
                    # write them to file
                    for variable in weights:
                        np.savetxt(output_path+"/"+variable, eval(variable))

        if special_doi == True: # calculate set accuracy for the current epoch and save the value
            accuracy[set_key][k] = float(TotalCorr) / float( p * batchSz )
            print("Accuracy on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(accuracy[set_key][k]))
            print("Accuracy on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(accuracy[set_key][k]),file=perf)

        else: # calculate set perplexity for the current epoch and save the value
            # calculate set perplexity
            perplexity[set_key][k] = np.exp(Epoch_Loss / p)
            print("Perplexity on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]))
            print("Perplexity on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]),file=perf)

    ## early stopping
    if early_stopping == True:
        if special_doi == True:
            # if development set accuracy is decreasing, stop training to prevent overfitting
            if k != 0 and accuracy['dev'][k] < accuracy['dev'][k-1]:
                break
        else:
            # if development set perplexity is increasing, stop training to prevent overfitting
            if k != 0 and perplexity['dev'][k] > perplexity['dev'][k-1]:
                break

# timer
print("Finishing time:" + time.strftime('%l:%M%p'))
# close the performance file
perf.close()

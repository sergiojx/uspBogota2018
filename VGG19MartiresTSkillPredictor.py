# Imports
import numpy as np
from numpy import genfromtxt
import pylab as pl
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.python.framework import ops
import urllib
import random
import sys
from IPython.display import clear_output
import re
import os, os.path, gc, time, shutil, matplotlib, glob

from trueskill import TrueSkill, Rating, quality_1vs1, rate_1vs1

TrueSkill(backend='scipy').cdf



TIE_VOTE_THRESHOLD = 0.25
TIE_REDISTRIBUTION = True
DITRIBUTION_ITERAT = 1


def actualOneMatchSycleVoteTrueSkillExplorer(randomIndexerSPath, imgSetSize, zoneName):
    IMG_SET_SIZE = imgSetSize
    # load synthetic vote list
    # randomChapVts = genfromtxt('../randomvotes/randomIndexerSChap.txt', delimiter=',')
    # randomChapVtsPredictions = genfromtxt('../predicts/randomIndexerSChap4predict.predict', delimiter=' ')
    
    randomChapVts = genfromtxt(randomIndexerSPath, delimiter=',')
    
    print(randomChapVts.shape)
            
    # initial default rating object building
    rankingDict = dict()
    for imgIdx in range(1, (IMG_SET_SIZE + 1)):
        rankingDict[imgIdx] = dict()
        rankingDict[imgIdx]["rating"] = Rating()
        rankingDict[imgIdx]["vCounter"] = 0
        rankingDict[imgIdx]["vIndex"] = 0
        rankingDict[imgIdx]["opponets"] = []
        for rowVote in range(randomChapVts.shape[0]):
            if randomChapVts[rowVote][0] == imgIdx:
                rankingDict[imgIdx]["opponets"].append({"opp":randomChapVts[rowVote][1], "out":randomChapVts[rowVote][2]})
        
        rankingDict[imgIdx]["vCounter"] = len(rankingDict[imgIdx]["opponets"])        
        

    print("%s published image set size: %d" % (zoneName,len(rankingDict)))
    
    zeroVotes = 0
    oneVotes = 0
    twoVotes = 0
    
    votCounter = randomChapVts.shape[0]
    while votCounter > 0:
        for imgIdx in range(1, (IMG_SET_SIZE + 1)):
            if rankingDict[imgIdx]["vIndex"] < rankingDict[imgIdx]["vCounter"]:
                opponentIdx  = rankingDict[imgIdx]["opponets"][rankingDict[imgIdx]["vIndex"]]["opp"]
                out = rankingDict[imgIdx]["opponets"][rankingDict[imgIdx]["vIndex"]]["out"]
                if out == 0:
                    zeroVotes = zeroVotes + 1
                    rankingDict[imgIdx]["rating"], rankingDict[opponentIdx]["rating"] = rate_1vs1(rankingDict[imgIdx]["rating"],
                                                                                                 rankingDict[opponentIdx]["rating"],
                                                                                                 drawn=True)
                elif out == 1:
                    oneVotes = oneVotes + 1
                    rankingDict[imgIdx]["rating"], rankingDict[opponentIdx]["rating"] = rate_1vs1(rankingDict[imgIdx]["rating"],
                                                                                                 rankingDict[opponentIdx]["rating"])
                elif out == 2:
                    twoVotes = twoVotes + 1
                    rankingDict[opponentIdx]["rating"], rankingDict[imgIdx]["rating"] = rate_1vs1(rankingDict[opponentIdx]["rating"],
                                                                                                  rankingDict[imgIdx]["rating"])
                else:
                    print("NO VALID VOTE CODE")
            
                rankingDict[imgIdx]["vIndex"] = rankingDict[imgIdx]["vIndex"] + 1
                votCounter = votCounter - 1
                
                
                
    print("Zero votes %d" % zeroVotes)
    print("One votes %d" % oneVotes)
    print("Two votes %d" % twoVotes)
    print("Total votes %d" % (zeroVotes+oneVotes+twoVotes))
    
    # obtain mu statistics
    mu = np.zeros((IMG_SET_SIZE, 1))
    for imgIdx in range(1, (IMG_SET_SIZE + 1)):
        mu[imgIdx-1] = rankingDict[imgIdx]["rating"].mu

    print("Min mu %f" % np.min(mu))
    print("Max mu %f" % np.max(mu))
    print("Mean mu %f" % np.mean(mu))
    print("Std mu %f" % np.std(mu))
    
    
    
    # obtain sigma statistics
    sigma = np.zeros((IMG_SET_SIZE, 1))
    for imgIdx in range(1, (IMG_SET_SIZE + 1)):
        sigma[imgIdx-1] = rankingDict[imgIdx]["rating"].sigma

    print("Min sigma %f" % np.min(sigma))
    print("Max sigma %f" % np.max(sigma))
    print("Mean sigma %f" % np.mean(sigma))
    print("Std sigma %f" % np.std(sigma))
        
    # generate output file
    out = np.zeros((IMG_SET_SIZE, 3))
    for imgIdx in range(1, (IMG_SET_SIZE + 1)):
        out[imgIdx-1][0] = imgIdx
        out[imgIdx-1][1] = rankingDict[imgIdx]["rating"].mu
        out[imgIdx-1][2] = rankingDict[imgIdx]["rating"].sigma
        
    
    fileName = "rdmIdxerS3%sOneMatchRating%s.csv" % (zoneName,'VGG19')
    print("Out file name %s" % fileName)
    np.savetxt(fileName, out, fmt='%i,%f,%f', delimiter=",")
    return mu, sigma, fileName, zeroVotes, oneVotes, twoVotes, (zeroVotes+oneVotes+twoVotes)

print("_______________________________Loading zone image list")
# Image set directory root
imgRoot = '/Users/SerG1oAC/Documents/dbintroUdacity/fullstack/vagrant/ggStreetView/map/localidades/martires/imgs/'
# Load published images
pblImgFile = open('mstrPublishedLst_Martires', 'r') 
pblImgSet =  pblImgFile.readlines() 
print('Published Image quantity: ' + str(len(pblImgSet)))
print(pblImgSet[0][:-2])
print(pblImgSet[len(pblImgSet) - 1][:-2])
print(pblImgSet[-1][:-2])


print("__________________________Load VGG19 model descriptors")
# Load image feature vectors extracted with the specified model
imageFeatures = genfromtxt('VGG19Martires_Ftrs.csv', delimiter=',')

print("______________________________Descriptor Normalization")
print("Image featured set shape: ", imageFeatures.shape)
imgFtrStd = np.expand_dims(np.std(imageFeatures, axis=0), axis=0) 
imgFtrMean = np.expand_dims(np.mean(imageFeatures, axis=0) , axis=0) 
imgNrmFeatures = ((imageFeatures - imgFtrMean)/(imgFtrStd+0.00000000001))

print("_____________________________________Load random votes")
# Load vote indexer file
voteSetFile = 'martirSRandomVote4Predict_SchemIII.txt'
print("Prediction taget file " + voteSetFile)
votesIndex = genfromtxt(voteSetFile, delimiter=',')
print("Vote set shape                   ", votesIndex.shape)

print("____________________________________Prediction session")
saver4Pre = tf.train.import_meta_graph('./transfer4uspVGG16NonEQUVerify_Jul_0518/model82.meta')
with tf.Session() as sess:
    # Step-2: Now let's load the weights saved using the restore method.
    saver4Pre.restore(sess, tf.train.latest_checkpoint('./transfer4uspVGG16NonEQUVerify_Jul_0518'))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()
    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    PREDICT_op = graph.get_tensor_by_name("PREDICT_op:0")
    SOFTMAX_op = graph.get_tensor_by_name('SOFTMAX_op:0')
    X = graph.get_tensor_by_name("X:0") 
    
    Y = graph.get_tensor_by_name("Y:0")
    DRP_En = graph.get_tensor_by_name("DRP_En:0")
    
    X_val = np.zeros((1, imageFeatures.shape[1]*2)) 
    Y_val = np.zeros((1,2))
    
    
    for i in range(0, votesIndex.shape[0]):
    # for i in range(0, votesIndex.shape[0]-ImgVtTestSize):
        # minus 1 because originall indexing list was done in MATLAB.
        # MATLAB index stars at 1
        imgAIdx = votesIndex[i][0] - 1
        imgBIdx = votesIndex[i][1] - 1
        voteCode = votesIndex[i][2]
        
        Y_val = np.zeros((1,2))
        Y_val[0][np.int(voteCode-1)] = 1
        X_val[0] = np.expand_dims(np.concatenate((imgNrmFeatures[np.int(imgAIdx)], imgNrmFeatures[np.int(imgBIdx)]), axis=0), axis=0)[0]
        
        result = sess.run(PREDICT_op, feed_dict={X: X_val, Y: Y_val, DRP_En: False})
        resultSftMx = sess.run(SOFTMAX_op, feed_dict={X: X_val, Y: Y_val, DRP_En: False})
        
        
        evalDiff = abs((resultSftMx[0][0]*1.0) - (resultSftMx[0][1]*1.0))
        if TIE_VOTE_THRESHOLD == 0 or evalDiff > TIE_VOTE_THRESHOLD:
            votesIndex[i][2] = np.int(result[0] + 1)
        else:
            votesIndex[i][2] = 0
        
        
        # clear_output(wait=True)
        print ("iteration %i: %f" % (i, 100*(i/votesIndex.shape[0])))
        sys.stdout.write("\033[F") 


print("________________________________Saving predictions")
np.savetxt("randomIndexerS_MartiresPredict4TSkill.csv", votesIndex, delimiter=",", fmt='%i')
    

print("________________________________Building TrueSkill Output File")
mu, sigma, fileName, zeroVotes, oneVotes, twoVotes, totalVotes = actualOneMatchSycleVoteTrueSkillExplorer(randomIndexerSPath = 'randomIndexerS_MartiresPredict4TSkill.csv',
                                                                                                          imgSetSize = 3788,
                                                                                                          zoneName = "Martirez")

print("Done!!!!")
      
        
                
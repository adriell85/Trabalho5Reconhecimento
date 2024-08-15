import numpy as np
from KNN import KNN
from DMC import DMC
from KmeansQuant import KMeansQuant

from openDatasets import openIrisDataset, openDermatologyDataset,openBreastDataset,openColumnDataset,openArtificialDataset,datasetSplitTrainTest
from plots import confusionMatrix, plotConfusionMatrix,plotDecisionSurface

def KNNRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/KNNRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações KNN {}.\n\n".format(convertDocName[base]))
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'KNN',convertDocName[base])
            ypredict = KNN(xtrain, ytrain, xtest, 5)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'KNN',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'KNN',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def DMCRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/DMCRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações DMC.\n\n")
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'DMC',convertDocName[base])
            ypredict = DMC(xtrain, ytrain, xtest)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'DMC',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'DMC',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


# def NayveBayesRuns(base):
#     from NaiveBayes import NaiveBayesClassifier
#     convertRun = {
#         0: openIrisDataset(),
#         1: openColumnDataset(),
#         2: openArtificialDataset(),
#         3: openBreastDataset(),
#         4: openDermatologyDataset()
#     }
#     convertDocName = {
#         0: 'Iris',
#         1: 'Coluna',
#         2: 'Artificial',
#         3: 'Breast',
#         4: 'Dermatology'
#
#     }
#
#     out = convertRun[base]
#     x = out[0]
#     y = out[1]
#     originalLabels = out[2]
#     accuracyList = []
#     fileName = "DadosRuns/NaiveRuns_{}.txt".format(convertDocName[base])
#     with open(fileName, 'w') as arquivo:
#         arquivo.write("Execução Iterações Naive {}.\n\n".format(convertDocName[base]))
#         for i in range(21):
#             print('\nIteração {}\n'.format(i))
#             xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'Naive Bayes Gaussian',convertDocName[base])
#             model = NaiveBayesClassifier()
#             model.fit(xtrain, ytrain,convertDocName[base],True,i)
#             ypredict = model.predict(xtest,convertDocName[base],i,False)
#             confMatrix = confusionMatrix(ytest, ypredict)
#             print('Confusion Matrix:\n', confMatrix)
#             plotConfusionMatrix(confMatrix,originalLabels,'Naive',i,convertDocName[base])
#             accuracy = np.trace(confMatrix) / np.sum(confMatrix)
#             print('ACC:', accuracy)
#             arquivo.write("ACC: {}\n".format(accuracy))
#             arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
#             accuracyList.append(accuracy)
#             plotDecisionSurface(xtrain, ytrain,'Naive',i,convertDocName[base])
#         print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
#         arquivo.write(
#             '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def BayesianGaussianDiscriminantRuns(base):
    from BayesianGaussianDiscriminant import GaussianDiscriminantAnalysis
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }
    classifierMode = [
        'lda',
        'qda'
    ]

    for mode in classifierMode:
        out = convertRun[base]
        x = out[0]
        y = out[1]
        originalLabels = out[2]
        accuracyList = []
        fileName = "DadosRuns/BayesianRuns_{}_{}.txt".format(convertDocName[base],mode)
        with open(fileName, 'w') as arquivo:
            arquivo.write("Execução Iterações Bayesian {} {}.\n\n".format(convertDocName[base],mode))
            for i in range(21):
                print('\nIteração {}\n'.format(i))
                xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80)
                model = GaussianDiscriminantAnalysis(mode)
                model.fit(xtrain, ytrain,convertDocName[base],True,i)
                ypredict = model.predict(xtest,ytest,convertDocName[base],i,False)
                confMatrix = confusionMatrix(ytest, ypredict)
                print('Confusion Matrix:\n', confMatrix)
                plotConfusionMatrix(confMatrix,originalLabels,'Bayesian_{}'.format(mode),i,convertDocName[base],)
                accuracy = np.trace(confMatrix) / np.sum(confMatrix)
                print('ACC:', accuracy)
                arquivo.write("ACC: {}\n".format(accuracy))
                arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
                accuracyList.append(accuracy)
                plotDecisionSurface(xtrain, ytrain,'Bayesian_{}'.format(mode),i,convertDocName[base])
            print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
            arquivo.write(
                '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))




def KmeansQuantRuns(base):
    # from BayesianGaussianDiscriminant import GaussianDiscriminantAnalysis
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }


    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []
    fileName = "DadosRuns/kMeansQuant_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações Kmeans {}.\n\n".format(convertDocName[base]))
        for i in range(21):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80)
            model = KMeansQuant()
            # for t in range(100):
            model.fit(X=xtrain,y=ytrain,baseName=convertDocName[base],isruningTrain=True,iteration=i)
            ypredict = model.predict(X=xtest,baseName= convertDocName[base],iteration=i, isRuningZ=False)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'kMeansQuant',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(accuracy)
            plotDecisionSurface(xtrain, ytrain,'kMeansQuant',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
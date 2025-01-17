import numpy as np
from plotGaussian import plotGaussianDistribution3d, dispersionDataBlindClass,dispersionDataByClass

class NaiveBayesClassifier:
    def fit(self, xtrain, ytrain,baseName,isruningTrain,iteration):
        xtrain = np.array(xtrain)
        nSamples,nFeatures = xtrain.shape
        self.classes = np.unique(ytrain)


        self.mean = []
        self.variance = []
        self.priorProb = []
        self.covariance = []



        for _class,c in enumerate(self.classes):
            _classSamples = xtrain[ytrain==c]
            if (isruningTrain):
                dispersionDataByClass(_classSamples, baseName, iteration,c)
            self.mean.append(np.mean(_classSamples,axis=0))
            self.variance.append(np.var(_classSamples,axis=0))
            self.priorProb.append(_classSamples.shape[0]/nSamples)
            self.covariance.append(np.cov(_classSamples, rowvar=False) + np.eye(nFeatures) * 1e-4)
        self.mean=np.array(self.mean)
        self.variance=np.array(self.variance)
        self.priorProb=np.array(self.priorProb)
        self.covariance = np.array(self.covariance)

        if(isruningTrain):
            dispersionDataBlindClass(xtrain, baseName,iteration, True)
            plotGaussianDistribution3d(baseName, iteration,self.means, self.covariance, self.classes, featureIndices=(1, 2))
            fileName = "DadosGaussiana/Dados_Plotagem_Gaussiana{}_base_{}_iteracao_{}.txt".format(baseName,baseName,iteration)
            with open(fileName, 'w') as arquivo:
                arquivo.write("Dados de Treino.\n\n{}\n".format(xtrain))

    def predict(self,xtest,baseName,iteration,isRuningZ):
        if(isRuningZ==False):
            dispersionDataBlindClass(xtest, baseName, iteration,False)
        fileName = "DadosGaussiana/Dados_Plotagem_Gaussiana{}_base_{}_iteracao_{}.txt".format(baseName,baseName,iteration)
        with open(fileName, 'a') as arquivo:
            arquivo.write("Dados de Teste.\n\n{}\n".format(xtest))
            arquivo.write('\nIteração: {} :::::::::::::::::\n'.format(iteration))
            predicts=[]
            count=0
            for xsample in xtest:
                posteriorsPros = []
                arquivo.write("\nAmostra {}::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n".format(count))
                count+=1
                for i,c in enumerate(self.classes):
                    priorprobability = np.log(self.priorProb[i])
                    conditionalClass = np.sum(np.log(self._pdf(i,xsample)))
                    posteriorProbability = priorprobability + conditionalClass
                    posteriorsPros.append(posteriorProbability)
                    arquivo.write("\nClasse {}\n".format(c))
                    arquivo.write("Probabilidade a priori: {:.4f}\n".format(np.exp(priorprobability)))
                    arquivo.write("Verossimilhança: {:.4f}\n".format(np.exp(conditionalClass)))
                    arquivo.write("Probabilidade a posteriori: {:.4f}\n".format(np.exp(posteriorProbability)))
                predicts.append(np.argmax(posteriorsPros))
            return np.array(predicts)

    def _pdf(self,iClass,sample):
        mean = self.mean[iClass]
        variance = self.variance[iClass]
        # Adicionando um pequeno valor à variância para evitar divisão por zero
        epsilon = 1e-7
        variance+= epsilon
        numerator = np.exp( - ( sample- mean ) ** 2 / (2 * variance) )
        denominator = np.sqrt( 2 * np.pi * variance )
        epsilon = 1e-10
        return np.maximum(numerator / denominator,epsilon)



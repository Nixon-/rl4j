package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/13/16.
 */

public class DQNFactoryStdDense implements DQNFactory {

    private Configuration conf;

    public DQNFactoryStdDense(final Configuration conf) {
        this.conf = conf;
    }

    public DQNFactoryStdDense setDefaultConfiguration(final Configuration config) {
        conf = config;
        return this;
    }

    public Configuration getConf() {
        return this.conf;
    }

    public DQN buildDQN(int shapeInputs[], int numOutputs) {
        return buildDQN(shapeInputs, numOutputs, this.getConf());
    }

    public static DQN buildDQN(int[] numInputs, int numOutputs, Configuration config) {

        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(config.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                //.regularization(true)
                //.l2(conf.getL2())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs[0])
                        .nOut(config.getNumHiddenNodes())
                        .activation("relu")
                        .build());


        for (int i = 1; i < config.getNumLayer(); i++) {
            confB
                    .layer(i, new DenseLayer.Builder()
                            .nIn(config.getNumHiddenNodes())
                            .nOut(config.getNumHiddenNodes())
                            .activation("relu")
                            .build());
        }

        confB
                .layer(config.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(config.getNumHiddenNodes())
                        .nOut(numOutputs)
                        .build());


        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        return new DQN(model);
    }

    public static class Configuration {
        private final int numLayer;
        private final int numHiddenNodes;
        private final double learningRate;
        private final double l2;

        public Configuration(final int numLayer, final int numHiddenNodes, final double learningRate, final double l2) {
            this.numLayer = numLayer;
            this.numHiddenNodes = numHiddenNodes;
            this.learningRate = learningRate;
            this.l2 = l2;
        }

        int getNumLayer() {
            return numLayer;
        }

        int getNumHiddenNodes() {
            return numHiddenNodes;
        }

        double getLearningRate() {
            return learningRate;
        }

        public double getL2() {
            return l2;
        }
    }
}

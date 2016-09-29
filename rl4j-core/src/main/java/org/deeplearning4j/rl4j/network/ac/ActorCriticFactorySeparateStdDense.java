package org.deeplearning4j.rl4j.network.ac;

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
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 *
 */
public class ActorCriticFactorySeparateStdDense implements ActorCriticFactorySeparate {

    private Configuration conf;

    public ActorCriticFactorySeparateStdDense(final Configuration conf) {
        this.conf = conf;
    }

    public ActorCriticFactorySeparateStdDense setDefaultConfiguration(final Configuration conf) {
        this.conf = conf;
        return this;
    }

    public Configuration getConf() {
        return this.conf;
    }

    public ActorCriticSeparate buildActorCritic(int[] numInputs, int numOutputs) {
        return buildActorCritic(numInputs, numOutputs, this.conf);
    }

    public static ActorCriticSeparate buildActorCritic(int[] numInputs, int numOutputs, Configuration config) {
        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(config.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rho(config.getRmsDecay())//.rmsDecay(config.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                //.regularization(true)
                //.l2(config.getL2())
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
                        .nOut(1)
                        .build());


        MultiLayerConfiguration mlnconf2 = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf2);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        NeuralNetConfiguration.ListBuilder confB2 = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(config.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rho(config.getRmsDecay())//.rmsDecay(config.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                //.regularization(true)
                //.l2(config.getL2())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs[0])
                        .nOut(config.getNumHiddenNodes())
                        .activation("relu")
                        .build());


        for (int i = 1; i < config.getNumLayer(); i++) {
            confB2
                    .layer(i, new DenseLayer.Builder()
                            .nIn(config.getNumHiddenNodes())
                            .nOut(config.getNumHiddenNodes())
                            .activation("relu")
                            .build());
        }

        confB2
                .layer(config.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(config.getNumHiddenNodes())
                        .nOut(numOutputs)
                        .build());


        MultiLayerConfiguration mlnconf = confB2.pretrain(false).backprop(true).build();
        MultiLayerNetwork model2 = new MultiLayerNetwork(mlnconf);
        model2.init();
        model2.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        return new ActorCriticSeparate(model, model2);
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

        public int getNumLayer() {
            return numLayer;
        }

        public int getNumHiddenNodes() {
            return numHiddenNodes;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public double getL2() {
            return l2;
        }
    }
}

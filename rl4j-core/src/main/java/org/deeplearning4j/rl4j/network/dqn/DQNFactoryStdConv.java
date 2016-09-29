package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/13/16.
 */
public class DQNFactoryStdConv implements DQNFactory {

    private Configuration conf;

    public DQNFactoryStdConv(final Configuration conf) {
        this.conf = conf;
    }

    public DQNFactoryStdConv setDefaultConfiguration(final Configuration config) {
        conf = config;
        return this;
    }

    public Configuration getConf() {
        return this.conf;
    }

    public DQN buildDQN(int shapeInputs[], int numOutputs) {
        return buildDQN(shapeInputs, numOutputs, this.getConf());
    }

    public static DQN buildDQN(int shapeInputs[], int numOutputs, Configuration config) {

        if (shapeInputs.length == 1)
            throw new AssertionError("Impossible to apply convolutional layer on a shape == 1");

        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(config.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rmsDecay(config.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .l2(config.getL2())
                .list()
                .layer(0, new ConvolutionLayer.Builder(8, 8)
                        .nIn(shapeInputs[0])
                        .nOut(16)
                        .stride(4, 4)
                        .activation("relu")
                        .build());

        confB
                .layer(1, new ConvolutionLayer.Builder(4, 4)
                        .nOut(32)
                        .stride(2, 2)
                        .activation("relu")
                        .build());

        confB
                .layer(2, new DenseLayer.Builder().nOut(256)
                        .activation("relu")
                        .build());

        confB
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nOut(numOutputs).build());

        new ConvolutionLayerSetup(confB, shapeInputs[1], shapeInputs[2], shapeInputs[0]);
        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        return new DQN(model);
    }

    public static class Configuration {
        final private double learningRate;
        final private double l2;

        public Configuration(final double learningRate, final double l2) {
            this.learningRate = learningRate;
            this.l2 = l2;
        }

        double getLearningRate() {
            return learningRate;
        }

        double getL2() {
            return l2;
        }
    }

}

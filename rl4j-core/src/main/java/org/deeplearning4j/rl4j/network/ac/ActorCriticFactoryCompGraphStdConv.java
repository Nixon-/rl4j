package org.deeplearning4j.rl4j.network.ac;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 * Standard factory for Conv net Actor Critic
 */
public class ActorCriticFactoryCompGraphStdConv implements ActorCriticFactoryCompGraph {

    private Configuration conf;

    public ActorCriticFactoryCompGraphStdConv(final Configuration conf) {
        this.conf = conf;
    }

    public ActorCriticFactoryCompGraphStdConv setDefaultConfiguration(final Configuration config) {
        conf = config;
        return this;
    }

    public Configuration getConf() {
        return this.conf;
    }

    public ActorCriticCompGraph buildActorCritic(int shapeInputs[], int numOutputs) {
        return buildActorCritic(shapeInputs, numOutputs, this.getConf());
    }

    public static ActorCriticCompGraph buildActorCritic(int shapeInputs[], int numOutputs, final Configuration config) {

        if (shapeInputs.length == 1)
            throw new AssertionError("Impossible to apply convolutional layer on a shape == 1");

        ComputationGraphConfiguration.GraphBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(config.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                //.updater(Updater.RMSPROP).rmsDecay(config.getRmsDecay())
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .l2(config.getL2())
                .graphBuilder()
                .setInputTypes(InputType.convolutional(shapeInputs[1], shapeInputs[2], shapeInputs[0]))
                .addInputs("input")
                .addLayer("0", new ConvolutionLayer.Builder(8, 8)
                        .nIn(shapeInputs[0])
                        .nOut(16)
                        .stride(4, 4)
                        .activation("relu")
                        .build(), "input");

        confB
                .addLayer("1", new ConvolutionLayer.Builder(4, 4)
                        .nOut(32)
                        .stride(2, 2)
                        .activation("relu")
                        .build(), "0");

        confB
                .addLayer("2", new DenseLayer.Builder().nOut(256)
                        .activation("relu")
                        .build(), "1");

        confB
                .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nOut(1).build(), "2");

        confB
                .addLayer("softmax", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax") //fixthat
                        .nOut(numOutputs).build(), "2");

        confB.setOutputs("value", "softmax");


        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();
        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        return new ActorCriticCompGraph(model);
    }

    public static class Configuration {
        private final double learningRate;
        private final double l2;
        private final double rmsDecay;

        public Configuration(double learningRate, double l2, double rmsDecay) {
            this.learningRate = learningRate;
            this.l2 = l2;
            this.rmsDecay = rmsDecay;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public double getL2() {
            return l2;
        }

        public double getRmsDecay() {
            return rmsDecay;
        }
    }

}

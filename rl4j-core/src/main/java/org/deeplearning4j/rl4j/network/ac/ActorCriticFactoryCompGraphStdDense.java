package org.deeplearning4j.rl4j.network.ac;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
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
 *
 */
public class ActorCriticFactoryCompGraphStdDense implements ActorCriticFactoryCompGraph {

    private Configuration conf;

    ActorCriticFactoryCompGraphStdDense(Configuration conf) {
        this.conf = conf;
    }

    public ActorCriticFactoryCompGraphStdDense setDefaultConfiguration(final Configuration config) {
        conf = config;
        return this;
    }

    public Configuration getConf() {
        return this.conf;
    }

    public ActorCriticCompGraph buildActorCritic(int[] numInputs, int numOutputs) {
        return buildActorCritic(numInputs, numOutputs, this.getConf());
    }

    public static ActorCriticCompGraph buildActorCritic(int[] numInputs, int numOutputs, final Configuration config) {

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
                .setInputTypes(InputType.feedForward(numInputs[0]))
                .addInputs("input")
                .addLayer("0", new DenseLayer.Builder()
                        .nIn(numInputs[0])
                        .nOut(config.getNumHiddenNodes())
                        .activation("relu")
                        .build(), "input");


        for (int i = 1; i < config.getNumLayer(); i++) {
            confB
                    .addLayer(i + "", new DenseLayer.Builder()
                            .nIn(config.getNumHiddenNodes())
                            .nOut(config.getNumHiddenNodes())
                            .activation("relu")
                            .build(), (i - 1) + "");
        }


        confB
                .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nOut(1).build(), (config.getNumLayer() - 1) + "");

        confB
                .addLayer("softmax", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax") //fixthat
                        .nOut(numOutputs).build(), (config.getNumLayer() - 1) + "");

        confB.setOutputs("value", "softmax");


        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();
        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        return new ActorCriticCompGraph(model);
    }

    public static class Configuration {
        private final int numLayer;
        private final int numHiddenNodes;
        private final double learningRate;
        private final double l2;

        public Configuration(int numLayer, int numHiddenNodes, double learningRate, double l2) {
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

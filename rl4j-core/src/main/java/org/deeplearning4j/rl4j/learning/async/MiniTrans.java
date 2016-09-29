package org.deeplearning4j.rl4j.learning.async;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Its called a MiniTrans because it is similar to a Transition
 * but without a next observation
 *
 * It is stacked and then processed by AsyncNStepQL or A3C
 * following the paper implementation https://arxiv.org/abs/1602.01783 paper.
 *
 */
public class MiniTrans<A> {
    private final INDArray obs;
    private final A action;
    private final INDArray[] output;
    private final double reward;

    MiniTrans(final INDArray obs, final A action, final INDArray[] output, final double reward) {
        this.obs = obs;
        this.action = action;
        this.output = output;
        this.reward = reward;
    }

    public INDArray getObs() {
        return obs;
    }

    public A getAction() {
        return action;
    }

    public INDArray[] getOutput() {
        return output;
    }

    public double getReward() {
        return reward;
    }
}

package org.deeplearning4j.rl4j.learning;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Value;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;

import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/27/16.
 *
 * Useful factorisations and helper methods for class inheriting
 * ILearning.
 *
 * Big majority of training method should inherit this
 *
 */
public abstract class Learning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
        implements ILearning<O, A, AS>, NeuralNetFetchable<NN>{

    final private Logger logger;
    private int stepCounter = 0;
    private int epochCounter = 0;

    private IHistoryProcessor historyProcessor = null;

    public Learning(LConfiguration conf) {
        logger = LoggerFactory.getLogger(this.getClass());
    }

    public static Integer getMaxAction(INDArray vector) {
        return Nd4j.argMax(vector, Integer.MAX_VALUE).getInt(0);
    }

    public static <O extends Encodable, A, AS extends ActionSpace<A>> INDArray getInput(MDP<O, A, AS> mdp, O obs) {
        INDArray arr = Nd4j.create(obs.toArray());
        int[] shape = mdp.getObservationSpace().getShape();
        if (shape.length == 1)
            return arr;
        else
            return arr.reshape(shape);
    }

    public static <O extends Encodable, A, AS extends ActionSpace<A>> InitMdp<O>
    initMdp(MDP<O, A, AS> mdp, IHistoryProcessor hp) {

        O obs = mdp.reset();

        O nextO = obs;

        int step = 0;
        double reward = 0;

        boolean isHistoryProcessor = hp != null;

        int skipFrame = isHistoryProcessor ? hp.getConf().getSkipFrame() : 1;
        int requiredFrame = isHistoryProcessor ? skipFrame * (hp.getConf().getHistoryLength() - 1) : 0;

        while (step < requiredFrame) {
            INDArray input = Learning.getInput(mdp, obs);

            // History processor can't be null here, or else requireFrame would be 0, and this
            // loop would never execute.
            hp.record(input);

            A action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP
            if (step % skipFrame == 0) {
                hp.add(input);
            }

            StepReply<O> stepReply = mdp.step(action);
            reward += stepReply.getReward();
            nextO = stepReply.getObservation();

            step++;
        }
        return new InitMdp<>(step, nextO, reward);
    }

    public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        System.arraycopy(shape, 0, nshape, 1, shape.length);
        return nshape;
    }

    @Override
    public int getStepCounter() {
        return stepCounter;
    }

    protected int getEpochCounter() {
        return epochCounter;
    }

    protected abstract DataManager getDataManager();

    public abstract NN getNeuralNet();

    protected int incrementStep() {
        return stepCounter++;
    }

    protected int incrementEpoch() {
        return epochCounter++;
    }

    protected void setHistoryProcessor(HistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    public Logger getLogger() {
        return logger;
    }

    protected INDArray getInput(O obs) {
        return getInput(getMdp(), obs);
    }

    protected InitMdp<O> initMdp() {
        return initMdp(getMdp(), getHistoryProcessor());
    }

    public IHistoryProcessor getHistoryProcessor(){
        return this.historyProcessor;
    }

    public static class InitMdp<E> {
        private int steps;
        private E lastObs;
        private double reward;

        InitMdp(final int steps, final E lastObs, final double reward) {
            this.steps = steps;
            this.lastObs = lastObs;
            this.reward = reward;
        }

        /**
         * @return  steps, whatever that is.
         */
        public int getSteps() {
            return this.steps;
        }

        /**
         * @return  Last Observation.
         */
        public E getLastObs() {
            return this.lastObs;
        }

        /**
         * @return  Reward.
         */
        public double getReward() {
            return this.reward;
        }

    }

}

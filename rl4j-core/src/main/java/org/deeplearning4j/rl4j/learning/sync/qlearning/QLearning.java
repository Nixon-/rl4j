package org.deeplearning4j.rl4j.learning.sync.qlearning;


import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.SyncLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.EpsGreedy;

import org.deeplearning4j.rl4j.util.DataManager.StatEntry;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 *
 * Mother class for QLearning in the Discrete domain and
 * hopefully one day for the  Continuous domain.
 */
public abstract class QLearning<O extends Encodable, A, AS extends ActionSpace<A>>
        extends SyncLearning<O, A, AS, IDQN> {

    final private IExpReplay<A> expReplay;

    public QLearning(QLConfiguration conf) {
        super(conf);
        expReplay = new ExpReplay<>(conf.getExpRepMaxSize(), conf.getBatchSize());
    }

    protected abstract EpsGreedy<O, A, AS> getEgPolicy();

    public abstract MDP<O, A, AS> getMdp();

    protected abstract IDQN getCurrentDQN();

    public abstract IDQN getTargetDQN();

    public abstract void setTargetDQN(IDQN dqn);

    protected IExpReplay<A> getExpReplay() {
        return this.expReplay;
    }

    protected INDArray dqnOutput(INDArray input) {
        return getCurrentDQN().output(input);
    }

    protected INDArray targetDqnOutput(INDArray input) {
        return getTargetDQN().output(input);
    }

    private void updateTargetNetwork() {
        getLogger().info("Update target network");
        setTargetDQN(getCurrentDQN().clone());
    }


    public IDQN getNeuralNet() {
        return getCurrentDQN();
    }

    public abstract QLConfiguration getConfiguration();

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract QLStepReturn<O> trainStep(O obs);

    protected StatEntry trainEpoch() {
        InitMdp<O> initMdp = initMdp();
        O obs = initMdp.getLastObs();

        double reward = initMdp.getReward();
        int step = initMdp.getSteps();

        Double startQ = Double.NaN;
        double meanQ = 0;
        int numQ = 0;
        List<Double> scores = new ArrayList<>();
        while (step < getConfiguration().getMaxEpochStep() && !getMdp().isDone()) {

            if (getStepCounter() % getConfiguration().getTargetDqnUpdateFreq() == 0) {
                updateTargetNetwork();
            }

            QLStepReturn<O> stepR = trainStep(obs);

            if (!stepR.getMaxQ().isNaN()) {
                if (startQ.isNaN())
                    startQ = stepR.getMaxQ();
                numQ++;
                meanQ += stepR.getMaxQ();
            }

            if (stepR.getScore() != 0)
                scores.add(stepR.getScore());

            reward += stepR.getStepReply().getReward();
            obs = stepR.getStepReply().getObservation();
            incrementStep();
            step++;
        }

        meanQ /= (numQ + 0.001); //avoid div zero


        return new QLStatEntry(getStepCounter(),
                getEpochCounter(), reward, step, scores, getEgPolicy().getEpsilon(), startQ, meanQ);
    }

    private class QLStatEntry implements StatEntry {
        private int stepCounter;
        private int epochCounter;
        private double reward;
        private int episodeLength;
        private List<Double> scores;
        private double epsilon;
        private double startQ;
        private double meanQ;

        QLStatEntry(final int stepCounter, final int epochCounter, final double reward,
                    final int episodeLength, final List<Double> scores, final double epsilon,
                    final double startQ, final double meanQ) {
            this.stepCounter = stepCounter;
            this.epochCounter = epochCounter;
            this.reward = reward;
            this.episodeLength = episodeLength;
            this.scores = scores;
            this.epsilon = epsilon;
            this.startQ = startQ;
            this.meanQ = meanQ;
        }

        @Override
        public int getStepCounter() {
            return stepCounter;
        }

        @Override
        public int getEpochCounter() {
            return epochCounter;
        }

        @Override
        public double getReward() {
            return reward;
        }

        public int getEpisodeLength() {
            return episodeLength;
        }

        public List<Double> getScores() {
            return scores;
        }

        public double getEpsilon() {
            return epsilon;
        }

        public double getStartQ() {
            return startQ;
        }

        public double getMeanQ() {
            return meanQ;
        }
    }

    public static class QLStepReturn<O> {
        private Double maxQ;
        private double score;
        private StepReply<O> stepReply;

        public QLStepReturn(final Double maxQ, final double score, final StepReply<O> stepReply) {
            this.maxQ = maxQ;
            this.score = score;
            this.stepReply = stepReply;
        }

        Double getMaxQ() {
            return this.maxQ;
        }

        double getScore () {
            return this.score;
        }

        StepReply<O> getStepReply() {
            return this.stepReply;
        }

    }

    public static class QLConfiguration implements LConfiguration {
        private int seed;
        private int maxEpochStep;
        private int maxStep;
        private int expRepMaxSize;
        private int batchSize;
        private int targetDqnUpdateFreq;
        private int updateStart;
        private double rewardFactor;
        private double gamma;
        private double errorClamp;
        private double minEpsilon;
        private int epsilonNbStep;
        private boolean doubleDQN;

        public QLConfiguration(final int seed, final int maxEpochStep, final int maxStep, final int expRepMaxSize,
                               final int batchSize, final int targetDqnUpdateFreq, final int updateStart,
                               final double rewardFactor, final double gamma, final double errorClamp,
                               final double minEpsilon, final int epsilonNbStep, final boolean doubleDQN) {
            this.seed = seed;
            this.maxEpochStep = maxEpochStep;
            this.maxStep = maxStep;
            this.expRepMaxSize = expRepMaxSize;
            this.batchSize = batchSize;
            this.targetDqnUpdateFreq = targetDqnUpdateFreq;
            this.updateStart = updateStart;
            this.rewardFactor = rewardFactor;
            this.gamma = gamma;
            this.errorClamp = errorClamp;
            this.minEpsilon = minEpsilon;
            this.epsilonNbStep = epsilonNbStep;
            this.doubleDQN = doubleDQN;
        }

        public boolean isDoubleDQN() {
            return this.doubleDQN;
        }

        public int getEpsilonNbStep() {
            return this.epsilonNbStep;
        }

        public double getMinEpsilon() {
            return this.minEpsilon;
        }

        public double getErrorClamp() {
            return this.errorClamp;
        }

        public double getRewardFactor() {
            return this.rewardFactor;
        }

        public int getExpRepMaxSize() {
            return expRepMaxSize;
        }

        public int getBatchSize() {
            return this.batchSize;
        }

        public int getUpdateStart() {
            return this.updateStart;
        }

        public int getTargetDqnUpdateFreq() {
            return targetDqnUpdateFreq;
        }

        @Override
        public int getSeed() {
            return this.seed;
        }

        @Override
        public int getMaxEpochStep() {
            return this.maxEpochStep;
        }

        @Override
        public int getMaxStep() {
            return this.maxStep;
        }

        @Override
        public double getGamma() {
            return this.gamma;
        }

        @Override
        public boolean equals(Object o) {
            //TODO Make sure this equals function is actually correct.
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QLConfiguration that = (QLConfiguration) o;

            if (seed != that.seed) return false;
            if (maxEpochStep != that.maxEpochStep) return false;
            if (maxStep != that.maxStep) return false;
            if (expRepMaxSize != that.expRepMaxSize) return false;
            if (batchSize != that.batchSize) return false;
            if (targetDqnUpdateFreq != that.targetDqnUpdateFreq) return false;
            if (updateStart != that.updateStart) return false;
            if (Double.compare(that.rewardFactor, rewardFactor) != 0) return false;
            if (Double.compare(that.gamma, gamma) != 0) return false;
            if (Double.compare(that.errorClamp, errorClamp) != 0) return false;
            if (Double.compare(that.minEpsilon, minEpsilon) != 0) return false;
            if (epsilonNbStep != that.epsilonNbStep) return false;
            return doubleDQN == that.doubleDQN;

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = seed;
            result = 31 * result + maxEpochStep;
            result = 31 * result + maxStep;
            result = 31 * result + expRepMaxSize;
            result = 31 * result + batchSize;
            result = 31 * result + targetDqnUpdateFreq;
            result = 31 * result + updateStart;
            temp = Double.doubleToLongBits(rewardFactor);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(gamma);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(errorClamp);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(minEpsilon);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + epsilonNbStep;
            result = 31 * result + (doubleDQN ? 1 : 0);
            return result;
        }
    }


}

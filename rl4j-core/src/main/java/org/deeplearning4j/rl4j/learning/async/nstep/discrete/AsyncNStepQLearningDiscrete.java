package org.deeplearning4j.rl4j.learning.async.nstep.discrete;


import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.Policy;

import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public abstract class AsyncNStepQLearningDiscrete<O extends Encodable>
        extends AsyncLearning<O, Integer, DiscreteSpace, IDQN> {

    final private AsyncNStepQLConfiguration configuration;
    final private MDP<O, Integer, DiscreteSpace> mdp;
    final private DataManager dataManager;
    final private AsyncGlobal<IDQN> asyncGlobal;

    AsyncNStepQLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, AsyncNStepQLConfiguration conf,
                                DataManager dataManager) {
        super(conf);
        this.mdp = mdp;
        this.dataManager = dataManager;
        this.configuration = conf;
        this.asyncGlobal = new AsyncGlobal<>(dqn, conf);
    }


    public AsyncThread newThread(int i) {
        return new AsyncNStepQLearningThreadDiscrete<>(mdp.newInstance(), asyncGlobal, configuration, i, dataManager);
    }

    public IDQN getNeuralNet() {
        return asyncGlobal.cloneCurrent();
    }

    public Policy<O, Integer> getPolicy() {
        return new DQNPolicy<O>(getNeuralNet());
    }

    @Override
    public AsyncNStepQLConfiguration getConfiguration() {
        return configuration;
    }

    @Override
    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return mdp;
    }

    @Override
    public DataManager getDataManager() {
        return dataManager;
    }

    public AsyncGlobal<IDQN> getAsyncGlobal() {
        return asyncGlobal;
    }

    public static class AsyncNStepQLConfiguration implements AsyncConfiguration {
        private final int seed;
        private final int maxEpochStep;
        private final int maxStep;
        private final int numThread;
        private final int nstep;
        private final int targetDqnUpdateFreq;
        private final int updateStart;
        private final double rewardFactor;
        private final double gamma;
        private final double errorClamp;
        private final float minEpsilon;
        private final int epsilonNbStep;

        public AsyncNStepQLConfiguration(int seed, int maxEpochStep, int maxStep, int numThread, int nstep,
                                         int targetDqnUpdateFreq, int updateStart, double rewardFactor,
                                         double gamma, double errorClamp, float minEpsilon, int epsilonNbStep) {
            this.seed = seed;
            this.maxEpochStep = maxEpochStep;
            this.maxStep = maxStep;
            this.numThread = numThread;
            this.nstep = nstep;
            this.targetDqnUpdateFreq = targetDqnUpdateFreq;
            this.updateStart = updateStart;
            this.rewardFactor = rewardFactor;
            this.gamma = gamma;
            this.errorClamp = errorClamp;
            this.minEpsilon = minEpsilon;
            this.epsilonNbStep = epsilonNbStep;
        }

        @Override
        public int getSeed() {
            return seed;
        }

        @Override
        public int getMaxEpochStep() {
            return maxEpochStep;
        }

        @Override
        public int getMaxStep() {
            return maxStep;
        }

        @Override
        public int getNumThread() {
            return numThread;
        }

        @Override
        public int getNstep() {
            return nstep;
        }

        @Override
        public int getTargetDqnUpdateFreq() {
            return targetDqnUpdateFreq;
        }

        @Override
        public int getUpdateStart() {
            return updateStart;
        }

        @Override
        public double getRewardFactor() {
            return rewardFactor;
        }

        @Override
        public double getGamma() {
            return gamma;
        }

        @Override
        public double getErrorClamp() {
            return errorClamp;
        }

        public float getMinEpsilon() {
            return minEpsilon;
        }

        public int getEpsilonNbStep() {
            return epsilonNbStep;
        }

        @Override
        public boolean equals(Object o) {
            // TODO make sure this is correct.
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            AsyncNStepQLConfiguration that = (AsyncNStepQLConfiguration) o;

            if (seed != that.seed) return false;
            if (maxEpochStep != that.maxEpochStep) return false;
            if (maxStep != that.maxStep) return false;
            if (numThread != that.numThread) return false;
            if (nstep != that.nstep) return false;
            if (targetDqnUpdateFreq != that.targetDqnUpdateFreq) return false;
            if (updateStart != that.updateStart) return false;
            if (Double.compare(that.rewardFactor, rewardFactor) != 0) return false;
            if (Double.compare(that.gamma, gamma) != 0) return false;
            if (Double.compare(that.errorClamp, errorClamp) != 0) return false;
            if (Float.compare(that.minEpsilon, minEpsilon) != 0) return false;
            return epsilonNbStep == that.epsilonNbStep;

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = seed;
            result = 31 * result + maxEpochStep;
            result = 31 * result + maxStep;
            result = 31 * result + numThread;
            result = 31 * result + nstep;
            result = 31 * result + targetDqnUpdateFreq;
            result = 31 * result + updateStart;
            temp = Double.doubleToLongBits(rewardFactor);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(gamma);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(errorClamp);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + (minEpsilon != +0.0f ? Float.floatToIntBits(minEpsilon) : 0);
            result = 31 * result + epsilonNbStep;
            return result;
        }
    }
}

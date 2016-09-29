package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 * Training for A3C in the Discrete Domain
 *
 * At the time of leaving my intenrship, this does not seem to work correctly
 * although all methods are fully implemented as described in the
 * https://arxiv.org/abs/1602.01783 paper.
 *
 */
public abstract class A3CDiscrete<O extends Encodable> extends AsyncLearning<O, Integer, DiscreteSpace, IActorCritic> {

    final public A3CConfiguration configuration;
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    final private IActorCritic iActorCritic;
    final private AsyncGlobal<IActorCritic> asyncGlobal;
    final private Policy<O, Integer> policy;
    final private DataManager dataManager;

    A3CDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic iActorCritic,
                A3CConfiguration conf, DataManager dataManager) {
        super(conf);
        this.iActorCritic = iActorCritic;
        this.mdp = mdp;
        this.configuration = conf;
        this.dataManager = dataManager;
        policy = new ACPolicy<>(iActorCritic);
        asyncGlobal = new AsyncGlobal<IActorCritic>(iActorCritic, conf);
    }

    @Override
    public A3CConfiguration getConfiguration() {
        return configuration;
    }

    @Override
    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return mdp;
    }

    @Override
    public AsyncGlobal<IActorCritic> getAsyncGlobal() {
        return asyncGlobal;
    }

    @Override
    public Policy<O, Integer> getPolicy() {
        return policy;
    }

    @Override
    public DataManager getDataManager() {
        return dataManager;
    }

    protected AsyncThread<O, Integer, DiscreteSpace, ?>  newThread(int i) {
        return new A3CThreadDiscrete<>(mdp.newInstance(), asyncGlobal, getConfiguration(), i, dataManager);
    }

    public IActorCritic getNeuralNet() {
        return iActorCritic;
    }

    public static class A3CConfiguration implements AsyncConfiguration {

        private final int seed;
        private final int maxEpochStep;
        private final int maxStep;
        private final int numThread;
        private final int nstep;
        private final int updateStart;
        private final double rewardFactor;
        private final double gamma;
        private final double errorClamp;

        public A3CConfiguration(int seed, int maxEpochStep, int maxStep, int numThread, int nstep, int updateStart,
                                double rewardFactor, double gamma, double errorClamp) {
            this.seed = seed;
            this.maxEpochStep = maxEpochStep;
            this.maxStep = maxStep;
            this.numThread = numThread;
            this.nstep = nstep;
            this.updateStart = updateStart;
            this.rewardFactor = rewardFactor;
            this.gamma = gamma;
            this.errorClamp = errorClamp;
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

        public int getTargetDqnUpdateFreq(){
            return -1;
        }

    }
}

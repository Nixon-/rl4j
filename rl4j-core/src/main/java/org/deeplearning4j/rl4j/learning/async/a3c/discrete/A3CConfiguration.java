package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;

public class A3CConfiguration implements AsyncConfiguration {

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

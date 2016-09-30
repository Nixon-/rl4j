package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.rl4j.learning.ILearning;

public class QLConfiguration implements ILearning.LConfiguration {
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

        if (seed != that.getSeed()) return false;
        if (maxEpochStep != that.getMaxEpochStep()) return false;
        if (maxStep != that.getMaxStep()) return false;
        if (expRepMaxSize != that.getExpRepMaxSize()) return false;
        if (batchSize != that.getBatchSize()) return false;
        if (targetDqnUpdateFreq != that.getTargetDqnUpdateFreq()) return false;
        if (updateStart != that.getUpdateStart()) return false;
        if (Double.compare(that.getRewardFactor(), rewardFactor) != 0) return false;
        if (Double.compare(that.getGamma(), gamma) != 0) return false;
        if (Double.compare(that.getErrorClamp(), errorClamp) != 0) return false;
        if (Double.compare(that.getMinEpsilon(), minEpsilon) != 0) return false;
        if (epsilonNbStep != that.getEpsilonNbStep()) return false;
        return doubleDQN == that.isDoubleDQN();

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
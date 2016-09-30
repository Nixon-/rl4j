package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;

public class AsyncNStepQLConfiguration implements AsyncConfiguration {
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

        if (seed != that.getSeed()) return false;
        if (maxEpochStep != that.getMaxEpochStep()) return false;
        if (maxStep != that.getMaxStep()) return false;
        if (numThread != that.getNumThread()) return false;
        if (nstep != that.getNstep()) return false;
        if (targetDqnUpdateFreq != that.getTargetDqnUpdateFreq()) return false;
        if (updateStart != that.getUpdateStart()) return false;
        if (Double.compare(that.getRewardFactor(), rewardFactor) != 0) return false;
        if (Double.compare(that.getGamma(), gamma) != 0) return false;
        if (Double.compare(that.getErrorClamp(), errorClamp) != 0) return false;
        if (Float.compare(that.getMinEpsilon(), minEpsilon) != 0) return false;
        return epsilonNbStep == that.getEpsilonNbStep();

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
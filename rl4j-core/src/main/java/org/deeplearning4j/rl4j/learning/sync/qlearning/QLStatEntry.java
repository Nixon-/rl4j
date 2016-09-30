package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.rl4j.util.DataManager;

import java.util.List;

class QLStatEntry implements DataManager.StatEntry {
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
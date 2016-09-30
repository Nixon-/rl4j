package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.gym.StepReply;

public class QLStepReturn<O> {
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
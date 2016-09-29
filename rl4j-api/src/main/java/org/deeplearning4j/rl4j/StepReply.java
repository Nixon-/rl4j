package org.deeplearning4j.rl4j;

import org.json.JSONObject;

/**
 * @param <T> type of observation
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/6/16.
 *
 *  StepReply is the container for the data returned after each step(action).
 */
public class StepReply<T> {

    private T observation;
    private double reward;
    private boolean done;
    private JSONObject info;

    public StepReply(final T observation, final double reward, final boolean done, final JSONObject info) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }

    public T getObservation() {
        return this.observation;
    }

    public double getReward() {
        return this.reward;
    }

    public boolean getDone() {
        return this.done;
    }

    public JSONObject getInfo() {
        return this.info;
    }

}

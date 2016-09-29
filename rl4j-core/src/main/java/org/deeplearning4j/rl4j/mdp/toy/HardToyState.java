package org.deeplearning4j.rl4j.mdp.toy;

import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 */
public class HardToyState implements Encodable {

    private double[] values;
    private int step;

    HardToyState(final double[] values, final int step) {
        this.values = values;
        this.step = step;
    }

    public double[] toArray() {
        return values;
    }

    public double[] getValues() {
        return this.values;
    }

    public int getStep() {
        return this.step;
    }
}

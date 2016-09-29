package org.deeplearning4j.rl4j.mdp.toy;

import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 */
class SimpleToyState implements Encodable {

    // Todo: Figure out what i is.
    private int i;

    private int step;

    SimpleToyState(final int i, final int step) {
        this.i = i;
        this.step = step;
    }

    @Override
    public double[] toArray() {
        double[] ar = new double[1];
        ar[0] = (20-i);
        return ar;
    }

    /**
     * @return  mystery value i;
     */
    public int getI() {
        return  this.i;
    }

    public int getStep() {
        return this.step;
    }

}

package org.deeplearning4j.rl4j.mdp.toy;


import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;

import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * A toy MDP where reward are given in every case.
 * Useful to debug
 */
public class SimpleToy implements MDP<SimpleToyState, Integer, DiscreteSpace> {

    private Logger log = LoggerFactory.getLogger("SimpleToy");
    final private int maxStep;
    //TODO 10 steps toy (always +1 reward2 actions), toylong (1000 steps), toyhard (7 actions, +1 only if
    // action = (step/100+step)%7, and toyStoch (like last but reward has 0.10 odd to be somewhere else).

    private DiscreteSpace actionSpace = new DiscreteSpace(2);

    private ObservationSpace<SimpleToyState> observationSpace = new ArrayObservationSpace<>(new int[]{1});
    private SimpleToyState simpleToyState;

    private NeuralNetFetchable<IDQN> fetchable;

    public SimpleToy(int maxStep) {
        this.maxStep = maxStep;
    }

    private void printTest(int maxStep) {
        INDArray input = Nd4j.create(maxStep, 1);
        for (int i = 0; i < maxStep; i++) {
            input.putRow(i, Nd4j.create(new SimpleToyState(i, i).toArray()));
        }
        INDArray output = fetchable.getNeuralNet().output(input);
        log.info(output.toString());
    }

    public void close() {
    }

    @Override
    public boolean isDone() {
        return simpleToyState.getStep() == maxStep;
    }

    @Override
    public ObservationSpace<SimpleToyState> getObservationSpace() {
        return this.observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return this.actionSpace;
    }

    public SimpleToyState reset() {
        if (fetchable != null)
            printTest(maxStep);

        return simpleToyState = new SimpleToyState(0, 0);
    }

    public StepReply<SimpleToyState> step(Integer a) {
        double reward = (simpleToyState.getStep() %  2 == 0) ? 1 - a: a;
        simpleToyState = new SimpleToyState(simpleToyState.getI()+1, simpleToyState.getStep() + 1);
        return new StepReply<>(simpleToyState, reward, isDone(), new JSONObject("{}"));
    }

    public SimpleToy newInstance() {
        SimpleToy simpleToy = new SimpleToy(maxStep);
        simpleToy.setFetchable(fetchable);
        return simpleToy;
    }

    public SimpleToy setFetchable(final NeuralNetFetchable<IDQN> fetchable) {
        this.fetchable = fetchable;
        return this;
    }

}

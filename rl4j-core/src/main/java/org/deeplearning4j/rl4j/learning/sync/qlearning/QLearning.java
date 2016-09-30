package org.deeplearning4j.rl4j.learning.sync.qlearning;


import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.SyncLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.EpsGreedy;

import org.deeplearning4j.rl4j.util.DataManager.StatEntry;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 *
 * Mother class for QLearning in the Discrete domain and
 * hopefully one day for the  Continuous domain.
 */
public abstract class QLearning<O extends Encodable, A, AS extends ActionSpace<A>>
        extends SyncLearning<O, A, AS, IDQN> {

    final private IExpReplay<A> expReplay;

    public QLearning(QLConfiguration conf) {
        super(conf);
        expReplay = new ExpReplay<>(conf.getExpRepMaxSize(), conf.getBatchSize());
    }

    protected abstract EpsGreedy<O, A, AS> getEgPolicy();

    public abstract MDP<O, A, AS> getMdp();

    protected abstract IDQN getCurrentDQN();

    public abstract IDQN getTargetDQN();

    public abstract void setTargetDQN(IDQN dqn);

    protected IExpReplay<A> getExpReplay() {
        return this.expReplay;
    }

    protected INDArray dqnOutput(INDArray input) {
        return getCurrentDQN().output(input);
    }

    protected INDArray targetDqnOutput(INDArray input) {
        return getTargetDQN().output(input);
    }

    private void updateTargetNetwork() {
        getLogger().info("Update target network");
        setTargetDQN(getCurrentDQN().clone());
    }

    public IDQN getNeuralNet() {
        return getCurrentDQN();
    }

    public abstract QLConfiguration getConfiguration();

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract QLStepReturn<O> trainStep(O obs);

    protected StatEntry trainEpoch() {
        InitMdp<O> initMdp = initMdp();
        O obs = initMdp.getLastObs();

        double reward = initMdp.getReward();
        int step = initMdp.getSteps();

        Double startQ = Double.NaN;
        double meanQ = 0;
        int numQ = 0;
        List<Double> scores = new ArrayList<>();
        while (step < getConfiguration().getMaxEpochStep() && !getMdp().isDone()) {

            if (getStepCounter() % getConfiguration().getTargetDqnUpdateFreq() == 0) {
                updateTargetNetwork();
            }

            QLStepReturn<O> stepR = trainStep(obs);

            if (!stepR.getMaxQ().isNaN()) {
                if (startQ.isNaN())
                    startQ = stepR.getMaxQ();
                numQ++;
                meanQ += stepR.getMaxQ();
            }

            if (stepR.getScore() != 0) {
                scores.add(stepR.getScore());
            }

            reward += stepR.getStepReply().getReward();
            obs = stepR.getStepReply().getObservation();
            incrementStep();
            step++;
        }

        meanQ /= (numQ + 0.001); //avoid div zero

        return new QLStatEntry(getStepCounter(),
                getEpochCounter(), reward, step, scores, getEgPolicy().getEpsilon(), startQ, meanQ);
    }
}

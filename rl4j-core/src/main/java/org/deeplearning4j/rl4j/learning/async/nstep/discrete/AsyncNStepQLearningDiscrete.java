package org.deeplearning4j.rl4j.learning.async.nstep.discrete;


import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.Policy;

import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public abstract class AsyncNStepQLearningDiscrete<O extends Encodable>
        extends AsyncLearning<O, Integer, DiscreteSpace, IDQN> {

    final private AsyncNStepQLConfiguration configuration;
    final private MDP<O, Integer, DiscreteSpace> mdp;
    final private DataManager dataManager;
    final private AsyncGlobal<IDQN> asyncGlobal;

    AsyncNStepQLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, AsyncNStepQLConfiguration conf,
                                DataManager dataManager) {
        super(conf);
        this.mdp = mdp;
        this.dataManager = dataManager;
        this.configuration = conf;
        this.asyncGlobal = new AsyncGlobal<>(dqn, conf);
    }


    public AsyncThread newThread(int i) {
        return new AsyncNStepQLearningThreadDiscrete<>(mdp.newInstance(), asyncGlobal, configuration, i, dataManager);
    }

    public IDQN getNeuralNet() {
        return asyncGlobal.cloneCurrent();
    }

    public Policy<O, Integer> getPolicy() {
        return new DQNPolicy<>(getNeuralNet());
    }

    @Override
    public AsyncNStepQLConfiguration getConfiguration() {
        return configuration;
    }

    @Override
    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return mdp;
    }

    @Override
    public DataManager getDataManager() {
        return dataManager;
    }

    public AsyncGlobal<IDQN> getAsyncGlobal() {
        return asyncGlobal;
    }

}

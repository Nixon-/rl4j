package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.function.Function;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * DQN or Deep Q-Learning in the Discrete domain
 *
 * http://arxiv.org/abs/1312.5602
 *
 */
public abstract class QLearningDiscrete<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

    final private QLConfiguration configuration;

    final private DataManager dataManager;

    final private MDP<O, Integer, DiscreteSpace> mdp;

    final private IDQN currentDQN;

    private DQNPolicy<O> policy;

    private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;

    private IDQN targetDQN;
    private int lastAction;
    private INDArray history[] = null;
    private double accuReward = 0;
    private int lastMonitor = -Constants.MONITOR_FREQ;


    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf, DataManager dataManager, int epsilonNbStep) {
        super(conf);
        this.configuration = conf;
        this.mdp = mdp;
        this.dataManager = dataManager;
        currentDQN = dqn;
        targetDQN = dqn.clone();
        policy = new DQNPolicy<>(getCurrentDQN());
        egPolicy = new EpsGreedy<>(policy, mdp, conf.getUpdateStart(), epsilonNbStep, conf.getMinEpsilon(), this);
    }

    public void postEpoch() {
        if (getHistoryProcessor() != null) {
            getHistoryProcessor().stopMonitor();
        }
    }

    public void preEpoch() {
        history = null;
        lastAction = 0;
        accuReward = 0;

        if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null
                && getDataManager().isSaveData()) {
            lastMonitor = getStepCounter();
            getHistoryProcessor().startMonitor(getDataManager().getVideoDir() + "/video-" +
                    getEpochCounter() + "-" + getStepCounter() + ".mp4");
        }
    }

    /**
     * Single step of training
     * @param obs last obs
     * @return relevant info for next step
     */
    protected QLStepReturn<O> trainStep(O obs) {

        Integer action;
        INDArray input = getInput(obs);
        boolean isHistoryProcessor = getHistoryProcessor() != null;


        if (isHistoryProcessor)
            getHistoryProcessor().record(input);

        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;
        int historyLength = isHistoryProcessor ? getHistoryProcessor().getConf().getHistoryLength() : 1;
        int updateStart = getConfiguration().getUpdateStart() + ((getConfiguration().getBatchSize() + historyLength) * skipFrame);

        Double maxQ = Double.NaN; //ignore if Nan for stats

        //if step of training, just repeat lastAction
        if (getStepCounter() % skipFrame != 0) {
            action = lastAction;
        } else {
            if (history == null) {
                if (isHistoryProcessor) {
                    getHistoryProcessor().add(input);
                    history = getHistoryProcessor().getHistory();
                } else
                    history = new INDArray[]{input};
            }
            //concat the history into a single INDArray input
            INDArray hstack = Transition.concat(Transition.dup(history));

            //if input is not 2d, you have to append that the batch is 1 length high
            if (hstack.shape().length > 2)
                hstack = hstack.reshape(Learning.makeShape(1, hstack.shape()));

            INDArray qs = getCurrentDQN().output(hstack);
            int maxAction = Learning.getMaxAction(qs);

            maxQ = qs.getDouble(maxAction);
            action = getEgPolicy().nextAction(hstack);
        }

        lastAction = action;

        StepReply<O> stepReply = getMdp().step(action);

        accuReward += stepReply.getReward()*configuration.getRewardFactor();

        //if it's not a skipped frame, you can do a step of training
        if (getStepCounter() % skipFrame == 0 || stepReply.isDone()) {

            INDArray ninput = getInput(stepReply.getObservation());
            if (isHistoryProcessor)
                getHistoryProcessor().add(ninput);

            INDArray[] nhistory = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[]{ninput};

            Transition<Integer> trans = new Transition<>(history, action, accuReward, stepReply.isDone(),
                    nhistory[0]);
            getExpReplay().store(trans);

            if (getStepCounter() > updateStart) {
                Pair<INDArray, INDArray> targets = setTarget(getExpReplay().getBatch());
                getCurrentDQN().fit(targets.getFirst(), targets.getSecond());
            }

            history = nhistory;
            accuReward = 0;
        }

        return new QLStepReturn<>(maxQ, getCurrentDQN().getLatestScore(), stepReply);
    }

    private Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
        if (transitions.size() == 0)
            throw new IllegalArgumentException("too few transitions");

        int size = transitions.size();

        int[] shape = getHistoryProcessor() == null ?
                getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        int[] nshape = makeShape(size, shape);
        INDArray obs = Nd4j.create(nshape);
        INDArray nextObs = Nd4j.create(nshape);
        int[] actions = new int[size];
        boolean[] areTerminal = new boolean[size];

        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);
            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.getAction();
            obs.putRow(i, Transition.concat(trans.getObservation()));
            nextObs.putRow(i, Transition.concat(Transition.append(trans.getObservation(),
                    trans.getNextObservation())));
        }

        INDArray dqnOutputAr = dqnOutput(obs);

        INDArray dqnOutputNext = dqnOutput(nextObs);
        INDArray qFunc;
        Function<Integer, Double> action;

        if (getConfiguration().isDoubleDQN()) {
            qFunc = targetDqnOutput(nextObs);
            final INDArray getMaxAction = Nd4j.argMax(dqnOutputNext, 1);
            action = (i) -> qFunc.getDouble(i, getMaxAction.getInt(i));

        } else {
            qFunc = Nd4j.max(dqnOutputNext, 1);
            action = qFunc::getDouble;
        }


        for (int i = 0; i < size; i++) {
            double yTar = transitions.get(i).getReward();
            if (!areTerminal[i]) {
                yTar += getConfiguration().getGamma() * action.apply(i);
            }
            double previousV = dqnOutputAr.getDouble(i, actions[i]);
            double lowB = previousV - getConfiguration().getErrorClamp();
            double highB = previousV + getConfiguration().getErrorClamp();
            double clamped = Math.min(highB, Math.max(yTar, lowB));

            dqnOutputAr.putScalar(i, actions[i], clamped);
        }

        return new Pair<>(obs, dqnOutputAr);
    }

    @Override
    public DataManager getDataManager() {
        return dataManager;
    }

    @Override
    public DQNPolicy<O> getPolicy() {
        return this.policy;
    }

    @Override
    public EpsGreedy<O, Integer, DiscreteSpace> getEgPolicy() {
        return this.egPolicy;
    }

    @Override
    public IDQN getCurrentDQN() {
        return this.currentDQN;
    }

    @Override
    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return this.mdp;
    }

    @Override
    public QLConfiguration getConfiguration() {
        return this.configuration;
    }

    @Override
    public void setTargetDQN(IDQN newTarget) {
        this.targetDQN = newTarget;
    }

    @Override
    public IDQN getTargetDQN() {
        return this.targetDQN;
    }

}

package org.deeplearning4j.rl4j.learning.async;


import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * This represent a local thread that explore the environment
 * and calculate a gradient to enqueue to the global thread/model
 *
 * It has its own version of a model that it syncs at the start of every
 * sub epoch
 *
 */
public abstract class AsyncThread<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet> extends Thread implements StepCountable {

    private final Logger log;
    private int stepCounter = 0;
    private int epochCounter = 0;
    private IHistoryProcessor historyProcessor;

    AsyncThread(AsyncGlobal<NN> asyncGlobal, int threadNumber) {
        log = LoggerFactory.getLogger("ThreadNum-" + threadNumber);
    }

    public void setHistoryProcessor(IHistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    @Override
    public int getStepCounter() {
        return stepCounter;
    }

    public int getEpochCounter() {
        return epochCounter;
    }

    public IHistoryProcessor getHistoryProcessor() {
        return historyProcessor;
    }

    @Override
    public void run() {


        try {
            log.info("Started!");
            Learning.InitMdp<O> initMdp = Learning.initMdp(getMdp(), historyProcessor);
            O obs = initMdp.getLastObs();
            double rewards = initMdp.getReward();
            int length = initMdp.getSteps();

            while (!getAsyncGlobal().isTrainingComplete() && getAsyncGlobal().isRunning()) {
                SubEpochReturn<O> subEpochReturn = trainSubEpoch(obs, getConf().getNstep());
                obs = subEpochReturn.getLastObs();
                stepCounter += subEpochReturn.getSteps();
                length += subEpochReturn.getSteps();
                rewards += subEpochReturn.getReward();
                double score = subEpochReturn.getScore();
                if (getMdp().isDone()) {

                    if (getThreadNumber() == 1)
                        getDataManager().appendStat(new AsyncStatEntry(getStepCounter(), epochCounter, rewards, length, score));

                    initMdp = Learning.initMdp(getMdp(), historyProcessor);
                    obs = initMdp.getLastObs();
                    rewards = initMdp.getReward();
                    length = initMdp.getSteps();
                    epochCounter++;
                }
            }
        } catch (Exception e) {
            log.error("Thread crashed");
            getAsyncGlobal().setRunning(false);
            e.printStackTrace();
        }
    }

    protected abstract int getThreadNumber();

    protected abstract AsyncGlobal<NN> getAsyncGlobal();

    protected abstract MDP<O, A, AS> getMdp();

    protected abstract AsyncConfiguration getConf();

    protected abstract DataManager getDataManager();

    protected abstract Policy<O, A> getPolicy(NN net);

    protected abstract SubEpochReturn<O> trainSubEpoch(O obs, int nstep);

    static class SubEpochReturn<O> {
        private final int steps;
        private final O lastObs;
        private final double reward;
        private final double score;

        public SubEpochReturn(final int steps, final O lastObs, final double reward, final double score) {
            this.steps = steps;
            this.lastObs = lastObs;
            this.reward = reward;
            this.score = score;
        }

        public int getSteps() {
            return steps;
        }

        public O getLastObs() {
            return lastObs;
        }

        public double getReward() {
            return reward;
        }

        public double getScore() {
            return score;
        }
    }

    private static class AsyncStatEntry implements DataManager.StatEntry {
        private final int stepCounter;
        private final int epochCounter;
        private final double reward;
        private final int episodeLength;
        private final double score;

        AsyncStatEntry(final int stepCounter, final int epochCounter, final double reward,
                       final int episodeLength, final double score) {
            this.stepCounter = stepCounter;
            this.epochCounter = epochCounter;
            this.reward = reward;
            this.episodeLength = episodeLength;
            this.score = score;
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

        public double getScore() {
            return score;
        }
    }

}

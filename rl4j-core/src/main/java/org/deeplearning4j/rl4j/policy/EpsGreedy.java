package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.security.SecureRandom;
import java.util.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/24/16.
 *
 * An epsilon greedy policy choose the next action
 * - randomly with epsilon probability
 * - deleguate it to constructor argument 'policy' with (1-epsilon) probability.
 *
 * epislon is annealed to minEpsilon over epsilonNbStep steps
 *
 */
public class EpsGreedy<O extends Encodable, A, AS extends ActionSpace<A>> extends Policy<O, A> {

    final private Logger log = LoggerFactory.getLogger("EpsGreedy");
    final private Policy<O, A> policy;
    final private MDP<O, A, AS> mdp;
    final private int updateStart;
    final private int epsilonNbStep;
    final private Random rd;
    final private double minEpsilon;
    final private StepCountable learning;

    public EpsGreedy(Policy<O, A> policy, MDP<O, A, AS> mdp, int updateStart, int epsilonNbStep,
                     double minEpsilon, StepCountable learning) {
        this.policy = policy;
        this.mdp = mdp;
        this.updateStart = updateStart;
        this.epsilonNbStep = epsilonNbStep;
        this.rd = new SecureRandom();
        this.minEpsilon = minEpsilon;
        this.learning = learning;
    }

    public A nextAction(INDArray input) {

        double ep = getEpsilon();
        if (learning.getStepCounter() % 500 == 1)
            log.info("EP: " + ep + " " + learning.getStepCounter());
        if (rd.nextFloat() > ep)
            return policy.nextAction(input);
        else {
            return mdp.getActionSpace().randomAction();
        }
    }

    public Policy<O, A> getPolicy() {
        return policy;
    }

    public MDP<O, A, AS> getMdp() {
        return mdp;
    }

    public int getUpdateStart() {
        return updateStart;
    }

    public int getEpsilonNbStep() {
        return epsilonNbStep;
    }

    public Random getRd() {
        return rd;
    }

    public double getMinEpsilon() {
        return minEpsilon;
    }

    public StepCountable getLearning() {
        return learning;
    }

    public double getEpsilon() {
        return Math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter() - updateStart) * 1f/epsilonNbStep));
    }
}

package org.deeplearning4j.rl4j.mdp.vizdoom;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;
import oshi.util.FormatUtil;
import vizdoom.*;

import java.util.ArrayList;
import java.util.List;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/28/16.
 *
 * Mother abstract class for all VizDoom scenarios
 *
 * is mostly configured by
 *
 *    String scenarioName;       name of the scenarioName
 *    double livingReward;   additional reward at each step for living
 *    double deathPenalty;   negative reward when ded
 *    int doomSkill;         skill of the ennemy
 *    int timeout;           number of step after which simulation time out
 *    int startTime;         number of internal tics before the simulation starts (useful to draw weapon by example)
 *    List<Button> buttons;  the list of inputs one can press for a given scenarioName (noop is automatically added)
 *
 *
 *
 */
public abstract class VizDoom implements MDP<VizDoom.GameScreen, Integer, DiscreteSpace> {

    final private static String DOOM_ROOT = "vizdoom";

    private DoomGame game;
    final private Logger log = LoggerFactory.getLogger("Vizdoom");
    final private GlobalMemory memory = new SystemInfo().getHardware().getMemory();
    final private List<int[]> actions;
    final private DiscreteSpace discreteSpace;
    final private ObservationSpace<GameScreen> observationSpace;
    final private boolean render;
    private double scaleFactor = 1;

    public VizDoom() {
        this(false);
    }

    VizDoom(boolean render) {
        this.render = render;
        actions = new ArrayList<>();
        game = new DoomGame();
        setupGame();
        discreteSpace = new DiscreteSpace(getConfiguration().getButtons().size() + 1);
        observationSpace = new ArrayObservationSpace<>(
                new int[]{game.getScreenHeight(), game.getScreenWidth(), 3});
    }


    private void setupGame() {

        Configuration conf = getConfiguration();

        game.setViZDoomPath(DOOM_ROOT + "/vizdoom");
        game.setDoomGamePath(DOOM_ROOT + "/scenarios/freedoom2.wad");
        game.setDoomScenarioPath(DOOM_ROOT + "/scenarios/" + conf.getScenarioName() + ".wad");

        game.setDoomMap("map01");

        game.setScreenFormat(ScreenFormat.RGB24);
        game.setScreenResolution(ScreenResolution.RES_800X600);
        // Sets other rendering options
        game.setRenderHud(false);
        game.setRenderCrosshair(false);
        game.setRenderWeapon(true);
        game.setRenderDecals(false);
        game.setRenderParticles(false);


        GameVariable[] gameVar = new GameVariable[]{
                GameVariable.KILLCOUNT,
                GameVariable.ITEMCOUNT,
                GameVariable.SECRETCOUNT,
                GameVariable.FRAGCOUNT,
                GameVariable.HEALTH,
                GameVariable.ARMOR,
                GameVariable.DEAD,
                GameVariable.ON_GROUND,
                GameVariable.ATTACK_READY,
                GameVariable.ALTATTACK_READY,
                GameVariable.SELECTED_WEAPON,
                GameVariable.SELECTED_WEAPON_AMMO,
                GameVariable.AMMO1,
                GameVariable.AMMO2,
                GameVariable.AMMO3,
                GameVariable.AMMO4,
                GameVariable.AMMO5,
                GameVariable.AMMO6,
                GameVariable.AMMO7,
                GameVariable.AMMO8,
                GameVariable.AMMO9,
                GameVariable.AMMO0
        };
        // Adds game variables that will be included in state.

        for (GameVariable aGameVar : gameVar) {
            game.addAvailableGameVariable(aGameVar);
        }

        // Causes episodes to finish after timeout tics
        game.setEpisodeTimeout(conf.getTimeout());

        game.setEpisodeStartTime(conf.getStartTime());

        game.setWindowVisible(render);
        game.setSoundEnabled(false);
        game.setMode(Mode.PLAYER);


        game.setLivingReward(conf.getLivingReward());

        // Adds buttons that will be allowed.
        List<Button> buttons = conf.getButtons();
        int size = buttons.size();

        actions.add(new int[size + 1]);
        for (int i = 0; i < size; i++) {
            game.addAvailableButton(buttons.get(i));
            int[] action = new int[size + 1];
            action[i] = 1;
            actions.add(action);
        }

        game.setDeathPenalty(conf.getDeathPenalty());
        game.setDoomSkill(conf.getDoomSkill());

        game.init();
    }

    VizDoom setScaleFactor(final double scaleFactor) {
        this.scaleFactor = scaleFactor;
        return this;
    }

    boolean isRender() {
        return this.render;
    }

    public boolean isDone() {
        return game.isEpisodeFinished();
    }

    public GameScreen reset() {
        log.info("free Memory: " + FormatUtil.formatBytes(memory.getAvailable()) + "/"
                + FormatUtil.formatBytes(memory.getTotal()));

        game.newEpisode();
        game.getGameScreen();
        return new GameScreen(game.getGameScreen());
    }


    public void close() {
        game.close();
    }


    public StepReply<GameScreen> step(Integer action) {
        double r = game.makeAction(actions.get(action)) * scaleFactor;
        log.info(game.getEpisodeTime() + " " + r + " " + action + " ");
        return new StepReply<>(new GameScreen(game.getGameScreen()), r, game.isEpisodeFinished(), null);
    }

    public ObservationSpace<GameScreen> getObservationSpace() {
        return observationSpace;
    }


    public DiscreteSpace getActionSpace() {
        return discreteSpace;
    }

    public abstract Configuration getConfiguration();

    public abstract VizDoom newInstance();

    public class Configuration {
        private String scenarioName;
        private double livingReward;
        private double deathPenalty;
        private int doomSkill;
        private int timeout;
        private int startTime;
        List<Button> buttons;

        Configuration(final String scenarioName, final double livingReward,
                             final double deathPenalty, final int doomSkill, final int timeout,
                             final int startTime, final List<Button>buttons) {
            this.setScenarioName(scenarioName)
                    .setLivingReward(livingReward)
                    .setDeathPenalty(deathPenalty)
                    .setDoomSkill(doomSkill)
                    .setTimeout(timeout)
                    .setStartTime(startTime)
                    .setButtons(buttons);
        }

        String getScenarioName() {
            return scenarioName;
        }

        Configuration setScenarioName(final String scenarioName) {
            this.scenarioName = scenarioName;
            return this;
        }

        double getLivingReward() {
            return livingReward;
        }

        Configuration setLivingReward(final double livingReward) {
            this.livingReward = livingReward;
            return this;
        }

        double getDeathPenalty() {
            return deathPenalty;
        }

        Configuration setDeathPenalty(final double deathPenalty) {
            this.deathPenalty = deathPenalty;
            return this;
        }

        int getDoomSkill() {
            return doomSkill;
        }

        Configuration setDoomSkill(final int doomSkill) {
            this.doomSkill = doomSkill;
            return this;
        }

        int getTimeout() {
            return timeout;
        }

        Configuration setTimeout(int timeout) {
            this.timeout = timeout;
            return this;
        }

        int getStartTime() {
            return startTime;
        }

        Configuration setStartTime(int startTime) {
            this.startTime = startTime;
            return this;
        }

        List<Button> getButtons() {
            return buttons;
        }

        Configuration setButtons(List<Button> buttons) {
            this.buttons = buttons;
            return this;
        }
    }

    public static class GameScreen implements Encodable {
        final private double[] array;

        GameScreen(int[] screen) {
            array = new double[screen.length];
            for (int i = 0; i < screen.length; i++) {
                array[i] = screen[i];
            }
        }

        public double[] toArray() {
            return array;
        }
    }

}

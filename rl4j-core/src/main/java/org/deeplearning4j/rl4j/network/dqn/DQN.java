package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 */
public class DQN implements IDQN {

    private final MultiLayerNetwork mln;

    public DQN(final MultiLayerNetwork mln) {
        this.mln = mln;
    }

    public static DQN load(String path) {
        DQN dqn = null;
        try {
            dqn = new DQN(ModelSerializer.restoreMultiLayerNetwork(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dqn;
    }

    public void fit(INDArray input, INDArray labels) {
        mln.fit(input, labels);
    }

    public void fit(INDArray input, INDArray[] labels) {
        fit(input, labels[0]);
    }

    public INDArray output(INDArray batch) {
        return mln.output(batch);
    }

    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[]{output(batch)};
    }

    @Override
    public DQN clone() {
        return new DQN(mln.clone());
    }

    public Gradient[] gradient(INDArray input, INDArray labels) {
        mln.setInput(input);
        mln.setLabels(labels);
        mln.computeGradientAndScore();
        //System.out.println("SCORE: " + mln.score());
        return new Gradient[]{mln.gradient()};
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        return gradient(input, labels[0]);
    }

    public void applyGradient(Gradient[] gradient, int batchSize) {
        mln.getUpdater().update(mln, gradient[0], 1, batchSize);
        mln.params().subi(gradient[0].gradient());
    }

    public double getLatestScore() {
        return mln.score();
    }

    public void save(OutputStream stream) {
        try {
            ModelSerializer.writeModel(mln, stream, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void save(String path) {
        try {
            ModelSerializer.writeModel(mln, path, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

package org.deeplearning4j.rl4j.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 *
 * An IHistoryProcessor come directly from the atari DQN paper.
 * It applies pre-processing the pixels of one state (gray-scaling + resizing)
 * then stacks it in different channels to be fed to a conv net
 */
public interface IHistoryProcessor {

    Configuration getConf();

    INDArray[] getHistory();

    void record(INDArray image);

    void add(INDArray image);

    void startMonitor(String filename);

    void stopMonitor();

    boolean isMonitoring();

    class Configuration {
        int historyLength;
        int rescaledWidth;
        int rescaledHeight;
        int croppingWidth;
        int croppingHeight;
        int offsetX;
        int offsetY;
        int skipFrame;

        /**
         * Default constructor.
         */
        public Configuration() {
            historyLength = 4;
            rescaledWidth = 84;
            rescaledHeight = 84;
            croppingWidth = 84;
            croppingHeight = 84;
            offsetX = 0;
            offsetY = 0;
            skipFrame = 4;
        }

        /**
         * @return Shape of image, where 0=history length, 1=cropping width, 2=cropping height.
         */
        public int[] getShape() {
            return new int[]{
                    getHistoryLength(),
                    getCroppingWidth(),
                    getCroppingHeight()};
        }

        /**
         * @return  Length of History.
         */
        public int getHistoryLength() {
            return this.historyLength;
        }

        /**
         * @return  Rescaled Width.
         */
        public int getRescaledWidth() {
            return this.rescaledWidth;
        }

        public int getRescaledHeight() {
            return this.rescaledHeight;
        }

        /**
         * @return  Cropping width.
         */
        public int getCroppingWidth() {
            return this.croppingWidth;
        }

        /**
         * @return  Cropping Height.
         */
        public int getCroppingHeight() {
            return this.croppingHeight;
        }

        /**
         * @return  X axis offset.
         */
        public int getOffsetX() {
            return this.offsetX;
        }

        /**
         * @return  Y axis offset.
         */
        public int getOffsetY() {
            return this.offsetY;
        }

        /**
         * @return  Skip Frame value.
         */
        public int getSkipFrame(){
            return this.skipFrame;
        }
    }
}

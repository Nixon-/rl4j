package org.deeplearning4j.rl4j.util;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/1/16.
 *
 * This class handle the recording of one training.
 * It creates the directory rl4j-data if it does not exist,
 * the folder for every training and handle every path and model savings
 */
public class DataManager {


    final private Logger log = LoggerFactory.getLogger("DataManager");
    final private String home = System.getProperty("user.home");
    final private ObjectMapper mapper = new ObjectMapper();
    private String dataRoot = home + "/" + Constants.DATA_DIR;
    private boolean saveData;
    private String currentDir;

    public DataManager() {
        create(dataRoot, false);
    }

    public DataManager(boolean saveData) {
        create(dataRoot, saveData);
    }

    public DataManager(String dataRoot, boolean saveData) {
        create(dataRoot, saveData);
    }

    private static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
        byte[] bytes = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(bytes)) != -1) {
            zipStream.write(bytes, 0, bytesRead);
        }
    }

    public static void save(String path, Learning learning) {
        BufferedOutputStream os = null;
        try {
            os = new BufferedOutputStream(new FileOutputStream(path));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        save(os, learning);
    }

    public static void save(OutputStream os, Learning learning) {

        ZipOutputStream zipfile = new ZipOutputStream(os);

        try {

            ZipEntry config = new ZipEntry("configuration.json");
            zipfile.putNextEntry(config);
            String json = new ObjectMapper().writeValueAsString(learning.getConfiguration());
            writeEntry(new ByteArrayInputStream(json.getBytes()), zipfile);

            ZipEntry dqn = new ZipEntry("dqn.bin");
            zipfile.putNextEntry(dqn);

            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            learning.getNeuralNet().save(bos);
            bos.flush();
            bos.close();

            InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
            writeEntry(inputStream, zipfile);


            if (learning.getHistoryProcessor() != null) {
                ZipEntry hpconf = new ZipEntry("hpconf.bin");
                zipfile.putNextEntry(hpconf);

                ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
                learning.getNeuralNet().save(bos2);
                bos2.flush();
                bos2.close();

                InputStream inputStream2 = new ByteArrayInputStream(bos2.toByteArray());
                writeEntry(inputStream2, zipfile);
            }


            zipfile.flush();
            zipfile.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static <C> Pair<IDQN, C> load(File file, Class<C> cClass) {
        LoggerFactory.getLogger("Serializer").info("Deserializing: " + file.getName());

        C conf = null;
        IDQN dqn = null;
        try {
            ZipFile zipFile = new ZipFile(file);
            ZipEntry config = zipFile.getEntry("configuration.json");
            InputStream stream = zipFile.getInputStream(config);
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String line = "";
            StringBuilder js = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                js.append(line).append("\n");
            }
            String json = js.toString();

            reader.close();
            stream.close();

            conf = new ObjectMapper().readValue(json, cClass);

            ZipEntry dqnzip = zipFile.getEntry("dqn.bin");
            InputStream dqnstream = zipFile.getInputStream(dqnzip);
            File tmpFile = File.createTempFile("restore", "dqn");
            Files.copy(dqnstream, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
            dqn = new DQN(ModelSerializer.restoreMultiLayerNetwork(tmpFile));
            dqnstream.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

        return new Pair<>(dqn, conf);
    }

    public static <C> Pair<IDQN, C> load(String path, Class<C> cClass) {
        return load(new File(path), cClass);
    }

    public static <C> Pair<IDQN, C> load(InputStream is, Class<C> cClass) {
        try {
            File tmpFile = File.createTempFile("restore", "learning");
            Files.copy(is, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
            return load(tmpFile, cClass);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null; //has crashed anyway
    }

    private void create(String dataRoot, boolean saveData) {
        this.saveData = saveData;
        this.dataRoot = dataRoot;
        createSubdir();
    }

    //FIXME race condition if you create them at the same time where checking if dir exists is
    // not atomic with the creation
    public String createSubdir() {

        if (!saveData)
            return "";

        File dr = new File(dataRoot);
        dr.mkdirs();
        File[] rootChildren = dr.listFiles();

        int i = 1;
        while (childrenExist(rootChildren, i + ""))
            i++;

        File f = new File(dataRoot + "/" + i);
        f.mkdirs();

        currentDir = f.getAbsolutePath();
        log.info("Created training data directory: " + currentDir);

        File mov = new File(getVideoDir());
        mov.mkdirs();

        File model = new File(getModelDir());
        model.mkdirs();

        File stat = new File(getStat());
        File info = new File(getInfo());
        try {
            stat.createNewFile();
            info.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return f.getAbsolutePath();
    }

    public String getVideoDir() {
        return currentDir + "/" + Constants.VIDEO_DIR;
    }

    public String getModelDir() {
        return currentDir + "/" + Constants.MODEL_DIR;
    }

    public String getInfo() {
        return currentDir + "/" + Constants.INFO_FILENAME;
    }

    public String getStat() {
        return currentDir + "/" + Constants.STATISTIC_FILENAME;
    }

    public boolean isSaveData() {
        return saveData;
    }

    public void appendStat(StatEntry statEntry) {

        if (!saveData)
            return;

        Path statPath = Paths.get(getStat());
        String toAppend = toJson(statEntry);
        try {
            Files.write(statPath, toAppend.getBytes(), StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private String toJson(Object object) {
        try {
            return mapper.writeValueAsString(object) + "\n";
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return "";
    }

    public void writeInfo(ILearning iLearning) {

        if (!saveData)
            return;

        Path infoPath = Paths.get(getInfo());

        Info info = new Info(iLearning.getClass().getSimpleName(), iLearning.getMdp().getClass().getSimpleName(),
                iLearning.getConfiguration(), iLearning.getStepCounter(), System.currentTimeMillis());
        String toWrite = toJson(info);

        try {
            Files.write(infoPath, toWrite.getBytes(), StandardOpenOption.TRUNCATE_EXISTING);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private boolean childrenExist(File[] files, String children) {
        boolean exists = false;
        for (int i = 0; i < files.length; i++) {
            if (files[i].getName().equals(children)) {
                exists = true;
                break;
            }
        }
        return exists;
    }

    public void save(Learning learning) {

        if (!saveData)
            return;

        save(getModelDir() + "/" + learning.getStepCounter() + ".training", learning);
        learning.getNeuralNet().save(getModelDir() + "/" + learning.getStepCounter() + ".model");

    }

    //In order for jackson to serialize StatEntry
    //please use Lombok @Value (see QLStatEntry)
    public interface StatEntry {
        int getEpochCounter();
        int getStepCounter();
        double getReward();
    }

    private static class Info {
        private final String trainingName;
        private final String mdpName;
        private final ILearning.LConfiguration conf;
        private final int stepCounter;
        private final long millisTime;

        Info(final String trainingName, final String mdpName, final ILearning.LConfiguration conf,
             final int stepCounter, final long millisTime) {
            this.trainingName = trainingName;
            this.mdpName = mdpName;
            this.conf = conf;
            this.stepCounter = stepCounter;
            this.millisTime = millisTime;
        }

        public String getTrainingName() {
            return trainingName;
        }

        public String getMdpName() {
            return mdpName;
        }

        public ILearning.LConfiguration getConf() {
            return conf;
        }

        public int getStepCounter() {
            return stepCounter;
        }

        public long getMillisTime() {
            return millisTime;
        }
    }
}

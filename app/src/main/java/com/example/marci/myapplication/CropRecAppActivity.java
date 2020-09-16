package com.example.marci.myapplication;

import android.Manifest;
import android.animation.Animator;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;

import android.util.Log;
import android.view.View;
import android.widget.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.MalformedURLException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import static android.Manifest.permission.*;

public class CropRecAppActivity extends AppCompatActivity  implements SensorEventListener {

    //Global variables
    private String LOG_TAG = CropRecAppActivity.class.getSimpleName();
    private SensorManager sensorManager;
    private final float[] accelerometerReading = new float[3];
    private final float[] magnetometerReading = new float[3];
    private final float[] rotationMatrix = new float[9];
    private final float[] mOrientationAngles = new float[3];
    DataNormalization trainImageScaler;
    private MultiLayerNetwork net;
    private static int IMAGE_WIDTH;
    private static int IMAGE_HEIGHT;
    private static int IMAGE_CHANNELS;
    ImageView sampleIV, outputIcon;
    Button takePictureBtn, selectPictureBtn;
    Uri imagenUri;
    TextView confTv;

    // Create the File where the photo should go
    File photoFile = null;

    RelativeLayout outLayout;
    ProgressBar progressBar;
    LinearLayout locationLayout;
    private LocationManager manager;
    private LocationListener locationListener;
    private String userLatitude;
    private String userLongitude;
    private TextView latitudeTV;
    private TextView longitudeTV;
    private TextView orientationTextView;
    private double[][] orientationArray;

    //Constants
    private static final int LOCATION_RQST = 800;
    private final int TAKE_PIC_WRITE_PERMISSION_RQST = 700;
    private final int OPEN_CAMERA_RQST = 200;
    private final int SELEC_IMAGEN = 300;
    private final int WRITE_PERMISSION_RQST = 400;
    private final int CAMERA_PERMISSION_RQST = 500;
    String[] classLabels = {"maize", "wheat"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTheme(R.style.AppTheme);
        setContentView(R.layout.croprecapp_layout);

        //Sensor Manager for phone orientation
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        //orientationTextView displays phones possible orientation: N, NE, E, SE, S, SW, W, NW
        orientationTextView = (TextView) findViewById(R.id.orientation_textview);

        //sampleIV displays image to be tested by CNN model
        sampleIV = findViewById(R.id.sample_iv);

        //confTv shows the level of confidence predicted by the CNN Model
        confTv = (TextView) findViewById(R.id.conf_tv);

        //outLayout shows the output class label
        outLayout = (RelativeLayout) findViewById(R.id.output_layout);

        //outputIcon shows the output class icon
        outputIcon = (ImageView) findViewById(R.id.output_icon);

        progressBar = (ProgressBar) findViewById(R.id.progress_horizontal) ;

        // locationLayout shows the users latitude and longitude
        locationLayout = (LinearLayout)findViewById(R.id.location_layout);
        latitudeTV = findViewById(R.id.latitude_textview);
        longitudeTV = findViewById(R.id.longitude_textview);

        offProgressBar();


        ///takePictureBtn launches phones camera for taking a picture
        takePictureBtn = findViewById(R.id.take_pic);

        //selectPictureBtn opens a FileChooser in order to select an image from galery
        selectPictureBtn = findViewById(R.id.select_img_btn);

        takePictureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                //Checks for granted Camera Permission
                if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

                    ActivityCompat.requestPermissions(CropRecAppActivity.this,
                            new String[]{CAMERA}, CAMERA_PERMISSION_RQST);

                }else{

                    takePicture();

                }

            }
        });

        selectPictureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                selectImage();
            }
        });

        orientationArray = new double[1][18];

    }

    /**
     * This method initializes the sample pre processing parameters.
     * It takes the needed image height, width and channel quantity, and the needed image values normalization.
     *
     * This values are based on the training data.
     */
    private void initSampleInputParams() {

        FeedForwardToCnnPreProcessor preProcessor =
                (FeedForwardToCnnPreProcessor) net.getLayerWiseConfigurations().getInputPreProcessors().get(0);
        IMAGE_HEIGHT = (int) preProcessor.getInputHeight();
        IMAGE_WIDTH = (int) preProcessor.getInputWidth();
        IMAGE_CHANNELS = (int) preProcessor.getNumChannels();

        Log.i(LOG_TAG, "inputHeight : " + IMAGE_HEIGHT);
        Log.i(LOG_TAG, "inputWidth : " + IMAGE_WIDTH);
        Log.i(LOG_TAG, "inputChannels : " + IMAGE_CHANNELS);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        try {
            trainImageScaler = getTrainImageScaler();
        } catch (IOException e) {
            Log.e(LOG_TAG,"getTrainImageScaler IOException: " + e.toString());
        }


    }

    /**
     * This method classifies the input image.
     *
     * @param  photoFile File with the test image to be classified.
     * @return String[] class label on index 0, and confidence value on index 1.
     */
    private String[] testImage(File photoFile) {

        String[] result = {"pred", "conf"};
        String prediction = "", probability = "";

        NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);

        try {

            INDArray image = loader.asMatrix(photoFile);
            trainImageScaler.transform(image);

            int[] outputPrediction = net.predict(image);
            prediction = classLabels[outputPrediction[0]];

            INDArray output = net.output(image);
            probability = String.valueOf(output.getDouble(outputPrediction[0]) * 100);

        } catch (IOException e) {

            Log.e(LOG_TAG, "net PREDICT IO EXCEPTION: " + e.toString());

        }

        result[0] = prediction;
        result[1] = probability;

        return result;
    }

    /**
     * This method launches phone location services and camera api.
     *
     */
    public void takePicture() {


        initLocationListener();
        dispatchTakePictureIntent();

    }
    /**
     * This method launches phone location change listener and {@link OrientationModelReaderAsyntask}
     */

    private void initLocationListener() {
        locationLayout.animate().alpha(1f).setDuration(150);
        manager = (LocationManager) getSystemService(LOCATION_SERVICE);
        locationListener = new LocationListener() {
            @Override
            public void onLocationChanged(Location location) {

                //Displays output
                DecimalFormat formatter = new DecimalFormat("####.##");

                userLatitude = String.valueOf(formatter.format(Double.valueOf(location.getLatitude())));
                userLongitude = String.valueOf(formatter.format(Double.valueOf(location.getLongitude())));

                latitudeTV.setText(userLatitude);
                longitudeTV.setText(userLongitude);

            }

            @Override
            public void onStatusChanged(String provider, int status, Bundle extras) {

            }

            @Override
            public void onProviderEnabled(String provider) {

            }

            @Override
            public void onProviderDisabled(String provider) {

            }
        };

        //launch OrientationModelReaderAsyntask once Sensor data is collected
        OrientationModelReaderAsyntask runner = new OrientationModelReaderAsyntask();
        runner.execute(orientationArray);

        if (Build.VERSION.SDK_INT < 23) {// no permission check required
            Log.i(LOG_TAG, "No checkSelfPermission");
            Log.i(LOG_TAG, "requestLocationUpdates");
            manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, locationListener);

        } else {
            Log.i(LOG_TAG, "checkSelfPermission");
            //checks if permission granted
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) { // permission not granted
                //ask for permission
                Log.i(LOG_TAG, "PERMISSION NOT GRANTED, ASK FOR IT");
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, LOCATION_RQST);
            } else { //permission granted
                //ask for location
                Log.i(LOG_TAG, "PERMISSION GRANTED, ASK FOR LOCATION");
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) { // permission  granted
                    Log.i(LOG_TAG, "requestLocationUpdates");
                    manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, locationListener);
                }
            }
        }

    }

    /**
     * This method opens a galery explorer for local image selection.
     */
    public void selectImage() {

        locationLayout.animate().alpha(0f).setDuration(150);

        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/jpeg");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        startActivityForResult(Intent.createChooser(intent, "Complete action using"), SELEC_IMAGEN);


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {

            case SELEC_IMAGEN: //Select image button result

                if (resultCode == RESULT_OK) {

                    imagenUri = data.getData();
                    sampleIV.setImageURI(imagenUri);

                    try {
                        Bitmap imageBitmap = getBitmapFromUri(imagenUri);

                        try {
                            photoFile = createImageFile();

                        } catch (IOException ex) {
                            // Error occurred while creating the File
                            Log.e(LOG_TAG,"createImageFile: " + ex.toString());
                            return;
                        }

                        // Continue only if the File was successfully created
                        if (photoFile != null) {

                            OutputStream os;
                            try {
                                os = new FileOutputStream(photoFile);
                                imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, os);
                                os.flush();
                                os.close();

                                Log.i(LOG_TAG,"execute ModelReaderAsyntask");

                                //launch the asyncTask now that the image has been saved
                                ModelReaderAsyntask runner = new ModelReaderAsyntask();
                                runner.execute(photoFile.getPath());


                            } catch (Exception e) {
                                Log.e(getClass().getSimpleName(), "Error writing bitmap", e);
                            }

                        }

                    } catch (IOException e) {
                        Log.e(LOG_TAG,"getBitmapFromUri: " + e.toString());
                    }

                }

                break;


            case OPEN_CAMERA_RQST: //Take picture button result

                Log.i(LOG_TAG,"OPEN_CAMERA_RQST ");

                if (resultCode == RESULT_OK) {

                    Log.i(LOG_TAG,"OPEN_CAMERA_RQST RESULT_OK ");

                    Bundle extras = data.getExtras();
                    Bitmap imageBitmap = (Bitmap) extras.get("data");
                    sampleIV.setImageBitmap(imageBitmap);

                    try {
                        photoFile = createImageFile();

                    } catch (IOException ex) {
                        // Error occurred while creating the File
                        Log.e(LOG_TAG,"createImageFile: " + ex.toString());
                        return;
                    }

                    // Continue only if the File was successfully created
                    if (photoFile != null) {

                        OutputStream os;
                        try {
                            os = new FileOutputStream(photoFile);
                            imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, os);
                            os.flush();
                            os.close();

                            Log.i(LOG_TAG,"execute ModelReaderAsyntask");

                            //launch the asyncTask now that the image has been saved
                            ModelReaderAsyntask runner = new ModelReaderAsyntask();
                            runner.execute(photoFile.getPath());


                        } catch (Exception e) {
                            Log.e(getClass().getSimpleName(), "Error writing bitmap", e);
                        }

                    }


                }

                break;


        }

    }

    /**
     * This method shows loader on screen.
     */
    public void onProgressBar() {
        outLayout.animate().scaleX(-1f).scaleY(-1f).setDuration(50);
        progressBar.animate().alpha(1).setDuration(150);

    }

    /**
     * This method hides loader from screen.
     */
    public void offProgressBar() {
        progressBar.animate().alpha(0).setDuration(150);
        outLayout.animate().scaleX(1f).scaleY(1f).setDuration(50);
    }

    private class ModelReaderAsyntask extends AsyncTask<String, Integer, String[]> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            onProgressBar();


        }

        @Override
        protected String[] doInBackground(String... uri) {

            String[] testResult = new String[2];
            // Main background thread, this will load the model and test the input image
            //load the model from the raw folder with a try / catch block
            try {

                // Load the pretrained network.
                InputStream inputStream = getResources().openRawResource(R.raw.relu_mse_e14_10222019235106_model);
                net = ModelSerializer.restoreMultiLayerNetwork(inputStream);
                initSampleInputParams();
                testResult = testImage(photoFile);

            } catch (MalformedURLException e) {
                Log.e(LOG_TAG, "imageStream MalformedURLException : " + e.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }

            return testResult;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }


        @Override
        protected void onPostExecute(String[] result) {
            super.onPostExecute(result);

            if(result[0].equals(classLabels[0])){
                //MAIZE
                outLayout.setBackground(getResources().getDrawable(R.drawable.maize_output_background,null));
                outputIcon.setImageResource(R.drawable.corn_ic);
            }else{
                //WHEAT
                outLayout.setBackground(getResources().getDrawable(R.drawable.wheat_output_background,null));
                outputIcon.setImageResource(R.drawable.wheat_ic);
            }

            //Displays output
            DecimalFormat formatter = new DecimalFormat("####.##");
            String labelText = String.format("Prediction: %s Probability: %s %%", result[0], formatter.format(Double.valueOf(result[1])));
            Log.i(LOG_TAG, "\n\noutput : " + labelText);
            confTv.setText(String.format("%s %%", formatter.format(Double.valueOf(result[1]))));
            outLayout.animate()
                    .alpha(1)
                    .scaleX(1.1f)
                    .scaleY(1.1f)
                    .setDuration(350).setListener(new Animator.AnimatorListener() {
                @Override
                public void onAnimationStart(Animator animation) {

                }

                @Override
                public void onAnimationEnd(Animator animation) {
                    outLayout.animate().scaleX(0.9f).scaleY(0.9f).setDuration(350);
                }

                @Override
                public void onAnimationCancel(Animator animation) {

                }

                @Override
                public void onAnimationRepeat(Animator animation) {

                }
            });

            offProgressBar();

        }

    }

    /**
     *This method retrieves image scaler file from the raw folder.
     */
    public DataNormalization getTrainImageScaler() throws IOException {

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                ||  ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(CropRecAppActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, WRITE_PERMISSION_RQST);

        }else{

            return getRawImageScaler();

        }

        return null;
    }

    /**
     *This method retrieves image scaler file from the raw folder.
     */
    private DataNormalization getRawImageScaler() throws  FileNotFoundException {

        InputStream inputStream = getResources().openRawResource(R.raw.image_scaler);
        OutputStream outputStream = new FileOutputStream(new File(getDataDir() + "/output"));

        int length;
        byte[] bytes = new byte[1024];

        try{
            // copy data from input stream to output stream
            while ((length = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, length);
            }

            FileInputStream fis = new FileInputStream(new File(getDataDir() + "/output"));
            // To read the Book object use the ObjectInputStream.readObject() method.
            // This method return Object type data so we need to cast it back the its
            // origin class, the Book class.

            ObjectInputStream ois = new ObjectInputStream(fis);
            DataNormalization scaler = (DataNormalization) ois.readObject();

            return scaler;

        }catch (IOException | ClassNotFoundException e) {
            Log.e(LOG_TAG,"getTrainImageScaler method: " + e.toString() );
            Log.e(LOG_TAG,"getTrainImageScaler method: " + e.toString() );
        }


        return null;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (grantResults.length > 0 && grantResults != null) {

            Log.i(LOG_TAG, "valid grantResults");

            switch (requestCode) {

                case WRITE_PERMISSION_RQST:

                    Log.i(LOG_TAG, "NO_PERMISSIONS_GRANTED");

                    if(grantResults[0] == PackageManager.PERMISSION_GRANTED){

                        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

                            Log.i(LOG_TAG, "All permissions granted" );

                        }else{
                            Toast.makeText(getApplicationContext(),"Write/Read external storage permission necessary.",Toast.LENGTH_SHORT).show();
                        }

                    }

                    break;

                case TAKE_PIC_WRITE_PERMISSION_RQST:

                    Log.i(LOG_TAG, "TAKE_PIC_WRITE_PERMISSION_RQST");

                    if(grantResults[0] == PackageManager.PERMISSION_GRANTED){

                        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

                            Log.i(LOG_TAG, "startTakePictureIntent" );
                            startTakePictureIntent();


                        }else{
                            Toast.makeText(getApplicationContext(),"Write external storage permission necessary.",Toast.LENGTH_SHORT).show();
                        }

                    }

                    break;






                case CAMERA_PERMISSION_RQST:

                    Log.i(LOG_TAG, "CAMERA_RQST");


                    if(grantResults[0] == PackageManager.PERMISSION_GRANTED){

                        if (ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA) == PackageManager.PERMISSION_GRANTED)  {

                            Log.i(LOG_TAG, "CAMERA_RQST PERMISSION_GRANTED" );

                            if(getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)){

                                Log.i(LOG_TAG, "hasSystemFeature");
                                takePicture();
                            }else{
                                Toast.makeText(getApplicationContext(),"No camera available.",Toast.LENGTH_SHORT).show();
                            }



                        }

                    }else{

                        Toast.makeText(getApplicationContext(),"Camera permissions necessary.",Toast.LENGTH_SHORT).show();

                    }

                    break;

                case LOCATION_RQST:

                    if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {//permission granted
                        Log.i(LOG_TAG, "PERMISSION_GRANTED"); //ask for location

                        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) { // permission  granted
                            Log.i(LOG_TAG, "requestLocationUpdates");
                            manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, locationListener);
                        }

                    } else {//permission not granted
                        Toast.makeText(this, "Location Permission Required", Toast.LENGTH_LONG).show();
                    }


                    break;

            }


        }

    }

    /**
     * if there is no SD card, create new directory objects to make directory on device
     */
    private File getOutputDirectory(String folderName) {

        Log.v(LOG_TAG, "getOutputDirectory");

        File directory = null;

        if (Environment.getExternalStorageState() == null) {
            //create new file directory object

            Log.v(LOG_TAG, "getExternalStorageState() == null");

            directory = new File(Environment.getDataDirectory()
                    + "/"+folderName+"/");

            Log.v(LOG_TAG, "directory path: " + Environment.getDataDirectory()
                    + "/"+folderName+"/");

            // if no directory exists, create new directory
            if (!directory.exists()) {
                Log.v(LOG_TAG, "directory dont exist");
                directory.mkdir();
            }

            // if phone DOES have sd card
        } else if (Environment.getExternalStorageState() != null) {

            Log.v(LOG_TAG, "getExternalStorageState() != null");


            // search for directory on SD card
            directory = new File(Environment.getExternalStorageDirectory()
                    + "/"+folderName+"/");
            // if no directory exists, create new directory
            if (!directory.exists()) {
                Log.v(LOG_TAG, "directory dont exist");
                directory.mkdir();
            }

        }// end of SD card checking

        return directory;

    }
    
    private File createImageFile() throws IOException {

        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + ".jpg";

        File storageDir = getOutputDirectory("CropRecApp");


        return new File(storageDir,imageFileName);
    }

    private void dispatchTakePictureIntent() {


        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(CropRecAppActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, TAKE_PIC_WRITE_PERMISSION_RQST);

        }else{
            startTakePictureIntent();
        }


    }

    private void startTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {

            startActivityForResult(takePictureIntent, OPEN_CAMERA_RQST);

        }
    }

    private Bitmap getBitmapFromUri(Uri uri) throws IOException {
        ParcelFileDescriptor parcelFileDescriptor =
                getContentResolver().openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();
        return image;
    }


    private class OrientationModelReaderAsyntask extends AsyncTask<double[][], Integer, String> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

        }

        @Override
        protected String doInBackground(double[][]... samples) {

            double[][] sample = samples[0];

            try {
                // Load the pretrained network.
                InputStream inputStream = getResources().openRawResource(R.raw.orientation_model);
                MultiLayerNetwork orientationNet = ModelSerializer.restoreMultiLayerNetwork(inputStream);
                Log.i(LOG_TAG, "orientationNet restore Done");
                INDArray matrix = Nd4j.create(sample);

                DataNormalization normalizer = getOrientationNormalizer();
                normalizer.transform(matrix);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

                int[] out = orientationNet.predict(matrix);
                Log.i(LOG_TAG, "predict: " + out[0]);

                return getOrientationLabel(out[0]);

            } catch (IOException e) {
                Log.e(LOG_TAG,  "ModelSerializer error: " + e.toString());
            }

            return null;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }


        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            Log.i(LOG_TAG,"OrientationModelReaderAsyntask onPostExecute result : " + result);

            orientationTextView.setText(result);

            //Displays output
           // oreintationTV.setText(result);

        }

    }

    private DataNormalization getOrientationNormalizer() throws  FileNotFoundException {

        InputStream inputStream = getResources().openRawResource(R.raw.orientation_normalizer);
        OutputStream outputStream = new FileOutputStream(new File(getDataDir() + "/orientation_output"));

        int length;
        byte[] bytes = new byte[1024];

        try{
            // copy data from input stream to output stream
            while ((length = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, length);
            }

            FileInputStream fis = new FileInputStream(new File(getDataDir() + "/orientation_output"));
            // To read the Book object use the ObjectInputStream.readObject() method.
            // This method return Object type data so we need to cast it back the its
            // origin class, the Book class.

            ObjectInputStream ois = new ObjectInputStream(fis);
            DataNormalization scaler = (DataNormalization) ois.readObject();

            return scaler;

        }catch (IOException | ClassNotFoundException e) {
            Log.e(LOG_TAG,"getNormalizer method: " + e.toString() );
        }


        return null;
    }

    private String getOrientationLabel(int orientationValue) {

        switch (orientationValue){
            case 0://NORTH
                return "N";
            case 1://NORTHEAST
                return "NE";
            case 2://EAST
                return "E";
            case 3://SOUTHEAST
                return "SE";
            case 4://WEST
                return "W";
            case 5://SOUTHWEST
                return "SW";
            case 6://NORTHWEST
                return "NW";
            case 7://SOUTH
                return "S";
        }

        return null;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
        // You must implement this callback in your code.
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Get updates from the accelerometer and magnetometer at a constant rate.
        // To make batch operations more efficient and reduce power consumption,
        // provide support for delaying updates to the application.
        //
        // In this example, the sensor reporting delay is small enough such that
        // the application receives an update before the system checks the sensor
        // readings again.

        Sensor accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if (accelerometer != null) {
            sensorManager.registerListener(CropRecAppActivity.this, accelerometer,
                    SensorManager.SENSOR_DELAY_NORMAL, SensorManager.SENSOR_DELAY_UI);
        }
        Sensor magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        if (magneticField != null) {
            sensorManager.registerListener(CropRecAppActivity.this, magneticField,
                    SensorManager.SENSOR_DELAY_NORMAL, SensorManager.SENSOR_DELAY_UI);
        }


    }

    @Override
    protected void onPause() {
        super.onPause();

        // Don't receive any more updates from either sensor.
        sensorManager.unregisterListener(this);

    }


    // Get readings from accelerometer and magnetometer. To simplify calculations,
    // consider storing these readings as unit vectors.
    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            //  Log.i(LOG_TAG, "TYPE_ACCELEROMETER onSensorChanged");
            System.arraycopy(event.values, 0, accelerometerReading,
                    0, accelerometerReading.length);

            updateOrientationAngles();

        } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            //   Log.i(LOG_TAG, "TYPE_ACCELEROMETER TYPE_MAGNETIC_FIELD");
            System.arraycopy(event.values, 0, magnetometerReading,
                    0, magnetometerReading.length);

            updateOrientationAngles();
        }

    }

    // Compute the three orientation angles based on the most recent readings from
    // the device's accelerometer and magnetometer.
    public void updateOrientationAngles() {

        // Update rotation matrix, which is needed to update orientation angles.
        SensorManager.getRotationMatrix(rotationMatrix, null,
                accelerometerReading, magnetometerReading);

        // "mRotationMatrix" now has up-to-date information.
        SensorManager.getOrientation(rotationMatrix, mOrientationAngles);

        orientationArray[0][0] = (double) accelerometerReading[0];
        orientationArray[0][1] = (double) accelerometerReading[1];
        orientationArray[0][2] = (double) accelerometerReading[2];
        orientationArray[0][3] = (double) magnetometerReading[0];
        orientationArray[0][4] = (double) magnetometerReading[1];
        orientationArray[0][5] = (double) magnetometerReading[2];
        orientationArray[0][6] = (double) mOrientationAngles[0];
        orientationArray[0][7] = (double) mOrientationAngles[1];
        orientationArray[0][8] = (double) mOrientationAngles[2];
        orientationArray[0][9] = (double) rotationMatrix[0];
        orientationArray[0][10] = (double) rotationMatrix[1];
        orientationArray[0][11] = (double) rotationMatrix[2];
        orientationArray[0][12] = (double) rotationMatrix[3];
        orientationArray[0][13] = (double) rotationMatrix[4];
        orientationArray[0][14] = (double) rotationMatrix[5];
        orientationArray[0][15] = (double) rotationMatrix[6];
        orientationArray[0][16] = (double) rotationMatrix[7];
        orientationArray[0][17] = (double) rotationMatrix[8];

        Log.i(LOG_TAG, "SENSOR INFO UPDATED");



//        orientationArray = {{ccelerometerReading[0], accelerometerReading[1] , accelerometerReading[2] ,
//                magnetometerReading[0] , magnetometerReading[1] , magnetometerReading[2] ,
//                mOrientationAngles[0] , mOrientationAngles[1] , mOrientationAngles[2] ,
//                rotationMatrix[0] , rotationMatrix[1] , rotationMatrix[2] ,
//                rotationMatrix[3] , rotationMatrix[4] , rotationMatrix[5] ,
//                rotationMatrix[6] , rotationMatrix[7] , rotationMatrix[8]}};


    }

}

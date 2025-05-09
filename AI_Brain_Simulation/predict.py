import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import render_template
#pip install pillow
import io


class accient:
    #def __init__(self, filename):
    #    self.filename = filename

    def accientImage(self, image_file):

        self.image_file = image_file

        # load the model
        model = load_model('AI_Simulation.h5')
        #Save the file to ./uploads

        basepath = os.path.dirname(__file__) # - org
        #basepath = os.path.dirname('Last Try')

        file_path = os.path.join(basepath, 'uploads', secure_filename(self.image_file.filename)) # - org
        # secure_filename - Pass it a filename and it will return a secure version of it.
        # This filename can then safely be stored on a regular file system and passed to os.

        #file_path = os.path.dirname('Last Try/temp')

        self.image_file.save(file_path)  # save the image for further use - org

        test_image = image.load_img(file_path, target_size=(224, 224)) # should be same as given in the code for input
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)  # expand dimension - flattening it

        preds = model.predict(test_image)
        #print(preds)

        preds = np.argmax(preds,axis=1)  # The numpy. argmax() function returns indices of the max element of the array in a particular axis.
        #print(preds)

      
        if preds == 0:
            prediction = "Alzheimer's disease on the Stage of MildDemented" 
            recommendation = "Recommendation: Regular physical exercise, mental stimulation activities, and medication as prescribed."
            return prediction + '\n' + recommendation
        elif preds == 1:
            prediction = "Alzheimer's disease on the Stage of ModerateDemented"
            recommendation = "Recommendation: Professional care and support, medication adherence, and safety measures at home."
            return prediction + '\n' + recommendation
        elif preds == 2:
            prediction = "Alzheimer's disease on the Stage of NonDemented"
            recommendation = "Recommendation: Maintain a healthy lifestyle with balanced diet and regular check-ups."
            return prediction + '\n' + recommendation
        elif preds == 3:
            prediction = "Alzheimer's disease on the Stage of VeryMildDemented"
            recommendation = "Recommendation: Engage in brain exercises, social activities, and routine medical follow-ups."
            return prediction + '\n' + recommendation
        elif preds == 4:
            prediction = "Spiral-healthy Not parkinson's disease"
            recommendation = "Recommendation: Continue healthy lifestyle practices and regular medical check-ups." 
            return prediction + '\n' + recommendation
        elif preds == 5:
            prediction = "Spiral-parkinson Affected"
            recommendation = "Recommendation: Follow prescribed treatment plan, physical therapy, and support groups."
            return prediction + '\n' + recommendation
        elif preds == 6:
            prediction = "Wave-healthy Not parkinson's disease"
            recommendation = "Recommendation: Maintain physical activity and regular health monitoring."
            return prediction + '\n' + recommendation
        
        elif preds == 7:
            prediction = "Wave-parkinson Affected"
            recommendation = "Recommendation: Adhere to medication regimen, occupational therapy, and consult healthcare providers regularly."
            return prediction + '\n' + recommendation
            
        
       

       
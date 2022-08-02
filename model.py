from importlib.resources import contents
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
from tqdm import tqdm

TRAINING_ENDPOINT = "https://electrifymodel.cognitiveservices.azure.com/"
training_key = "59437711c3f1481389dd7cef664d94d7"
PREDICTION_ENDPOINT = "https://electrifymodel-prediction.cognitiveservices.azure.com/"
prediction_key = "73f8bbe6f79f465a94808162b678a017"
prediction_resource_id = "/subscriptions/ae4c6595-9114-443a-a5a8-4b8d20ee17b0/resourceGroups/Electrify-Model/providers/Microsoft.CognitiveServices/accounts/ElectrifyModel-Prediction"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

# Create a new project
print ("Creating project...")
project_name = "Electrify Backend"
project = trainer.create_project(project_name)

base_image_location = os.path.join(os.path.dirname(__file__), "Images")

print("Adding images...")

image_list = []
folder_list = os.listdir(base_image_location)
resistors = dict.fromkeys(folder_list)
for name in resistors.keys():
    resistors[name] = trainer.create_tag(project.id, name)

# Go through every folder in images and add every image in those folders to image list

for folder in folder_list:
    num_files = len(os.listdir(os.path.join(base_image_location, folder)))
    x = 0
    # print("Opening folder: {}. Contains {} files".format(folder, num_files))
    for image_num in range(1, num_files):
        file_name = "{}_({}).jpg".format(folder, image_num)
        if os.path.isfile(os.path.join(base_image_location, folder, file_name)):
            x += 1
            with open(os.path.join (base_image_location, folder, file_name), "rb") as image_contents:
                image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[resistors[folder].id]))
    print("{} images from {} added successfully".format(x, folder))    

# separate list into 64 image chunks and upload

chunks = [image_list[x:x+64] for x in range(0, len(image_list), 64)]
for chunk in tqdm(chunks):
    upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=chunk))
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            if (image.status == "OKDuplicate"):
                print(image)
        exit(-1)

if (input("Images uploaded successfully, continue? (Y/n)") == "n"):
    trainer.delete_project(project.id)
    quit()

# print ("Training...")
# iteration = trainer.train_project(project.id)
# with tqdm(total=100) as pbar:
#     while (iteration.status != "Completed"):
#         iteration = trainer.get_iteration(project.id, iteration.id)
#         # print ("Training status: " + iteration.status)
#         # print ("Waiting 10 seconds...")
#         time.sleep(10)

# # The iteration is now trained. Publish it to the project endpoint
# trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
# print ("Done!")

# # Now there is a trained endpoint that can be used to make a prediction
# prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
# predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

# with open(os.path.join (base_image_location, "Test/220K_1-4W.jpg"), "rb") as image_contents:
#     results = predictor.classify_image(
#         project.id, publish_iteration_name, image_contents.read())

#     # Display the results.
#     for prediction in results.predictions:
#         print("\t" + prediction.tag_name +
#               ": {0:.2f}%".format(prediction.probability * 100))



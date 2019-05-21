import cv2
import os
import goClassifier

# Save path to read and write image.
#inputImagePath = os.path.abspath("./src/inputImage/")
outputImagePath = os.path.abspath("./src/outputImage/")

# Save path directory.
inputFileDir = os.path.abspath("./src/inputImage/")
inputFileDirList = os.listdir(inputFileDir)
inputFileDirList.sort()

print(inputFileDirList)

for imageName in inputFileDirList:
    # Read Image.
    if str(imageName) == ".keep":
        continue
        
    originalImage = cv2.imread(str(inputFileDir) + "/" + str(imageName))
    originalHeight, originalWidth, originalChanels = originalImage.shape
    originalHeight += 0.1
    originalWidth += 0.1

    goClassifier.processingImage(originalImage, goClassifier.preprocessingImage(originalImage), originalHeight, originalWidth, outputImagePath)
import os
_src = "D:/Onedrive/OneDrive - Hanoi University of Science and Technology/Projects/Python/FaceRecognize/training_data/images/e12d6826-ba45-11eb-8601-fc15b400cacf/"
# _src = "D:/Data/Downloads/output/"
_ext = ".jpg"
for i,filename in enumerate(os.listdir(_src)):
    if filename.endswith(_ext):
        os.rename(_src+ filename, _src+ str(i)+ _ext)
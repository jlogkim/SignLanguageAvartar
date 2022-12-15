# SignLanguageAvartar

Hi, this is github page for 2022 snu nlp class's final project.

Our project goal:
Input: Sentence
Output: 3D Avartar with pose sequence of sign language

To run py file, you should download other git repository under here.
## For Text to Pose
1. I used code and basic architecture from Progressive Transformers SLP.
  clone the github below
  https://github.com/BenSaunders27/ProgressiveTransformersSLP

2. We used HOW2SIGN Dataset 
  https://how2sign.github.io/
  You can easily download dataset by download_how2sign.sh file  
  I downloaded by "rgb_front_2D_keypoints" and "english_translation_re-aligned" options


Additionaly I recommend https://github.com/leeamy1203/dlf2020 page to other model using and overall pipeline.


## Pipeline
1. ``` python make_datafile.py ```
2. ``` python train.py ```
3. ``` python check_output.py ``` -> to make json file

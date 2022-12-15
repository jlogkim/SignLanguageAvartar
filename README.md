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

## For Pose to Avatar
1. I used code and basic architecture from SMPLify, SMPLpix.
  clone the github below
  https://github.com/sergeyprokudin/smplpix
  https://github.com/vchoutas/smplify-x

2. Please follow the instructions for SMPLpix and SMPLify

  for SMPLify, the command is as follows:

  ``` python smplifyx/main.py --config cfg_files/fit_smplx.yaml \
    --data_folder  ./data \
    --output_folder ./data/smplifyx_results \
    --visualize="True" \
    --gender="male" \
    --model_folder ./models \
    --vposer_ckpt ./vposer_v1_0 \
    --part_segm_fn ./smplx_parts_segm.pkl ```
  
3. for SMPLpix, the command is as follows:
    ```python smplpix/eval.py --workdir ./content/smplpix_logs/ --data_dir ./content/smplpix_logs/smplpix_data/test```
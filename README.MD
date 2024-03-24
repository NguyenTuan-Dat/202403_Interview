## Install Environment

```
conda create -n imageprocess python=3.9
conda activate imageprocess
pip3 install -r requirements.txt
```

## Run E1

```
python3 E1.py -i [INPUT_IMAGE_PATH] -o [OUTPUT_IMAGE_PATH]
```

Input image (left) and output image (right) \
![E1.png](image_for_readme%2FE1.png)

### Explain:

**Step1:** Using object detection model to detect objects in image \
![E1_step1.png](image_for_readme%2FE1_step1.png) \
**Step2:** Select points in image based on bboxes \
![E1_step2.png](image_for_readme%2FE1_step2.png) \
**Step3:** Using Segment Anything Model(SAM) and points in **Step2** to get mask \
![E1_step3.jpg](image_for_readme%2FE1_step3.jpg) \
**Step4:** Using mask to filter raw image \
![E1_step4.png](image_for_readme%2FE1_step4.png)

## Run E2

### Clone Stable Diffusion Openvino Project

```
git clone https://github.com/bes-dev/stable_diffusion.openvino.git
```

### Run

```
python3 E2.py -i [INPUT_IMAGE_PATH] -o [OUTPUT_PATH] -p [PROMPT] -s [STENGTH]
```

Input image (left) and output image (right) create by `promt: "snow, Christmas tree"` and `strength 0.7` \
![E2.png](image_for_readme%2FE2.png) \
**Step1:** Using mask in **Step3 of E1** to create mask of the region to inpaint on the initial image \
![E2_step1.png](image_for_readme%2FE2_step1.png) \
**Step2:** Using diffusion model to create new background \
![E2_step2.png](image_for_readme%2FE2_step2.png)





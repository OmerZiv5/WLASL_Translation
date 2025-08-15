## **ASL2TEXT**

This project is an end‑to‑end pipeline to translate isolated American Sign Language (ASL) words into English labels using the WLASL dataset. This repository contains three Python entry points:

GetFrames.py – converts raw WLASL videos into fixed‑length frame samples (plus optional frame‑difference images) in a class‑organized folder tree.

IsoNetwork.py – trains a ResNet18+GRU classifier on pre‑split datasets (e.g., Train/Test frame folders you prepared).

NonIsoNetwork.py – performs an internal stratified split (train/test) from a single frames root and trains the same model.

ASL2TEXT_Project_Report.pdf - the accompanying project report, detailing our process and considerations.

The core idea: the project takes the raw WLASL videos, extracts 5 frames out of each video in equal intervals and repeats this process with a random small offset several times to create a larger
dataset out of the available videos.
Then the model (being either the non-isolated model or the isolated one) take the relevant folders and trains on them to teach itself how to classify the frames (extracted from the original
videos) into the correct word that is being signed.

How to use the project:
1. Decide if you want to use a non-isolated dataset or an isolated one. The project can run in 2 ways, either you give it a folder of videos and you let the code devide the videos into train and
test sets randomly (from now on we will refer to this option as "non-isolated") or you make the devision yourself (for example the test set can be the videos of one specific signer)
in which case the project expects 2 seperate folders, one for training videos and the other for the test videos (from now on we will refer to this option as "isolated").
2. Run the GetFrames file. If you chose "non-isolated" in step 1 then change in lines 11-13 the path to your input folder (containing your videos) and chose a location where the output folder and
differences folder would be saved.
if you chose "isolated" you need to run this file twice, once for your training videos and once for your test videos and you have to select where will each of the outputs will be saved.
3. Run the network file. If you chose "non-isolated" run the file called "NonIsoNetwork.py", this file expects all of the folders from the previous step to be saved in the save folder as the file
itself so make sure that this is indeed the case. You need to change in line 78 the parameter called "root_dir" (the first param of the function) to have the name of the folder that contains the
frames you want to train on.
If you chose "isolated" run the file called "IsoNetwork.py", to run this file you need to change the paths in lines 88-89 to be the path to your train frames in line 88 and test frames in line 89
the path can be any path in your computer.


### **Hyper parameters:**<br/>
Learning rate: 10^-4 <br/>
Optimizer: Adam <br/>
Number of frames per video: 5 <br/>
Range of frame sampling: ±5 <br/>
Color jittering parameters: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 <br/>
Resizing new size: 224, 224 <br/>
Normalization parameters: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] <br/>
Number of repeats: 25 (meaning how many times we are sampling 5 slightly different frames from each video) <br/>

### **Credits:**<br/>
Omer Ziv - omer-ziv@campus.technion.ac.il <br/>
Raviv Segal raviv.s@campus.technion.ac.il <br/>

August 2025

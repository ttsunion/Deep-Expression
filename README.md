# Deep-Express
An Attention Based Open-Source End to End Speech Synthesis Framework, No CNN, No RNN, No MFCC!!!

Till now, all piplines in speech synthesis area is not really end to end. No mattter Deep voice or Tacotron claimed by baidu or google company, and so forth.                                                                                                                      

Because none of them gave up traditional audio preprocessing, like MFCC. But I was always wandering why can't we get kernals of MFCC through backpropagation?                                                                                                           

Therefore, I wanna to open up Deep Express framework, to synthesis audio signals from text directly.  

In previous frameworks, people tended to normalized wave data, and may eventually loss of sound rhythm. In Deep Express, I am planning to training my model using 16 bit intergers directly.

This project is under revisement...............

# Step1
python preprocess.py

# Step2
python train.py

# Progress
It's works!!! But, some noise still exist, and I'm try my best to improve the performance of Deep-Express.

# Meanings of this project
1. First time to use DNN-based model to synthesis speech
2. First time to use chars-signals end to end method, also first time not to use mfcc in speech synthesis piplines.
3. First time to sythesis very good speech not to hurt the sound rhythm.
4. First end to end speech synthesis framework that don't need post process.
5. Till now, Deep-Express is the most fast end-to-end model in speech synthesis area.
6. A new algorithm (weight-share DNN) was introduced in this project. 

# How to contact me?
If you have any questions or suggestions or ideas about Deep-Express, please let me know. Besides, if Deep-Express could help you in some cases, leave a star.

My Email:

fangshuming519@gmail.com

fangshuming@gig.ac.cn

# Citation
If you publish work based on Deep-Express, please cite:

https://github.com/ttsunion/Deep-Express

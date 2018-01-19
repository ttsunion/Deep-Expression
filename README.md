# Deep Express
An Attention Based Open-Source End to End Speech Synthesis Framework, No CNN, No RNN, No MFCC!!!

Till now, all piplines in speech synthesis area are not really end to end. No mattter Deep voice or Tacotron claimed by baidu or google company, and so forth.                                                                                                                      

Because

1. None of them gave up traditional audio preprocessing, like MFCC. But I was always wandering why can't we get kernals of MFCC through backpropagation?                                   
2. All piplines in speech synthesis area before Deep-Express need lots of preprocessing in text encoding and speech preprocessing and post processing.

For instance, WaveNet (A. van den Oord et al., 2016, https://arxiv.org/pdf/1609.03499.pdf) require significant domain expertise to produce, involving elaborate text-analysis systems as well as a robust lexicon (Jonathan Shen et al., 2017, https://arxiv.org/pdf/1712.05884.pdf). Both Tacotron (Yuxuan Wang et al., 2017,  https://arxiv.org/pdf/1711.00520.pdf) and Tacotron 2 (Jonathan Shen et al., 2017, https://arxiv.org/pdf/1712.05884.pdf) and Deep Voice 3 (Wei Ping et al., 2017, https://arxiv.org/pdf/1710.07654.pdf) use vocoder (Griffin-Lim or WORLD or WaveNet algorithms) for final audio synthesis. 

Therefore, I wanna to open up Deep Express framework, to synthesis audio signals from text directly.  

In previous frameworks, people tended to normalized wave data, and may eventually loss of sound rhythm. In Deep Express, model was trained by using 16bit-interger signals directly.

# Step1
python preprocess.py

# Step2
python train.py

# Progress
It works well!!! However, some noises still exist, and I'm try my best to improve the performance of Deep-Express. This project is under revisement.

# Meanings of this project
1. First time to use DNN-based model to synthesis speech
2. First time to use chars-signals end to end method, also first time not to use mfcc in speech synthesis piplines.
3. First time to sythesis very good speech not to hurt the sound rhythm.
4. First end to end speech synthesis framework that don't need post process.
5. Till now, Deep Express is the most fast end-to-end model in speech synthesis area.
6. A new algorithm, weight-share DNN, (w = tf.tile(tf.truncated_normal((step_size, D), mean=0.0, stddev=1, dtype=tf.float32, seed=None), [int(D/step_size), 1], name = 'w')) was introduced in this project. 

# How to contact me?
If you have any questions or suggestions or ideas about Deep-Express, please let me know. Besides, if Deep-Express could help you in some cases, leave a star. Moreover, if you have job opportunity in speech synthesis in China.....

My Email:

fangshuming519@gmail.com

fangshuming@gig.ac.cn

# Citation
If you publish work based on Deep-Express, please cite:

https://github.com/ttsunion/Deep-Express

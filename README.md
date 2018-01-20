# Why Deep Express
An Attention Based Open-Source End to End Speech Synthesis Framework, No CNN, No RNN, No MFCC!!!

Before Deep Express, none of piplines in speech synthesis area is really end to end solution. No mattter Deep voice or Tacotron claimed by baidu or google company, and so forth.                                                                                                                      

Because

1. None of them gave up traditional audio domain knowledges, like voice framing, STFT, MFCC, etc.. But I was always wandering for example, why can't we get kernals of MFCC through backpropagation?                                   
2. All piplines in speech synthesis area before Deep-Express need lots of preprocessing in text encoding and speech preprocessing and post processing.

For instance, WaveNet (Aaron van den Oord et al., 2016) require significant domain expertise to produce, involving elaborate text-analysis systems as well as a robust lexicon (Jonathan Shen et al., 2017). Both Tacotron (Yuxuan Wang et al., 2017) and Tacotron 2 (Jonathan Shen et al., 2017) and Deep Voice 3 (Wei Ping et al., 2017) use vocoder (Griffin-Lim or WORLD or WaveNet algorithms) for final audio synthesis. 

Therefore, I wanna to open up Deep Express framework, to synthesis audio signals from text directly.  

In previous frameworks (Aaron van den Oord et al., 2016; Jonathan Shen et al., 2017; Yuxuan Wang et al., 2017; Wei Ping et al., 2017), people tended to normalized audio data, and may eventually loss of sound rhythm. Even though they claimed that they synthesized natural human voice, synthesized audios from their systems, there is still a gap with real vocals. In Deep Express, model was trained by using 16bit-interger signals directly to synthesize amazing real human voice.

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
If you have any questions or suggestions or ideas about Deep-Express, please feel free to contact me. Besides, if Deep-Express could help you in some cases, leave a star. 

My Email:

fangshuming519@gmail.com

fangshuming@gig.ac.cn

# Citation
If you publish work based on Deep Express, please cite:

https://github.com/ttsunion/Deep-Express

# Acknowledgement
Layer-normalization and positional encoding function were copied from Kyubyong directly (https://github.com/Kyubyong/transformer). The remaining codes were all hard-coded myself.

# References
1 Aaron van den Oord et al., 2016, WAVENET: A GENERATIVE MODEL FOR RAW AUDIO, https://arxiv.org/pdf/1609.03499.pdf

2 Jonathan Shen et al., 2017, NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM
PREDICTIONS, https://arxiv.org/pdf/1712.05884.pdf

3 Yuxuan Wang et al., 2017, TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS, https://arxiv.org/pdf/1703.10135.pdf

4 Wei Ping et al., 2017, DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH, https://arxiv.org/pdf/1710.07654.pdf

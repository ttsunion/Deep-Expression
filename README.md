# Deep-Express
An Attention Based Open-Source End to End Speech Synthesis Framework, No CNN, No RNN, No MFCC!!!

Till now, all piplines in speech synthesis area is not really end to end. No mattter Deep voice or Tacotron claimed by baidu or google company, and so forth.                                                                                                                      

Because none of them gave up traditional audio preprocessing, like MFCC. But I was always wandering why can't we get kernals of MFCC through backpropagation?                                                                                                           

Therefore, I wanna to open up Deep Express framework, to synthesis audio signals from text directly.  

In previous frameworks, people tended to normalized wave data, and may eventually loss of sound rhythm. In Deep Express, I am planning to training my model using 16 bit intergers directly.

This project is under development...............

# Step1
python preprocess.py

# Step2
python train.py

# Progress
The code should work now, after enough training, loss can reach enough lower value (from ~2.3 to ~0.01). Besides, signals is transformed in order to locating in the range of (-1, 1) by dividing 2**15. Next step, is to write scipts for wave synthesis from transformed signals.

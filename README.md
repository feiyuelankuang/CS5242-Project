# CS5242 Project

## prerequisite

please download dataset to data folder before running the code.

also put .csv under data folder

note that training dataset should be under data/train/ instead of data/train/train/, same for testing data

## usage
python train.py --model 'your model' --lr 'your learning rate' --batch_size 'your batch size'

for new model, run python train_embedding --model 'MalwareNet' --lr 'your learning rate' --batch_size 'your batch size'

## related paper or blog

https://dzone.com/articles/malware-detection-with-convolutional-neural-networ

http://sei.pku.edu.cn/~yaoguo/papers/Kan-COMPSAC-18.pdf

https://medium.com/slalom-engineering/detecting-malicious-requests-with-keras-tensorflow-5d5db06b4f28

https://arxiv.org/pdf/1906.04593

https://www.covert.io/research-papers/deep-learning-security/Convolutional%20Neural%20Networks%20for%20Malware%20Classification.pdf

https://devblogs.nvidia.com/malware-detection-neural-networks/

https://www.scitepress.org/papers/2018/66858/66858.pdf

https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050918X00052/1-s2.0-S1877050918302965/main.pdf

https://www.evilsocket.net/2019/05/22/How-to-create-a-Malware-detection-system-with-Machine-Learning/

https://pdfs.semanticscholar.org/fdd6/b60a581ac9f08ad35f0017c5ae0dd4ee6d02.pdf

https://publications.waset.org/10005499/pdf

https://www.cse.msu.edu/~farazah/aisec-ccs09.pdf

https://link.springer.com/content/pdf/10.1007%2F978-981-13-6621-5_8.pdf

## related github sites (understand, not copy)

https://github.com/riak16/Malware-Detection-using-Deep-Learning

https://github.com/danhph/Cinder/tree/254c8b363f795e75d2313cc516274ae959049973

https://github.com/Patil-Kranti/Dynamic-Malware-Detection-Using-Machine-learning

https://github.com/Crystallee612/Malware-detection-based-on-API-sequences?files=1

## last semester/year's CS5242 project related to malware detection (understand, not copy)

https://github.com/SayHiRay/malware-detection

https://github.com/lylin17/malware_detection

https://github.com/lth08091998/CS5242-Project

## why batch size (and some neural network parameters) are preferably chosen to be a power of 2

https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2?rq=1

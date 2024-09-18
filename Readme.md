# Readme

To use dataset LatticeModulus, please unzip LatticeModulus.zip to [PATH\LatticeModulus], and load dataset by running:
~~~python
dataset = LatticeModulus('[your unzip path]\LatticeModulus', file_name='data')
~~~

To use dataset LatticeStiffness, please run:
~~~python
dataset = LatticeStiffness('[your path]\LatticeStiffness', file_name='training')
~~~
The dataset will be downloaded and processed automatically.


Training data:

~~~shell
python train.py --result_path [PATH] --data_path [PATH TO DATAROOT]
~~~

Generate Unit cell:

~~~shell
python generate1.py --model_path [PATH TO CHECKPOINT]  --save_mat_path [PATH TO SAVE RESULTS]
~~~





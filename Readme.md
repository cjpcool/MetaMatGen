# Readme
## Datasets preparation
To use dataset LatticeModulus, please unzip LatticeModulus.zip to [PATH\LatticeModulus], and load dataset by running:
~~~python
dataset = LatticeModulus('[your unzip path]\LatticeModulus', file_name='data')
~~~

To use dataset LatticeStiffness, please run:
~~~python
dataset = LatticeStiffness('[your path]\LatticeStiffness', file_name='training')
~~~
The dataset will be downloaded and processed automatically.

## Train
Training data:

~~~shell
python train.py --result_path [PATH] --data_path [PATH TO DATAROOT]
~~~
## Generation
Generate Unit cell:

~~~shell
python generate1.py --model_path [PATH TO CHECKPOINT]  --save_mat_path [PATH TO SAVE RESULTS]
~~~
**Conditional generation** process contains two steps:
1. Create conditioned properties saving in [cond_path]
2. Run generate1.py for generation. The generated node coordinates and edge_index will be saved in "[save_mat_path]/[dataset]_Lattice_[i].npz", and visualized .png image will be saved in [vis_save_root]/[save_mat_path]/[dataset]_Lattice_[i].png", where i is the indices of generated number.


A generate example:
~~~shell
python generate1.py --dataset LatticeModulus --cond_path ys.csv --num_gen 1 --use_gpu true --save_mat_path generated_mat/test --vis_save_root ./vis/
~~~







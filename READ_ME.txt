1. Dependencies
   1. Python 3.6.8
   2. Keras 2.2.4
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. PIL 6.0.0
1. Generate test backdoored data
   1. data/gen_backdoor.py: Change the poison_data function
      1. You can specify the target label.
   1. Execute the python script by running
python gen_backdoor.py '<trigger filename>' '<clean test data filename>'
   1. The poisoned data will be stored under data/bd_data directory.
1. Path to the backdoored model:
   1. Store your model in model/bd_net directory
1. Evaluate the backdoored model:
   1. eval.py: Change the data_preprocessing function and execute the script by running
python eval.py ‘<clean test data filename>’ ‘<backdoored test data filename>’ ‘<backdoored model filename>
   1. This will print the classification accuracy and attack success rate.
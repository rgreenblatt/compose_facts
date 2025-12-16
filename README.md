I've removed the composition dataset to reduce leakage, but you can regenerate it using `python3 create_compositional_dataset.py -n 300 && python3 create_compositional_dataset.py -n 300 --element-names`. PLEASE DON'T PUBLICLY POST THIS DATASET INCLUDING BY PUSHING IT TO GITHUB. (I also removed the correct answer in the "run_manifold_eval.py" file, you can manually edit this back in to run this file.)

See write_up.md for details.

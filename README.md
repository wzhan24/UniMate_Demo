# UniMate
## training
Run train.py to train.

## generating structure
We have already trained and saved a model, so the model can already generate structures.
"input_property.txt" indicates the intended properties for the structure. The properties has twelve dimensions, with the first three, the second three, and the last six indicating relative Young's Modulus, Shear Modulus, and Poisson's Ratio, respectively.
Run generate_structure.py to generate a structure.
The result will be shown in output_structure_0.txt. 

## predicting properties
Similar to structure generation, you can input the structure you have (a 3D graph) in "input_structure.txt". We provide a sample structure in the txt file already, for demonstration.
Run predict_properties.py to predict the properties of the structure. Similarly, the predicted properties has twelve dimensions as above.
The result will be shown in output_properties_0.txt.

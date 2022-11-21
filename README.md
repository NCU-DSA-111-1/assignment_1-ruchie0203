# Assignment_1
* Using Neural Network to learn XOR
---
## Compile & Run
```
# Compile
gcc main.c -lm layer.c neuron.c

# Run
./a.out

Enter the number of Layers in Neural Network: 4 
Enter number of neurons in layer[1]:  2 
Enter number of neurons in layer[2]:  4 
Enter number of neurons in layer[3]:  4 
Enter number of neurons in layer[4]:  1 

Created Layer: 1 
Number of Neurons in Layer 1: 2 
Neuron 1 in Layer 1 created 
Neuron 2 in Layer 1 created 

Created Layer: 2 
Number of Neurons in Layer 2: 4 
Neuron 1 in Layer 2 created 
Neuron 2 in Layer 2 created 
Neuron 3 in Layer 2 created 
Neuron 4 in Layer 2 created 

Created Layer: 3 
Number of Neurons in Layer 3: 4 
Neuron 1 in Layer 3 created 
Neuron 2 in Layer 3 created 
Neuron 3 in Layer 3 created 
Neuron 4 in Layer 3 created 

Created Layer: 4 
Number of Neurons in Layer 4: 1 
Neuron 1 in Layer 4 created 

Initializing weights... 
0:w[0][0]: 0.840188 
1:w[0][0]: 0.394383 
2:w[0][0]: 0.783099 
3:w[0][0]: 0.798440 
0:w[0][1]: 0.911647 
1:w[0][1]: 0.197551 
2:w[0][1]: 0.335223 
3:w[0][1]: 0.768230 
0:w[1][0]: 0.277775 
1:w[1][0]: 0.553970 
2:w[1][0]: 0.477397 
3:w[1][0]: 0.628871 
0:w[1][1]: 0.513401 
1:w[1][1]: 0.952230 
2:w[1][1]: 0.916195 
3:w[1][1]: 0.635712 
0:w[1][2]: 0.141603 
1:w[1][2]: 0.606969 
2:w[1][2]: 0.016301 
3:w[1][2]: 0.242887 
0:w[1][3]: 0.804177 
1:w[1][3]: 0.156679 
2:w[1][3]: 0.400944 
3:w[1][3]: 0.129790 
0:w[2][0]: 0.998924 
0:w[2][1]: 0.512932 
0:w[2][2]: 0.612640 
0:w[2][3]: 0.637552 

Enter the learning rate (Usually 0.15):  0.15 

Enter the number of training examples:  4 
Enter the Inputs for training example[0]:  0 0 
Enter the Inputs for training example[1]:  0 1 
Enter the Inputs for training example[2]:  1 0 
Enter the Inputs for training example[3]:  1 1 

Enter the Desired Outputs (Labels) for training example[0]:  0 
Enter the Desired Outputs (Labels) for training example[1]:  1 
Enter the Desired Outputs (Labels) for training example[2]:  1 
Enter the Desired Outputs (Labels) for training example[3]:  0 

Enter input to test: 0 0 
Output: 0 
Enter input to test: 0 1 
Output: 1 
Enter input to test: 1 0 
Output: 1 
Enter input to test: 1 1 
Output: 0 
Enter input to test: 2 
End the file! 

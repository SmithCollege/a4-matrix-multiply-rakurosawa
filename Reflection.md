## Assignment 4 Reflection

### Analyze your results, when does it make sense to use the various approaches?
According the the data, it essentially never makes sense to use the CPU version of matrix multiplication.  For every size of array tested, the CPU consistently had the worst run time by a large amount.  The GPU implementation of the data is better than the CPU, but is on average still worse than both the tiled and cuBLAS approaches.  The tiled approach follows along the path of the GPU for input sizes of 128 and 256.  However, once the array sizes reached 512 and greater, it was faster than the GPU implementation.

### How did your speed compare with cuBLAS?
The cuBLAS implementation appears to follow a linear progression as shown in the charts.  It becomes faster than all of the other implementations at an array size of 1024, but is slower before that.  However, it is not slower by too much and so it is the overall best option for matrix multiplicatoin as determined by the runtime.

### What went well with this assignment?
I think the implementation of the CPU and GPU versions went well once I understood the method to matrix multiplication.

### What was difficult?
It too me a while to understand what matrix multiplication was actually supposed to be doing.  Once I began to understand the method, it became much easiser.  I also struggled to understand the tiled implementation as well as how to implement the cuBLAS variation because I found the examples online overwhelming.

### How would you approach differently?
I don't really think I would do anything differently, I found this assignment to feel more managable and made sure to start earlier than normal so I wasn't as stressed when working on it.

### Anything else you want me to know?
I turned this in on time! :D 
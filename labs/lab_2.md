# 3.
In the arguments being passed into the quantize_transform_pass function, only the linear mase operator is being passed in. Therefore, when the pass is run, the linear operator is the only one being changed. And in the jsc-tiny model there is only one linear operator. 

NVCC=nvcc -w  
all: norm norm_opt1 norm_opt2 norm_opt3
norm: norm.cu
	$(NVCC) -o $@ $^

norm_opt1: norm_opt1.cu
	$(NVCC) -o $@ $^
	
norm_opt2: norm_opt2.cu
	$(NVCC) -o $@ $^
	
norm_opt3: norm_opt3.cu
	$(NVCC) -o $@ $^
	
clean:
	rm norm norm_opt1 norm_opt2 norm_opt3
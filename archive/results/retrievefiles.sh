rsync -av --include='*/' 
          --include='actuatorhist49999.txt'   
          --include='amplitudes.txt'   
          --include='losses.txt'   
          --exclude='*' 
          habrok:/scratch/p309238/archerfish/basisfunctions/ ./basisfunctions/
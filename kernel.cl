#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5
#define HBLKSZ 8
#define WBLKSZ 32

__kernel 
void get_bias(__global float C[NUM][IMROW][IMROW], 
              __global float bias[NUM])
{
        int idx = get_global_id(0); 
        int idy = get_global_id(1);
        int idz = get_global_id(2);

        C[idx][idy][idz] = bias[idx];
}                       


__kernel
void convolution(__global float C[NUM][IMROW][IMROW],
                 __global float weight[NUM][NUM][KERNEL][KERNEL],
                 __global float Cin[NUM][INIMROW][INIMROW])
{

        int i,j,w,p,q,h,jj;
        i = get_global_id(0);
        h = get_global_id(1);
        w = get_global_id(2);
       
        int i_l, h_l, w_l;
        i_l = get_local_id(0);
        h_l = get_local_id(1);
        w_l = get_local_id(2);


        __local float cin_buf[KERNEL+HBLKSZ][KERNEL+WBLKSZ];
	__local float weight_buf[KERNEL][KERNEL];

	float cihw = C[i][h][w];
	
 	for(j = 0; j < NUM; j++){

	    cin_buf[h_l][w_l] = Cin[j][h][w];
	    if(w_l > WBLKSZ-KERNEL){
	    	   cin_buf[h_l][w_l+4] = Cin[j][h][w+4];
       	    }
	    if(h_l > HBLKSZ-KERNEL){
	    	   cin_buf[h_l+4][w_l] = Cin[j][h+4][w];
	    }
	    if(w_l > WBLKSZ-KERNEL && h_l > HBLKSZ-KERNEL){
	    	   cin_buf[h_l+4][w_l+4] = Cin[j][h+4][w+4];
	    }

	    if(w_l < KERNEL && h_l < KERNEL)
	    	   weight_buf[h_l][w_l] = weight[i][j][h_l][w_l];

	    barrier( CLK_LOCAL_MEM_FENCE );

            for(p = 0; p < KERNEL; p++){
                  for(q = 0; q < KERNEL; q++){
		  	cihw += weight_buf[p][q] * cin_buf[h_l+p][w_l+q];
                  }
            }

	    barrier( CLK_LOCAL_MEM_FENCE );
	}
	
        C[i][h][w] = cihw;

}                       


__kernel
void ReLU(__global float C[NUM][IMROW][IMROW])
{
        int idx = get_global_id(0);
        int idy = get_global_id(1);
        int idz = get_global_id(2);

        C[idx][idy][idz] = 0 > C[idx][idy][idz] ? 0 : C[idx][idy][idz];

}

__kernel
void max_pooling(__global float C[NUM][IMROW][IMROW],
                __global float Cout[NUM][OUTIMROW][OUTIMROW])
{
        int i = get_global_id(0);
        int h = get_global_id(1);
        int w = get_global_id(2);

        float local_max = C[i][h*2][w*2];
        local_max = local_max > C[i][h*2 + 1][w*2] ? local_max : C[i][h*2+1][w*2];
        local_max = local_max > C[i][h*2 + 1][w*2 + 1] ? local_max : C[i][h*2+1][w*2+1];
        local_max = local_max > C[i][h*2][w*2 + 1] ? local_max : C[i][h*2][w*2+1];
        Cout[i][h][w] = local_max;

}                     
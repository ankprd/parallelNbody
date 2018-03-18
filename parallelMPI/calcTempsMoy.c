#include <stdio.h>
#include <stdlib.h>

int main(){
    double tCalc = 0;
    double tSend = 0;

    int j;
    for(j = 0; j < 8 * 16; j++){
        char nomF[4] = "i.t";
        nomF[0] = j + '0';
        //printf("%c", rank - '0');
        FILE* f_out3 = fopen(nomF, "r");
        double t1, t2;
        fscanf(f_out3, "%lf%lf", &t1, &t2);
        tCalc += t1;
        tSend += t2;
	    fclose(f_out3);
    }
    printf("%f %f", tCalc / (8. * 16.), tSend / (8. * 16.));
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>

#define GRID_SIZE 4
#define CMD_SIZE 255
#define PRM_SIZE 50
#define TRN_GROUPS 5

int main( int argc, char *argv[] )
{

  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  FILE *fp;
  char path[5021];
  char *s;
  int  p0,p1;
  float precision;
  //First iteration
  float arrC[GRID_SIZE] = {0.03,0.5,32,8192};
  float arrg[GRID_SIZE] = {0.00003,0.002,0.125,8};
  //Second iteration
  //float arrC[GRID_SIZE] = {30,40,60,100};
  //float arrg[GRID_SIZE] = {0.00003,0.002,0.125,8};
  //Third iteration
  //float arrC[GRID_SIZE] = {39,43,47,51};
  //float arrg[GRID_SIZE] = {0.00001,0.00005,0.0001,0.0005};
  //Fourth iteration
  //float arrC[GRID_SIZE] = {39,43,47,51};
  //float arrg[GRID_SIZE] = {0.0003,0.0006,0.0009,0.0012};
  char commad[CMD_SIZE];
  char parmtrs[PRM_SIZE];
  float better[GRID_SIZE] = {0,0,0,0};
  int betterIdx = 0;
  int idx,jdx;
  char *allocSet;
  //Training set
  int trainSet = world_rank/GRID_SIZE;
  /*
  if(world_rank < TRN_GROUPS)
  {
    trainSet = world_rank;
  }
  else
  {
    trainSet = world_rank%TRN_GROUPS;
  }
  */
  if(world_rank >= GRID_SIZE)
  {
    idx = world_rank%GRID_SIZE;
  }
  else
  {
    idx = world_rank;
  }
  
  for(jdx = 0; jdx < GRID_SIZE; jdx++)
  {
      // Training command
      // svm-train -g 0.0009 -c 45 set1aeio ../set1aeio.model
      memset(commad, 0, CMD_SIZE);
      strcpy(commad, "/home/mpiuser/libsvm-3.22/svm-train -g ");
      //Add gamma command parameteres
      memset(parmtrs, 0, PRM_SIZE);
      sprintf(parmtrs, "%g", arrg[idx]);
      strcat(commad, parmtrs);
      strcat(commad, " -c ");

      //Add C command parameteres
      memset(parmtrs, 0, PRM_SIZE);
      sprintf(parmtrs, "%g", arrC[jdx]);
      strcat(commad, parmtrs);
      strcat(commad, " ");
      switch(trainSet)
      {
        case 0:{strcat(commad, "set1aeio /home/mpiuser/set1aeio");}break;
        case 1:{strcat(commad, "set1aeiu /home/mpiuser/set1aeiu");}break;
        case 2:{strcat(commad, "set1aeou /home/mpiuser/set1aeou");}break;
        case 3:{strcat(commad, "set1aiou /home/mpiuser/set1aiou");}break;
        case 4:{strcat(commad, "set1eiou /home/mpiuser/set1eiou");}break;
        default:{strcat(commad, "set1aeio /home/mpiuser/set1aeio");}
      }	
      memset(parmtrs, 0, PRM_SIZE);
      sprintf(parmtrs, "%i", world_rank);
      strcat(commad, parmtrs);
      strcat(commad, ".model");
      printf("%s_%i>%s\n",processor_name,world_rank, commad);

      /* Open the command for reading.*/ 
      fp = popen(commad, "r");
      if (fp == NULL) {
        printf("%s_%i>Failed to run command\n",processor_name,world_rank );
        exit(1);
      }
      
      /* Read the output a line at a time - output it. */
      while (fgets(path, sizeof(path)-1, fp) != NULL) {
        //printf("%s_%i>%s", processor_name,world_rank,path);
      }
      
      /* close */
      pclose(fp);
      

      // Prediction command
      // svm-predict set1u ../set1aeio.model ../set1aeio.predict
      memset(commad, 0, CMD_SIZE);
      strcpy(commad, "/home/mpiuser/libsvm-3.22/svm-predict ");
      switch(trainSet)
      {
        case 0:{strcat(commad, "set1u /home/mpiuser/set1aeio");}break;
        case 1:{strcat(commad, "set1o /home/mpiuser/set1aeiu");}break;
        case 2:{strcat(commad, "set1i /home/mpiuser/set1aeou");}break;
        case 3:{strcat(commad, "set1e /home/mpiuser/set1aiou");}break;
        case 4:{strcat(commad, "set1a /home/mpiuser/set1eiou");}break;
        default:{strcat(commad, "set1u /home/mpiuser/set1aeio");}
      }
      memset(parmtrs, 0, PRM_SIZE);
      sprintf(parmtrs, "%i", world_rank);
      strcat(commad, parmtrs);
      strcat(commad, ".model ");
      switch(trainSet)
      {
        case 0:{strcat(commad, "/home/mpiuser/set1aeio");}break;
        case 1:{strcat(commad, "/home/mpiuser/set1aeiu");}break;
        case 2:{strcat(commad, "/home/mpiuser/set1aeou");}break;
        case 3:{strcat(commad, "/home/mpiuser/set1aiou");}break;
        case 4:{strcat(commad, "/home/mpiuser/set1eiou");}break;
        default:{strcat(commad, "/home/mpiuser/set1aeio");}
      }
      memset(parmtrs, 0, PRM_SIZE);
      sprintf(parmtrs, "%i", world_rank);
      strcat(commad, parmtrs);
      strcat(commad, ".predict");

      printf("%s_%i>%s\n", processor_name, world_rank,commad);
      /* Open the command for reading.*/ 
      fp = popen(commad, "r");
      if (fp == NULL) {
        printf("%s_%i>Failed to run command\n",processor_name ,world_rank);
        exit(1);
      } 

      /* Read the output a line at a time - output it.*/
      while (fgets(path, sizeof(path)-1, fp) != NULL) {
        printf("%s_%i>%s", processor_name, world_rank,path);
        s = strstr(path,"=");
        p0 = (s-path) + 2;
        s = strstr(path,"%");
        p1 = (s-path) - 1;
        precision = atof(&path[p0]);
        //printf("%s>Precision  %2.4f\n",processor_name, precision);
      }
      better[jdx] = precision;
      /* close */
      pclose(fp); 
  }
  
  switch(trainSet)
  {
    case 0:{allocSet = "set1u set1aeio";}break;
    case 1:{allocSet = "set1o set1aeiu";}break;
    case 2:{allocSet = "set1i set1aeou";}break;
    case 3:{allocSet = "set1e set1aiou";}break;
    case 4:{allocSet = "set1a set1eiou";}break;
    default:{allocSet = "set1u set1aeio";}
  } 
  MPI_Barrier(MPI_COMM_WORLD);
  // printf("%s_%i_%s_%i> Better___________________________%2.6f %2.6f %2.6f percent\n",processor_name,world_rank, allocSet, arrg[idx], better[0],better[1],better[2]);
  printf("%s_%i_%s_%3.6f> %2.6f %2.6f %2.6f %2.6f\n",processor_name,world_rank,allocSet,arrg[idx], better[0],better[1],better[2],better[3]);
  MPI_Finalize();


  return 0;
}
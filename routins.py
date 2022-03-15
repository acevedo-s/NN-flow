import matplotlib.pyplot as plt ; import numpy as np
import csv ; import h5py 
import os ; import sys ; import time
from tensorflow.keras import losses

def normalize(non_normalized_np_array):
  normalized_np_array=(non_normalized_np_array+1.)/2.
  return	normalized_np_array

def unnormalize(normalized_np_array):
  unnormalized_np_array=(normalized_np_array-1/2.)*2.
  return	unnormalized_np_array
 
def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

def mag(configuration,L):
  m=0
  for i in range(L**2):
      m+=(configuration[i]-1./2)*2
  return m/L/L

def mag_s(configuration,L):
  m=0
  for i in range(L**2):
      m+=(-1)**(i)*configuration[i]
  return 2*m/L/L

def Ener(data,L):
  data=(data-1/2)*2
  E=0
  for i in range(L):
    for j in range(L):
      E+=(-1)*(data[(i*L+j)%(L*L)] * data[((i+1)*L+j)%(L*L)] + data[(i*L+j)%(L*L)] * data[(i*L+(j+1))%(L*L)])
  return E/L/L
  
  
def compare(original,synthetic,L):
  """This routine compares a snapshot from the system with a synthetic snapshot
  original and synthetic are vectors of L*L components"""
  original=original.reshape(L,L)
  synthetic=synthetic.reshape(L,L)
  size_x=7;size_y=size_x
  plt.figure(figsize=(size_x, size_y), dpi=80)
  # display original
  ax = plt.subplot(1, 2, 1)
  plt.title("Original",size=16)
  plt.imshow(original[:,:],vmin=0,vmax=1)
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  # display reconstruction
  bx = plt.subplot(1, 2, 2)
  plt.title("Reconstructed",size=16)
  plt.imshow(synthetic[:,:], cmap='gray',vmin=0,vmax=1)
  plt.gray()
  bx.get_xaxis().set_visible(False)
  bx.get_yaxis().set_visible(False)
  plt.show()
  return



def NN_flow(initial_configuration,autoencoder,iterations,order_parameter,L,Energy,round_flag):
  """Flows takes a single initial configuration as input and feeds the trained
  autoencoder with it. Then takes the reconstruction, apply the round function 
  to get an Ising binary variable and feeds the autoencoder
  again with the resultin image. This is repeated 'iteration' times.
  Energy and magnetization are calculated in each step to monitor the NN-flow
  Format warning:
  The autoencoder takes as input a set of images. To be able to reconstruct a single image
  I feed the autoencoder with a list with only one element...
  """
  x=initial_configuration
  m_list=[];E_list=[]
  m_list.append(order_parameter(initial_configuration[0],L))
  E_list.append(Energy(initial_configuration[0],L))
  for i in range(iterations):
    x=autoencoder.encoder(x).numpy()
    x=autoencoder.decoder(x).numpy()
    if (round_flag):
      x=np.round(x)
    m_list.append(order_parameter(x[0,:],L)) 
    E_list.append(Energy(x[0,:],L)) 
  return x,np.array(m_list),np.array(E_list)

def flows(n_snapshot,n_realizations,iterations,n_T,autoencoder,data,temperatures,L,round_flag):
  """
  flows calculates the NN_flow of several configurations with the same temperature.
  n_snapshot is an index that indexes a configuration in data
  n_T is the number of different temperatures in data 
  it takes the index n_snapshot and moves n_T in the data list to compute 
  the flow for configurations of different realizations with the same temperature
  """
  t_i=time.time()
  print('initial_T=',temperatures[n_snapshot])
  final_configurations=[]
  mag_lists=[]
  E_lists=[]
  for i in range(n_realizations):
    index=n_snapshot+i*n_T
    initial_configuration=data[index:index+1]
    final_configuration,mag_list,E_list=NN_flow(initial_configuration,autoencoder,iterations,mag,L,Ener,round_flag)
    final_configurations.append(final_configuration)
    mag_lists.append(mag_list)
    E_lists.append(E_list)
  computing_time=round((time.time()-t_i)/60.,1)
  print("computing time: ",computing_time," minutes")
  return final_configurations,mag_lists,E_lists
  
def data_by_temperatures(data,n_T):
  """ data_by_temperatures organices data in n_T subsets, one for each temperature, and returns the np array data_T,
  """
  data_T=[]
  for i in range(n_T): #n_T subsets of data, one for each temperature
    data_T.append([])
  for i in range(len(data)):
    data_T[i%n_T].append(data[i])
  data_T=np.array(data_T)
  print("data_T shape is: ",np.shape(data_T))
  return data_T
 
def get_RE_vs_T(data_T,autoencoder,n_T,T_list,round_flag):
  t_i=time.time()#initial_time
  mse=losses.MeanSquaredError()
  reconstructions_T=np.array([autoencoder.predict(x) for x in data_T])
  if round_flag:
    reconstructions_T=np.round(reconstructions_T)
  RE_T=[]
  for i in range(n_T):
    RE_T.append([mse(data_T[i][j],reconstructions_T[i][j]).numpy() for j in range(len(data_T[i]))])
  print('RE_T shape:',np.shape(RE_T))
  #list to do scatter plot where each temperature is repeated n_data times:
  T_scatter_list=[]
  n_data=np.shape(data_T)[1]
  for i in range(n_T):
    T_scatter_list.append([T_list[i] for j in range(n_data)])
  computing_time=round((time.time()-t_i)/60.,1)
  print("computing time: ",computing_time," minutes")
  return T_scatter_list,RE_T
  
  
def get_x_star(path,mag_flag):
  """This routine returns the first (0) and the last (star) element in a AE-flow
   in mean value, together with the corresponding standard deviation """
  #if x=Energy, mag_flag must  be set to 0
  #if x=magnetization, mag_flag must be set to 1
  x_lists=np.load(path)
  if (mag_flag): 
    x_lists=np.absolute(x_lists)
  x_mean=np.mean(x_lists,axis=0)
  x_std=np.std(x_lists,axis=0)
  x_star=x_mean[-1]
  x_star_std=x_std[-1]
  x_0=x_mean[0]
  x_0_std=x_std[0]
  return x_star,x_star_std,x_0,x_0_std


  
  
  
  

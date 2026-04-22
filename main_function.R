

# calculate the number of responses

NumResponse <- function(Y){
  mv1 = numeric(dim(Y)[2])
  
  for(i in 1:length(mv1)){
    mv1[i] = max(Y[,i])+1
  }
  return(mv1)
}


ECDM_main <-function(Y,V,covarites,
                     K_a,K_g,
                     iteration){
  n  = nrow(Y)
  J_y = ncol(Y)
  J_v = ncol(V)
  
  M_y = NumResponse(Y)
  M_v = NumResponse(V)
  
  #setting design matrix
  A1=matrix(c(1),nrow=2^K_a,ncol=1)
  A0=matrix(c(0),nrow=2^K_a,ncol=K_a)
  
  for(i in 0:(2^K_a-1))
  { for(k in 1:K_a)
    if(intToBits(i)[k]==01) A0[i+1,k]=1}
  
  A = cbind(A1,A0)   
  
  G1 = matrix(c(1),nrow=2^K_g,ncol=1)
  G0 = matrix(c(0),nrow=2^K_g,ncol=K_g)
  
  for(i in 0:(2^K_g-1))
  { for(k in 1:K_g)
    if(intToBits(i)[k]==01) G0[i+1,k]=1}
  
  G = cbind(G1,G0)  
  
  
  
  
  #setting threshold
  thres_y = matrix(0,J_y,max(M_y)+1)
  for(j in 1:J_y)
  {
    thres = seq(0,1*(M_y[j]-2),by=1)
    for(k in 1:(M_y[j]+1))
    {
      if(k == 1)
        thres_y[j,k] = -Inf
      else if(k == M_y[j]+1)
        thres_y[j,k] = Inf
      else
        thres_y[j,k] = thres[k-1]
    }
  }
  
  thres_v = matrix(0,J_v,max(M_v)+1)
  for(j in 1:J_v)
  {
    thres = seq(0,1*(M_v[j]-2),by=1)
    for(k in 1:(M_v[j]+1))
    {
      if(k == 1)
        thres_v[j,k] = -Inf
      else if(k == M_v[j]+1)
        thres_v[j,k] = Inf
      else
        thres_v[j,k] = thres[k-1]
    }
  }
  
  #setting initial value
  
  c1=1;c0=500
  #Y
  B_i=matrix(c(0),nrow=J_y,ncol=(K_a+1)) #setting initial value for B_gibbs(Beta)
  Q_MH_1=matrix(c(0),nrow=J_y,ncol=K_a) #setting initial value for Q in MH algorithm
  qqq_1=matrix(c(1),nrow=J_y,ncol=1)
  Q_qta0_1=matrix(c(0),nrow=J_y,ncol=K_a)
  Q_qta_1=cbind(qqq_1,Q_qta0_1) #setting matrix for Q_qta
  omega1=0.4 #setting initial value for omega
  VQ_1=Q_qta_1/c1+(1-Q_qta_1)/c0 #calculating V given Q_qta
  
  #V
  L_i=matrix(c(0),nrow=J_v,ncol=(K_g+1))
  Q_MH_2=matrix(c(0),nrow=J_v,ncol=K_g) #setting initial value for Q in MH algorithm
  qqq_2=matrix(c(1),nrow=J_v,ncol=1)
  Q_qta0_2=matrix(c(0),nrow=J_v,ncol=K_g)
  Q_qta_2=cbind(qqq_2,Q_qta0_2) #setting matrix for Q_qta
  omega2=0.4 #setting initial value for omega
  VQ_2=Q_qta_2/c1+(1-Q_qta_2)/c0 #calculating V given Q_qta
  
  
  
  
  Y_star_new = matrix(0,nrow = n,ncol=J_y)
  V_star_new = matrix(0,nrow = n,ncol=J_v)
  
  A_cate = matrix(0,nrow = n, ncol = (K_a+1))
  
  n_cate_a = rep(0,2^K_a)
  i_c_a = numeric(n)
  pi_a = rep(1/(2^K_a),2^K_a)
  for(i in 1:n)
  {
    i_c_a[i] = sample(0:((2^K_a)-1),1,prob = pi_a)
    A_cate[i,] = A[i_c_a[i]+1,]
  }
  n_cate_a[(as.integer(names(table(i_c_a)))+1)] <- as.integer(table(i_c_a))
  
  
  G_cate = matrix(0,nrow = n, ncol = (K_g+1))
  G_catecov = matrix(0,nrow = n, ncol = (K_g+2))
  
  n_cate_g = rep(0,2^K_g)
  i_c_g = numeric(n)
  pi_g = rep(1/(2^K_g),2^K_g)
  for(i in 1:n)
  {
    i_c_g[i] = sample(0:((2^K_g)-1),1,prob = pi_g)
    G_cate[i,] = G[i_c_g[i]+1,]
   
  }
  G_catecov = cbind(G_cate,covarites)
  
  n_cate_g[(as.integer(names(table(i_c_g)))+1)] <- as.integer(table(i_c_g))
  
  Sita = matrix(rnorm((ncol(G_catecov))*K_a,0.01,1),(ncol(G_catecov)),K_a)
  Means_prior = matrix(0,nrow(Sita),K_a)
  Cov_prior = array(0,c(nrow(Sita),nrow(Sita),K_a))
  for(i in 1:K_a)
  {
    Cov_prior[,,i] = diag(nrow(Sita))
  }
  
  W = matrix(0.5,K_a,n)
  
  Q1_afburn = list()
  Q2_afburn = list()
  B_i_afburn = list()
  L_i_afburn = list()
  Sita_afburn = list()
  i_a_afburn = t(i_c_a)
  i_g_afburn = t(i_c_g)
  pi_a_afburn = t(pi_a)
  pi_g_afburn = t(pi_g)
  
  Q1_afburn[[1]] = Q_MH_1
  Q2_afburn[[1]] = Q_MH_2
  B_i_afburn[[1]] = B_i
  L_i_afburn[[1]] = L_i
  Sita_afburn[[1]] = Sita
  
  
  p_y <- array(0, dim = c(iteration, n, J_y))
  p_v <- array(0, dim = c(iteration, n, J_v))
  
  P_YVber = matrix(0,n,iteration)
  
  for(t in 1:iteration)
  {
    if(t %%1000 ==0)
      print(t)
    
    #Y----
    Y_combination = f_alp_Ystar_pi(n_cate_a,i_c_a,A_cate,A,B_i,thres_y,Y,M_y,Sita,G_cate,G_catecov)
    n_cate_a = Y_combination$n_cate
    i_c_a = Y_combination$i_cate
    A_cate = Y_combination$A_cate
    Y_star_new = Y_combination$Y_star_gibbs
    pi_a = Y_combination$pi_gibbs
    
    
    # print("Yover")
    
    #V----
    V_combination = f_gam_Vstar_pi(n_cate_g,i_c_g,G_cate,G,L_i,thres_v,V,M_v,Sita,A_cate,covarites,G_catecov)
    n_cate_g = V_combination$n_cate
    i_c_g = V_combination$i_cate
    G_cate = V_combination$G_cate
    V_star_new = V_combination$V_star_gibbs
    pi_g = V_combination$pi_gibbs
    G_catecov = cbind(G_cate,covarites)
    
    # print("Vover")
    
    #sita----
    SitacoefW = f_Sitacoef_w(A_cate,W,G_cate,Sita,Means_prior,Cov_prior,G_catecov)
    W = SitacoefW$W
    Sita = SitacoefW$Sita
    
    #print("Sitaover")
    
    #Q1----
    Q1_combin = f_Q_Beta_omega(A_cate,B_i,Y_star_new,c1,c0,omega1,VQ_1,Q_MH_1,Q_qta_1)
    #Q1_combin = fixQ_Beta_omega(A_cate,B_i,Y_star_new,c1,c0,omega1,VQ_1,Q_MH_1,Q_qta_1)
    Q_MH_1 = Q1_combin$Q_MH
    Q_qta_1 = Q1_combin$Q_qta
    B_i = Q1_combin$B_i
    VQ_1 =Q1_combin$V_Q
    omega1 = Q1_combin$omega
    
    #Q_2----
    Q2_combin = f_Q_Beta_omega(G_cate,L_i,V_star_new,c1,c0,omega2,VQ_2,Q_MH_2,Q_qta_2)
    #Q2_combin = fixQ_Beta_omega(G_cate,L_i,V_star_new,c1,c0,omega2,VQ_2,Q_MH_2,Q_qta_2)
    Q_MH_2 = Q2_combin$Q_MH
    Q_qta_2 = Q2_combin$Q_qta
    L_i = Q2_combin$B_i
    VQ_2 =Q2_combin$V_Q
    omega2 = Q2_combin$omega
    
    #record----
    {
      Q1_afburn[[t+1]] = Q_MH_1
      Q2_afburn[[t+1]] = Q_MH_2
      B_i_afburn[[t+1]] = B_i
      L_i_afburn[[t+1]] = L_i
      Sita_afburn[[t+1]] = Sita
      i_a_afburn = rbind(i_a_afburn,t(i_c_a))
      i_g_afburn = rbind(i_g_afburn,t(i_c_g))
      pi_a_afburn = rbind(pi_a_afburn,t(pi_a))
      pi_g_afburn = rbind(pi_g_afburn,t(pi_g))
    }
    
    
    # likelihood for bic
    WAIC = WAIC_y_v(Y,V,i_c_a,i_c_g,A,G,B_i,L_i,M_y,M_v,thres_y,thres_v,Sita,G_catecov,A_cate)
    P_YVber[,t] = WAIC$P_YVber
   
  }
  
  #pbic =((2*K_a+1)*J_y+(2*K_g+1)*J_v+K_a*(K_g+1+ncol(covarites)))*log(n)-2*sum(apply(log(P_YVber[,burn_in:iteration]),MARGIN = 2, FUN = mean))
  
  return(list(
    Q1_list = Q1_afburn,
    Q2_list = Q2_afburn,
    B_list = B_i_afburn,
    L_list = L_i_afburn,
    Sita_list = Sita_afburn,
    P_YVber = P_YVber
  ))
}


PostMean <- function(Mat_afburn,burn_in){
  mat_dim <- dim(Mat_afburn[[1]])
  sum_matrix <- matrix(0, nrow = mat_dim[1], ncol = mat_dim[2])
  for (mat in Mat_afburn[burn_in:length(Mat_afburn)]) {
    sum_matrix <- sum_matrix + mat
  }
  # calculate mean matrix
  average_mat <- sum_matrix / (length(Mat_afburn)-burn_in+1)
  return(average_mat)
}



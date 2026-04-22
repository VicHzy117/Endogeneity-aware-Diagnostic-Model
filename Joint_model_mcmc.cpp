#include <RcppArmadillo.h>
#include <iostream>
#include <limits>
#include <vector>
//#include "PG.h"
#include <Rcpp.h>
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]



double TRUNC = 0.64;
double pi = acos(-1);


double p_igauss(double x, double mu,double lambda)
{
  double  y;
  double Z = 1.0 / mu;
  double b = sqrt(lambda / x) * (x * Z - 1);
  double a = -1.0 * sqrt(lambda / x) * (x * Z + 1);
  y = exp(R::pnorm(b,0,1,true,true)) + exp(2 * lambda * Z + R::pnorm(a,0,1,true,true));
  return y;
}




double mass_texpon(double Z)
{
  double x = TRUNC;
  double fz = pow(pi,2)/ 8 + pow(Z,2)/ 2;
  double b = sqrt(1.0 / x) * (x * Z - 1);
  double a = -1.0 * sqrt(1.0 / x) * (x * Z + 1);
  
  double x0 = log(fz) + fz * TRUNC;
  double xb = x0 - Z + R::pnorm(b,0,1,true,true);
  double xa = x0 + Z + R::pnorm(a,0,1,true,true);
  
  double qdivp = 4 / pi * ( exp(xb) + exp(xa) );
  
  return 1.0 / (1.0 + qdivp);
}




double r_tigauss(double Z)
{
  double R=TRUNC;
  Z = abs(Z);
  double mu = 1/Z;
  double X = R + 1;
  if (mu > R) {
    double alpha = 0.0;
    while (runif(1)[0] > alpha) {
      NumericVector E =rexp(2);
      while ( pow(E[0],2) > 2 * E[1] / R) {
        E = rexp(2);
      }
      X = R / pow((1 + R*E[0]),2);
      alpha = exp(-0.5 * pow(Z,2) * X);
    }
  }
  else {
    while (X > R) {
      double lambda = 1.0;
      double Y =pow(rnorm(1)[0],2);
      X = mu + 0.5 * pow(mu,2) / lambda * Y -
        0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + pow((mu * Y),2));
      if ( runif(1)[0] > mu / (mu + X) ) {
        X = pow(mu,2)  / X;
      }
    }
  }
  return X;
}




double a_coef(double n,double x)
{
  double a;
  if ( x>TRUNC )
    a=pi * (n+0.5) * exp( -pow((n+0.5),2)*pow(pi,2)*x/2 );
  else
    a=pow((2/pi/x),1.5) * pi * (n+0.5) * exp( -2*pow((n+0.5),2)/x );
  return a;
}


// [[Rcpp::export]]
double rpg_devroye_1(double Z)
{
  Z = abs(Z) * 0.5;
  double fz = pow(pi,2)/ 8 + pow(Z,2)/ 2;
  int num_trials = 0;
  int total_iter = 0;
  double X;
  
  while (TRUE)
  {
    num_trials = num_trials + 1;
    if ( runif(1)[0] < mass_texpon(Z) ) {
      X = TRUNC + rexp(1)[0] / fz;
    }
    else {
      X = r_tigauss(Z);
    }
    
    
    
    double S = a_coef(0,X);
    double Y = runif(1)[0]*S;
    int n = 0;
    
    while (TRUE)
    {
      n = n + 1;
      total_iter = total_iter + 1;
      if ( n % 2 == 1 )
      {
        S = S - a_coef(n,X);
        if ( Y<=S ) break;
      }
      else
      {
        S = S + a_coef(n,X);
        if ( Y>S ) break;
      }
    }
    if ( Y<=S ) break;
  }
  return 0.25 * X;
  
}



// [[Rcpp::export]]
double rpg_devroye_R(int h,double z)
{
  int n = h;
  
  double x = 0;
  for (int j = 0; j < n; ++j) {
    double temp = rpg_devroye_1(z);
    x = x + temp;
  }
  return x;
}



// [[Rcpp::export]]
double rtruncnorm1(double mean, double sd, double lower, double upper) 
{
  double value;
  
  do {
    value = R::rnorm(mean, sd);
  } while (value < lower || value > upper);
  return value;
}



std::vector<double> rdirichlet1(const std::vector<double>& alpha) 
{ 
  std::vector<double> gamma_samples(alpha.size());  
  // 生成Gamma分布的随机样本
  for (size_t i = 0; i < alpha.size(); ++i) 
  {
    gamma_samples[i] = R::rgamma(alpha[i], 1.0);
  }
  
  // 归一化得到Dirichlet分布样本
  double sum = std::accumulate(gamma_samples.begin(), gamma_samples.end(), 0.0);
  for (size_t i = 0; i < gamma_samples.size(); ++i) 
  {
    gamma_samples[i] /= sum;
  }
  
  return gamma_samples;
}


double dtruncnorm1(double x, double mean, double sd, double lower, double upper) {
  double zLower = (lower - mean) / sd;
  double zUpper = (upper - mean) / sd;
  
  if (zLower >= zUpper) {
    // Invalid range, return NaN
    return std::numeric_limits<double>::quiet_NaN();
  }
  
  double cdfLower = R::pnorm(zLower, 0, 1, 1, 0);
  double cdfUpper = R::pnorm(zUpper, 0, 1, 1, 0);
  
  double pdf = R::dnorm(x, mean, sd, 0) / (cdfUpper - cdfLower);
  
  return pdf;
}


double findMin(double a, double b) 
{
  return (a < b) ? a : b;
}


double rbeta1(double alpha, double beta) {
  
  double x = R::rgamma(alpha, 1.0);
  double y = R::rgamma(beta, 1.0);
  
  // 归一化得到 Beta 分布样本
  return x / (x + y);
}


// 将arma::mat转换为std::vector<double>
std::vector<double> matToVector(const arma::mat& matrix) {
  std::vector<double> result;
  
  // 获取矩阵的行数和列数
  size_t numRows = matrix.n_rows;
  size_t numCols = matrix.n_cols;
  
  // 遍历矩阵的每个元素，并添加到向量中
  for (size_t i = 0; i < numRows; ++i) {
    for (size_t j = 0; j < numCols; ++j) {
      result.push_back(matrix(i, j));
    }
  }
  
  return result;
}


arma::mat getSubCol(const arma::mat& matrix, int colIndex) {
  // 索引从1开始，而C++中从0开始
  // colIndex -= 1;
  
  // 使用 submat 获取去掉某一列后的子矩阵
  return matrix.cols(find(arma::linspace(0, matrix.n_cols - 1, matrix.n_cols) != colIndex));
}



arma::mat sampleMultivariateNormal(const arma::vec& mean, const arma::mat& cov, int n) {
  // 生成多元正态分布的随机样本
  return arma::repmat(mean, 1, n) + arma::chol(cov).t() * arma::randn(mean.n_elem, n);
}




// [[Rcpp::export]]
arma::cube f_Thres(arma::mat A,arma::mat B_i,arma::mat Tau,arma::mat Y,arma::vec M)
{
  int n=Y.n_rows;
  int J=B_i.n_rows; 
  int K=B_i.n_cols-1;
  
  arma:mat p_gA = A*(arma::trans(B_i)) ;
  arma::cube Thres(J,pow(2,K),max(M));
  for(int j=0;j<J;j++)
    for(int c=0;c<pow(2,K);c++)
      for(int m=0;m<M(j);m++)
      {
        Thres(j,c,m) = R::pnorm(Tau(j,m+1)-p_gA(c,j),0.0,1.0,1,0)-R::pnorm(Tau(j,m)-p_gA(c,j),0.0,1.0,1,0);
      }
      return Thres;
}

// [[Rcpp::export]]
Rcpp::List f_alp_Ystar_pi(arma::vec n_cate,arma::vec i_cate,arma::mat A_cate,arma::mat A,arma::mat B_i,
                          arma::mat Tau,arma::mat Y,arma::vec M,arma::mat Theta,arma::mat G_cate,arma::mat Gcate_Covariate)
{
  int nc0=1;
  int n=Y.n_rows;
  int J=B_i.n_rows; 
  int K=B_i.n_cols-1;
  double con;
  double prob=1;
  arma::vec pi_gibbs(pow(2,K));
  int m = max(M);
  arma::cube Thres(n,pow(2,K),m); 
  arma::mat Y_star_gibbs(n,J);
  arma::mat ProbBernoulii = Gcate_Covariate*(Theta);
  arma::vec ProbCate(pow(2,K));
  arma::vec x=arma::regspace(0, 1, pow(2,K)-1);
  arma::vec numer(pow(2,K));
  Thres = f_Thres(A,B_i,Tau,Y,M);
  //Rcout<<ProbBernoulii<<std::endl;
  for(int i=0;i<n;i++)
  {
    n_cate(i_cate(i))-=1;
    for(int c=0;c<pow(2,K);c++)
    {
      double log_sita_c = 0;
      numer(c)=0;
      for(int j=0;j<J;j++)
      {
        int h = Y(i,j);
        double sita_jc = Thres(j,c,h);
        log_sita_c+=log(Thres(j,c,h));
      }
      numer(c)+=log_sita_c; //+log(nc0+n_cate(c));
      //Rcout<<numer(c)<<std::endl;
      //Rcpp::Rcout<<numer(c)<<std::endl;
     
      
      double log_pber = 0;
      for(int k=0;k<K;k++)
      {
        double pber = 1/(1+exp(-ProbBernoulii(i,k)));
        //Rcout<<pber<<" "<<A_cate(i,k+1)<<std::endl;
        
        //double prob = pow(pber,A_cate(i,k+1))*pow((1-pber),(1-A_cate(i,k+1)));
        double prob = pow(pber,A(c,k+1))*pow((1-pber),(1-A(c,k+1)));
        //Rcout<<prob<<std::endl;
        //Rcout<<prob<<std::endl;
        if(prob==0)
        {
          prob+=1e-200;
        }
        log_pber+=log(prob);
      }
      
      numer(c)+=log_pber;
      //Rcpp::Rcout<<numer(c)<<std::endl;
    }
    //con = min(numer);
    //ProbCate = exp(numer-con-log(sum(exp(numer-con))));
    
    con = max(numer);
    //Rcout<<max(numer)<<" "<<i<<std::endl;
    ProbCate = exp(numer-con);
    //Rcout<<i<<std::endl;
    //Rcout<<ProbCate<<std::endl;
    Rcpp::Function sampleFunc("sample");
    Rcpp::NumericVector sampled = sampleFunc(x, 1, Rcpp::Named("replace", false),ProbCate);  // Convert R vector back to Armadillo vector
    
    i_cate(i)=sampled(0);
    n_cate(i_cate(i))=n_cate(i_cate(i))+1;
    A_cate.row(i)=A.row(i_cate(i));
    //Rcpp::Rcout<<"cate over"<<std::endl;
    
    arma::mat p_gYstar = A_cate*(arma::trans(B_i));
    for(int j=0;j<J;j++)
    {
      //Rcpp::Rcout<<"J"<<j<<std::endl;
      double miu_star = p_gYstar(i,j);
      Y_star_gibbs(i,j) = rtruncnorm1(miu_star,1.0,Tau(j,Y(i,j)),Tau(j,Y(i,j)+1));
      
    }
  }
  
  {//step2:formula comes from page 460(11) in paper "Bayesian Estimation of the DINA Model With Gibbs Sampling"
    arma::mat N1=arma::trans(n_cate)+ones(1,pow(2,K));
    std::vector<double> result1 = matToVector(N1);
    pi_gibbs=rdirichlet1(result1);    //pi_gibbs_trace[,r]=pi_gibbs
  }
  return Rcpp::List::create(Rcpp::Named("n_cate") = n_cate,
                            Rcpp::Named("i_cate") = i_cate ,
                            Rcpp::Named("A_cate") = A_cate,
                            Rcpp::Named("Y_star_gibbs")=Y_star_gibbs,
                            Rcpp::Named("pi_gibbs")=pi_gibbs 
  );
  
}

// [[Rcpp::export]]
Rcpp::List f_gam_Vstar_pi(arma::vec n_cate,arma::vec i_cate,arma::mat G_cate,arma::mat G,arma::mat L_i,
                          arma::mat Tau,arma::mat V,arma::vec M,arma::mat Theta,
                          arma::mat A_cate,arma::mat COV,arma::mat G_cateCov)
{
  int nc0=1;
  int n=V.n_rows;
  int J=L_i.n_rows; 
  int K=L_i.n_cols-1;
  
  int K_a = A_cate.n_cols-1;
  
  double con;
  double prob=1;
  arma::vec smallnumer(pow(2,K));
  arma::vec pi_gibbs(pow(2,K));
  int m = max(M);
  arma::cube Thres(n,pow(2,K),m); 
  arma::mat V_star_gibbs(n,J);
  //arma::mat ProbBernoulii = (G_Covariate)*(Theta);
  arma::vec ProbCate(pow(2,K));
  arma::vec x=arma::regspace(0, 1, pow(2,K)-1);
  arma::vec numer(pow(2,K));
  arma::mat cov_i(pow(2,K),COV.n_cols);
  //arma::mat G_Covariate((K+1+COV.n_cols),pow(2,K));
  arma::mat G_Covariate(pow(2,K),(K+1+COV.n_cols));
  Thres=f_Thres(G,L_i,Tau,V,M);
  
  // for(int l=0;l<pow(2,K);l++)
  // {
  //   smallnumer(l)=0;
  //   for(int i=0;i<n;i++)
  //   {
  //     for(int k=0;k<K_a;k++)
  //     {
  //       double pber = 1/(1+exp(-ProbBernoulii(i,k)));
  //       smallnumer(l)+= log(pow(pber,A_cate(i,k+1))*pow((1-pber),1-A_cate(i,k+1)));
  //     }
  //     
  //   }
  //   smallnumer(l)+=log(n_cate(l)+nc0);
  // }
  // smallnumer = smallnumer-sum(smallnumer);
  
  vec covvec(pow(2,K));
  for(int i=0;i<n;i++)
  {
    for(int col=0;col<COV.n_cols;col++)
    {
      covvec = rep(COV(i,col),pow(2,K));
      cov_i.col(col) = covvec;
    }
    //Rcpp::Rcout<<"people"<<i<<std::endl;
    //cov_i = rep(COV(i),pow(2,K));
    //G_Covariate = join_rows(G,cov_i);
    G_Covariate = join_horiz(G,cov_i);
    
    arma::mat ProbBernoulii = (G_Covariate)*(Theta);
    //Rcout<<ProbBernoulii<<endl;
    n_cate(i_cate(i))-=1;
    for(int c=0;c<pow(2,K);c++)
    {
      double log_sita_c = 0;
      numer(c)=0;
      for(int j=0;j<J;j++)
      {
        int h = V(i,j);
        double sita_jc = Thres(j,c,h);
        log_sita_c+=log(sita_jc);
      }
      numer(c)+=log_sita_c+log(nc0+n_cate(c));
      //numer(c) = log_sita_c+log(pi_gibbs(c));
      //Rcpp::Rcout<<numer(c)<<std::endl;
      for(int k=0;k<K_a;k++)
      {
        double pber = 1/(1+exp(-ProbBernoulii(c,k)));
        double prob = pow(pber,A_cate(i,k+1))*pow((1-pber),(1-A_cate(i,k+1)));
       // Rcout<<prob<<std::endl;
        if(prob==0)
        {
          prob+=1e-200;
        }
        
        numer(c)+=log(prob);
        //numer(c)+= log(pow(pber,A_cate(i,k+1))*pow((1-pber),1-A_cate(i,k+1))+1e-300);
      }
      
      //numer(c)+=smallnumer(c);
     // Rcpp::Rcout<<numer(c)<<std::endl;
    }
    
    //con = min(numer);
    //ProbCate = exp(numer-con-log(sum(exp(numer-con))));
    con = max(numer);
    ProbCate = exp(numer-con);
    Rcpp::Function sampleFunc("sample");
    // Rcout<<ProbCate<<std::endl;
     //Rcout<<i<<std::endl;
    // Rcout<<ProbCate<<std::endl;
    Rcpp::NumericVector sampled = sampleFunc(x, 1, Rcpp::Named("replace", false),ProbCate);  // Convert R vector back to Armadillo vector
    i_cate(i)=sampled(0);
    n_cate(i_cate(i))=n_cate(i_cate(i))+1;
    G_cate.row(i)=G.row(i_cate(i));
    //G_cateCov.row(i)=G_Covariate.row(i_cate(i));
    
    arma::mat p_gVstar = G_cate*(arma::trans(L_i));
    for(int j=0;j<J;j++)
    {
      double miu_star = p_gVstar(i,j);
      V_star_gibbs(i,j) = rtruncnorm1(miu_star,1.0,Tau(j,V(i,j)),Tau(j,V(i,j)+1));
      
    }
  }
  
  
  {//step2:formula comes from page 460(11) in paper "Bayesian Estimation of the DINA Model With Gibbs Sampling"
    arma::mat N1=arma::trans(n_cate)+ones(1,pow(2,K));
    std::vector<double> result1 = matToVector(N1);
    pi_gibbs=rdirichlet1(result1);    //pi_gibbs_trace[,r]=pi_gibbs
  }
  
  return Rcpp::List::create(Rcpp::Named("n_cate") = n_cate,
                            Rcpp::Named("i_cate") = i_cate ,
                            Rcpp::Named("G_cate") = G_cate,
                            Rcpp::Named("V_star_gibbs")=V_star_gibbs,
                            Rcpp::Named("pi_gibbs")=pi_gibbs
                              //Rcpp::Named("G_catecov") = G_cateCov
  );
  
}


//Means_prior K2+1xK1 Cov_prior (K2+1xK2+1)xK1   W K1xn Sita (K2+1)xK1
// [[Rcpp::export]]
Rcpp::List f_Sitacoef_w(arma::mat A_cate,arma::mat W,arma::mat G_cate,arma::mat Sita,
                        arma::mat Means_prior, arma::cube Cov_prior,arma::mat Gcate_Covariate)
{
  int n = A_cate.n_rows;
  int K_1 = Sita.n_cols;
  int K_2 = Sita.n_rows;
  arma::mat gamma_sita = Gcate_Covariate*Sita;
  arma::mat Diag = arma::zeros(n,n);
  arma::mat matrix = arma::zeros(K_2,K_2);
  arma::vec m_w = arma::zeros(K_2);
  arma::mat U_w = arma::zeros(K_2,K_2);
  //arma::mat A_func = getSubCol(A_cate,0);
  arma::mat A_func = A_cate.cols(1,A_cate.n_cols-1);
  A_func = A_func-(1.0/2);
  for(int k=0;k<K_1;k++)
  {
    for(int i=0;i<n;i++)
    {
      //gamma_sita = G_cate.row(i)%Sita.col(k);
      W(k,i) = rpg_devroye_R(1,gamma_sita(i,k));
    }
    
    Diag = arma::diagmat(W.row(k));
    matrix = (arma::trans(Gcate_Covariate))*Diag*Gcate_Covariate;
    U_w = arma::inv(matrix+(arma::inv(Cov_prior.slice(k))));
    m_w = U_w*((arma::trans(Gcate_Covariate))*A_func.col(k)+(arma::inv(Cov_prior.slice(k)))*Means_prior.col(k));
    Sita.col(k) = sampleMultivariateNormal(m_w,U_w,1);
  }
  
  return Rcpp::List::create(Rcpp::Named("Sita")=Sita,
                            Rcpp::Named("W")=W);
}


// [[Rcpp::export]]
Rcpp::List f_Q_Beta_omega(arma::mat A_cate,arma::mat B_i,arma::mat Y_star_gibbs,int c1,int c0,double omega,arma::mat V_Q,arma::mat Q_MH,arma::mat Q_qta)
{
  int J=B_i.n_rows;int K=B_i.n_cols-1;
  double Tran1,Tran2,random,Trans,sigma,u;double Inf = std::numeric_limits<double>::infinity();
  arma::mat D=arma::trans(A_cate)*A_cate; 
  
  // Rcpp::Function runif("runif");
  {//#step3:formula
    for(int j=0;j<J;++j)
    { for(int k=0;k<K;++k)
    { 
      Tran1=dtruncnorm1(B_i(j,k+1),0.0,sqrt(1/c1),0.0,Inf);
      Tran2=dtruncnorm1(B_i(j,k+1),0.0,sqrt(1.0/c0),0.0,Inf);  
      
      Trans=Tran2/Tran1*(1-omega)/omega;
      
      arma::vec random_vector = Rcpp::as<arma::vec>(Rcpp::runif(1, 0.0, 1.0));
      random=random_vector(0);
      
      if( random<findMin( 1.0,pow(Trans,(2*Q_MH(j,k)-1) ) ) ) //#condition for accepting transformation
      {
        Q_MH(j,k)=1-Q_MH(j,k); //#transformation
        Q_qta(j,k+1)=Q_MH(j,k);//#calculating the value of Q_qta after transformation
        V_Q=Q_qta/c1+(1-Q_qta)/c0; }//#calculating the value of V_Q after transformation
      
      // Q_MH_trace[j,k,r]=Q_MH[j,k] #record the value of Q_MH
    }
    
    
    for(int l=0;l<(K+1);++l)
    { sigma=1/(D(l,l)+1/V_Q(j,l));
      arma::mat u1=arma::trans(A_cate.col(l))*(Y_star_gibbs.col(j)-getSubCol(A_cate,l)*arma::trans(getSubCol(B_i.row(j),l)));
      u=u1(0,0);
      if(l==0) {B_i(j,l)=(rnorm(1,sigma*u,sqrt(sigma)))(0);}
      else {B_i(j,l)=rtruncnorm1(sigma*u,sqrt(sigma),0.0,Inf) ;}
    } 
    }
  }
  
  { omega=rbeta1(accu(Q_MH)+1,J*K-accu(Q_MH)+1) ;}  
  return Rcpp::List::create(Rcpp::Named("Q_MH") = Q_MH,
                            Rcpp::Named("Q_qta") = Q_qta ,
                            Rcpp::Named("B_i") = B_i,
                            Rcpp::Named("V_Q")=V_Q,
                            Rcpp::Named("omega")=omega
                              
  );
}

// [[Rcpp::export]]
Rcpp::List fixQ_Beta_omega(arma::mat A_cate,arma::mat B_i,arma::mat Y_star_gibbs,int c1,int c0,double omega,arma::mat V_Q,arma::mat Q_MH,arma::mat Q_qta)
{
  int J=B_i.n_rows;int K=B_i.n_cols-1;
  double Tran1,Tran2,random,Trans,sigma,u;double Inf = std::numeric_limits<double>::infinity();
  arma::mat D=arma::trans(A_cate)*A_cate; 
  
  // Rcpp::Function runif("runif");
  {//#step3:formula
    for(int j=0;j<J;++j)
    { 
      
    
    
    for(int l=0;l<(K+1);++l)
    { sigma=1/(D(l,l)+1/V_Q(j,l));
      arma::mat u1=arma::trans(A_cate.col(l))*(Y_star_gibbs.col(j)-getSubCol(A_cate,l)*arma::trans(getSubCol(B_i.row(j),l)));
      u=u1(0,0);
      if(l==0) {B_i(j,l)=(rnorm(1,sigma*u,sqrt(sigma)))(0);}
      else {B_i(j,l)=rtruncnorm1(sigma*u,sqrt(sigma),0.0,Inf) ;}
    } 
    }
  }
  
  { omega=rbeta1(accu(Q_MH)+1,J*K-accu(Q_MH)+1) ;}  
  return Rcpp::List::create(Rcpp::Named("Q_MH") = Q_MH,
                            Rcpp::Named("Q_qta") = Q_qta ,
                            Rcpp::Named("B_i") = B_i,
                            Rcpp::Named("V_Q")=V_Q,
                            Rcpp::Named("omega")=omega
                              
  );
}


// [[Rcpp::export]]
Rcpp::List WAIC_y_v(mat Y, mat V, vec i_cate_a, vec i_cate_g, mat A, mat G,
                    mat B_i, mat L_i,vec M_y, vec M_v,mat tau_y,mat tau_v,
                    mat Theta,mat Gcate_Covariate,mat A_cate)
{
  int n=V.n_rows;
  int J_y=B_i.n_rows;
  int J_v=L_i.n_rows; 
  int m_y = max(M_y);
  int m_v = max(M_v);
  int K_y = A.n_cols-1;
  int K_v = G.n_cols-1;
  arma::cube Thres_y(n,pow(2,K_y),m_y); 
  Thres_y = f_Thres(A,B_i,tau_y,Y,M_y);
  arma::cube Thres_v(n,pow(2,K_v),m_v); 
  Thres_v = f_Thres(G,L_i,tau_v,V,M_v);
  arma::mat ProbBernoulii = (Gcate_Covariate)*(Theta);
  
  arma::mat P_Y(n,J_y);
  arma::mat P_V(n,J_v);
  arma::mat P_ber(n,K_y);
  
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<J_y;j++)
    {
      P_Y(i,j) = Thres_y(j,i_cate_a(i),Y(i,j));
    }
    
    for(int j=0;j<J_v;j++)
    {
      P_V(i,j) = Thres_v(j,i_cate_g(i),V(i,j));
    }
    
    for(int k=0;k<K_y;k++)
    {
      double pber = 1/(1+exp(-ProbBernoulii(i,k)));
      double prob = pow(pber,A_cate(i,k+1))*pow((1-pber),(1-A_cate(i,k+1)));
      P_ber(i,k) = prob;
    }
  }
  
  arma::mat combined = join_horiz(P_Y, P_V, P_ber);
  // 按行累乘
  arma::vec result = prod(combined, 1);
  
  
  return Rcpp::List::create(Named("P_YVber")=result);
  
}




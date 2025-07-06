#!/usr/bin/env python
# coding:utf-8
#from rpy2.rpy_classic import *
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects
from numpy import *
import numpy as np
from scipy.stats import norm
import scipy.stats as st
import scipy
import math
from scipy.stats import spearmanr # Spearman's rank corr
#from rankcorr import *

#set_default_mode(BASIC_CONVERSION)

class MYSTAT:
  def mean_diff_test(self,xi,yi,flag,sig_level,ret=0):
    #母集団が異なるxiとyiの平均値の差が有意であるか検定する
    #二群の平均値の差の検定
    #sig_level=0.05 は有意水準5%(両側)
    xnum=len(xi)
    ynum=len(yi)
    xmean=float(self.mean(xi)) #xiの平均
    ymean=float(self.mean(yi)) #yiの平均
    xvar=float(self.var(xi))   #xiの不遍分散
    yvar=float(self.var(yi))   #yiの不遍分散
    mdiff=float(ymean)-float(xmean)   #差

    df=0    #自由度
    tstat=999.9 #t値

    #検定方法
    if int(flag) == 1:    #等分散性を仮定した
       df=xnum+ynum-2 #自由度
       v=((xnum-1)*xvar+(ynum-1)*yvar)/float(df) # プールした不偏分散
       tstat=abs((xmean-ymean))/math.sqrt(v*(1/float(xnum)+1/float(ynum))) # 検定統計量
    elif int(flag) == 2: #Welchの方法による
       df = (xvar/xnum+yvar/ynum)**2/((xvar/xnum)**2/(xnum-1)+(yvar/ynum)**2/(ynum-1)) # 自由度（小数点つき）
       tstat=abs((xmean-ymean))/math.sqrt(xvar/float(xnum)+yvar/float(ynum)) # 検定統計量
    elif int(flag) == 3:  #Welchの方法でautocorrelationによる自由度調整を含む場合
       acx=self.autocorrelation(xi,1) #lag1の自己相関係数 for xi
       acy=self.autocorrelation(yi,1) #lag1の自己相関係数 for yi
       xnum2=int((xnum*(1-acx)/(1+acx))) #自己相関による有効サンプル数
       ynum2=int((ynum*(1-acy)/(1+acy))) #自己相関による有効サンプル数
       if xnum2 > xnum:
           xnum2=xnum
       if ynum2 > ynum:
           ynum2=ynum
       df = (xvar/xnum2+yvar/ynum2)**2/((xvar/xnum2)**2/(xnum2-1)+(yvar/ynum2)**2/(ynum2-1)) # 自由度（小数点つき）
       tstat=abs((xmean-ymean))/math.sqrt(xvar/float(xnum)+yvar/float(ynum)) # 検定統計量

   #e=r.pt(float(tstat),int(df),0,lower_tail=0)[0] #P値
    if (int(flag) == 2 or int(flag) == 3) and (xvar == 0.0 and yvar == 0.0 ):
       diff=0
    elif (int(flag) == 3 and (xnum2 == 0 or ynum2 == 0 )):
       diff=0
    else:
       kwargs={'lower.tail':0}
       e=r.pt(float(tstat),int(df),0,**kwargs)[0] #P値
       pvalue=2*float(e)
       diff=0
       if pvalue < float(sig_level):
           diff=1 #有意な差あり
       else:
           diff=0 #有意な差なし

   #信頼区間の計算
    if ret == 1: 
      #x,yの信頼区間
       trustx=self.tvalue(xnum-1,sig_level/2.0)*xvar/math.sqrt(float(xnum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)
       trusty=self.tvalue(ynum-1,sig_level/2.0)*yvar/math.sqrt(float(ynum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)

      #差の信頼区間
       xvar2=xvar*(xnum-1)/xnum
       yvar2=yvar*(ynum-1)/ynum
       std1=(xvar2*float(xnum)+yvar2*float(ynum))/float(xnum-1+ynum-1) #推定母分散
       dstd=math.sqrt(std1*(1/float(xnum)+(1/float(ynum)))) #差の標本標準誤差
       trustd=self.tbunpu((xnum-1+ynum-1),sig_level/2.0)*dstd #tbunpuのinputは片側有意水準なので２で割る

    if ret == 0:
       return xmean,ymean,mdiff,diff
    elif ret == 1: #信頼区間をリターン
       return xmean,ymean,mdiff,diff,trustx,trusty,trustd
    elif ret == 2: #pvalueをリターン
       return xmean,ymean,mdiff,diff,pvalue

  def mean_diff_test_given(self,xnum,xmean,xvar,ynum,ymean,yvar,flag,sig_level,acx=0.0,acy=0.0,ret=0):
    #母集団が異なるxiとyiの平均値の差が有意であるか検定する
    #二群の平均値の差の検定
    #sig_level=0.05は有意水準5%(両側)

    mdiff=float(ymean)-float(xmean)   #差

    df=0    #自由度
    tstat=999.9 #t値
    pvalue=-9.99E33

    #検定方法
    if int(flag) == 1:    #等分散性を仮定した
       df=xnum+ynum-2 #自由度
       v=((xnum-1)*xvar+(ynum-1)*yvar)/float(df) # プールした不偏分散
       tstat=abs((xmean-ymean))/math.sqrt(v*(1/float(xnum)+1/float(ynum))) # 検定統計量
    elif int(flag) == 2: #Welchの方法による
       df = (xvar/xnum+yvar/ynum)**2/((xvar/xnum)**2/(xnum-1)+(yvar/ynum)**2/(ynum-1)) # 自由度（小数点つき）
       tstat=abs((xmean-ymean))/math.sqrt(xvar/float(xnum)+yvar/float(ynum)) # 検定統計量
    elif int(flag) == 3:  #Welchの方法でautocorrelationによる自由度調整を含む場合
       acx=self.autocorrelation(xi,1) #lag1の自己相関係数 for xi
       acy=self.autocorrelation(yi,1) #lag1の自己相関係数 for yi
       xnum2=int((xnum*(1-acx)/(1+acx))) #自己相関による有効サンプル数
       ynum2=int((ynum*(1-acy)/(1+acy))) #自己相関による有効サンプル数
       if xnum2 > xnum:
           xnum2=xnum
       if ynum2 > ynum:
           ynum2=ynum
       df = (xvar/xnum2+yvar/ynum2)**2/((xvar/xnum2)**2/(xnum2-1)+(yvar/ynum2)**2/(ynum2-1)) # 自由度（小数点つき）
       tstat=abs((xmean-ymean))/math.sqrt(xvar/float(xnum)+yvar/float(ynum)) # 検定統計量

   #e=r.pt(float(tstat),int(df),0,lower_tail=0)[0] #P値
    if (int(flag) == 2 or int(flag) == 3) and (xvar == 0.0 and yvar == 0.0 ):
       diff=0
    elif (int(flag) == 3 and (xnum2 == 0 or ynum2 == 0 )):
       diff=0
    else:
       kwargs={'lower.tail':0}
       e=r.pt(float(tstat),int(df),0,**kwargs)[0] #P値
       pvalue=2*float(e)
       diff=0
       if pvalue < float(sig_level):
           diff=1 #有意な差あり
       else:
           diff=0 #有意な差なし

   #信頼区間の計算
    if ret == 1: 
      #x,yの信頼区間
       trustx=self.tvalue(xnum-1,sig_level/2.0)*xvar/math.sqrt(float(xnum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)
       trusty=self.tvalue(ynum-1,sig_level/2.0)*yvar/math.sqrt(float(ynum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)

      #差の信頼区間
       xvar2=xvar*(xnum-1)/xnum
       yvar2=yvar*(ynum-1)/ynum
       std1=(xvar2*float(xnum)+yvar2*float(ynum))/float(xnum-1+ynum-1) #推定母分散
       dstd=math.sqrt(std1*(1/float(xnum)+(1/float(ynum)))) #差の標本標準誤差
       trustd=self.tbunpu((xnum-1+ynum-1),sig_level/2.0)*dstd #tbunpuのinputは片側有意水準なので２で割る

    if ret == 0:
       return diff,pvalue
    elif ret == 1: #信頼区間をリターン
       return diff,pvalue,trustx,trusty,trustd
    elif ret == 2: #pvalueをリターン
       return xmean,ymean,mdiff,diff,pvalue
    elif ret == 3:
       return pvalue

  def mean_paired_diff_test(self,xi,yi,sig_level,ret=0):
    #対応のあるxiとyiの平均値の差が有意であるか検定する
    #sig_level=0.05 は有意水準5%(両側)

    xmean=float(self.mean(xi)) #xiの平均
    ymean=float(self.mean(yi)) #yiの平均
    xvar=float(self.var(xi))   #xiの不遍分散
    yvar=float(self.var(yi))   #yiの不遍分散
    xnum=len(xi)
    ynum=len(yi)
    if xnum != ynum:
       print("error in mean_paired_diff_test")
       print("xnum should be equal to ynum")
       raise
    dd=yi-xi
    dmean=float(self.mean(dd)) #yi-xiの平均
    dsd=float(self.sd(dd))     #yi-xiの標準偏差
    mdiff=float(ymean)-float(xmean)   #差

    pvalue=-9.99E33
    df=xnum-1 #自由度
    tstat=abs(dmean)/(dsd/math.sqrt(float(xnum)))

   #e=r.pt(float(tstat),int(df),0,lower_tail=0)[0] #P値
    if xvar == 0.0 and yvar == 0.0:
       diff=0
    elif xnum == 0 or ynum == 0:
       diff=0
    else:
       kwargs={'lower.tail':0}
       e=r.pt(float(tstat),int(df),0,**kwargs)[0] #P値
       pvalue=2*float(e)
       diff=0
       print(tstat,pvalue)
       if pvalue < float(sig_level):
           diff=1 #有意な差あり
       else:
           diff=0 #有意な差なし

   #信頼区間の計算
    if ret == 1: 
      #x,yの信頼区間
       trustx=self.tvalue(xnum-1,sig_level/2.0)*xvar/math.sqrt(float(xnum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)
       trusty=self.tvalue(ynum-1,sig_level/2.0)*yvar/math.sqrt(float(ynum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)

      #差の信頼区間
       xvar2=xvar*(xnum-1)/xnum
       yvar2=yvar*(ynum-1)/ynum
       std1=(xvar2*float(xnum)+yvar2*float(ynum))/float(xnum-1+ynum-1) #推定母分散
       dstd=math.sqrt(std1*(1/float(xnum)+(1/float(ynum)))) #差の標本標準誤差
       trustd=self.tbunpu((xnum-1+ynum-1),sig_level/2.0)*dstd #tbunpuのinputは片側有意水準なので２で割る

    if ret == 0:
       return xmean,ymean,mdiff,diff
    elif ret == 1: #信頼区間をリターン
       return xmean,ymean,mdiff,diff,trustx,trusty,trustd

  def mean_paired_diff_test_given(self,xnum,xmean,xvar,xsd,ynum,ymean,yvar,ysd,xycor,sig_level,ret=0):
    #対応のあるxiとyiの平均値の差が有意であるか検定する
    #sig_level=0.05は有意水準5%(両側)

    if xnum != ynum:
       print("error in mean_paired_diff_test")
       print("xnum should be equal to ynum")
       raise

    mdiff=float(ymean)-float(xmean)   #差

    pvalue=-9.99E33
    df=xnum-1 #自由度

    tstat=abs(ymean-xmean)/math.sqrt((xsd+ysd-2*xycor*math.sqrt(xsd*ysd))/xnum)

   #e=r.pt(float(tstat),int(df),0,lower_tail=0)[0] #P値
    if xvar == 0.0 and yvar == 0.0:
       diff=0
    elif xnum == 0 or ynum == 0:
       diff=0
    else:
       kwargs={'lower.tail':0}
       e=r.pt(float(tstat),int(df),0,**kwargs)[0] #P値
       pvalue=2*float(e)
       diff=0
       if pvalue < float(sig_level):
           diff=1 #有意な差あり
       else:
           diff=0 #有意な差なし

   #信頼区間の計算
    if ret == 1: 
      #x,yの信頼区間
       trustx=self.tvalue(xnum-1,sig_level/2.0)*xvar/math.sqrt(float(xnum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)
       trusty=self.tvalue(ynum-1,sig_level/2.0)*yvar/math.sqrt(float(ynum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)

      #差の信頼区間
       xvar2=xvar*(xnum-1)/xnum
       yvar2=yvar*(ynum-1)/ynum
       std1=(xvar2*float(xnum)+yvar2*float(ynum))/float(xnum-1+ynum-1) #推定母分散
       dstd=math.sqrt(std1*(1/float(xnum)+(1/float(ynum)))) #差の標本標準誤差
       trustd=self.tbunpu((xnum-1+ynum-1),sig_level/2.0)*dstd #tbunpuのinputは片側有意水準なので２で割る

    if ret == 0:
       return diff,pvalue
    elif ret == 1: #信頼区間をリターン
       return diff,pvalue,trustx,trusty,trustd

  def conf_interval(self,xi,sig_level,ret=0):
    #信頼区間を計算する
    #sig_level=0.05 は有意水準5%(両側)
    xnum=len(xi)
    xmean=float(self.mean(xi)) #xiの平均
    xvar=float(self.var(xi))   #xiの不遍分散

    df=0    #自由度
    tstat=999.9 #t値

   #信頼区間の計算
    trustx=self.tvalue(xnum-1,sig_level/2.0)*xvar/math.sqrt(float(xnum-1)) #tvalueは片側有意水準をinput(i.e,2で割る)
   
    if ret==0:
      return trustx
    elif ret==1:
      return xmean-trustx, xmean+trustx


  def var_test(self,xi,yi,sig_level,ret=0):
    #母集団が異なるxiとyiの二群の等分散性の検定する
    #sig_level=0.05 は有意水準5%
    xnum=len(xi)
    ynum=len(yi)
    xmean=float(self.mean(xi)) #xiの平均
    ymean=float(self.mean(yi)) #yiの平均
    xvar=float(self.var(xi))   #xiの不遍分散
    yvar=float(self.var(yi))   #yiの不遍分散

    df1=0    #自由度
    df2=0    #自由度
    f=0

    if xvar > yvar: 
       f=xvar/yvar
       df1=xnum-1
       df2=ynum-1
    else:
       f=yvar/xvar
       df1=ynum-1
       df2=xnum-1

    kwargs={'lower.tail':0}
    e=r.pf(float(f),int(df1),int(df2),0,**kwargs)[0] #P値
    pvalue=2*float(e)
    diff=0
    if pvalue < float(sig_level):
        diff=1 #分散は等しくない
    else:
        diff=0 #分散は等しい

    if ret==0:
       return xvar,yvar,diff
    elif ret==1:
       return xvar,yvar,diff,pvalue

  def var_test_given(self,xnum,xvar,ynum,yvar,sig_level,ret=0):
    #母集団が異なるxiとyiの二群の等分散性の検定する
    #sig_level=0.05 は有意水準5%
    #xnum : xのサンプル数
    #ynum : yのサンプル数
    #xvar :xiの不遍分散
    #yvar :yiの不遍分散

    df1=0    #自由度
    df2=0    #自由度
    f=0

    if xvar > yvar: 
       f=xvar/yvar
       df1=xnum-1
       df2=ynum-1
    else:
       f=yvar/xvar
       df1=ynum-1
       df2=xnum-1

    kwargs={'lower.tail':0}
    e=r.pf(float(f),int(df1),int(df2),0,**kwargs)[0] #P値
    pvalue=2*float(e)
    diff=0
    if pvalue < float(sig_level):
        diff=1 #分散は等しくない
    else:
        diff=0 #分散は等しい

    return diff,pvalue

  def cal_contingency_table(self,mdata,odata,thre_obs,thre_mdl):
    """
                   |         |
                   |    b    |    a    
                   |         |
   mdata  thre_mdl |--------------------
                   |         |
                   |    d    |    c
                   |         |
                   ----------------------
                         thre_obs

                           odata 
    """

    a1 = np.ma.masked_where((mdata>=thre_mdl)&(odata>=thre_obs),mdata)
    b1 = np.ma.masked_where((mdata>=thre_mdl)&(odata<thre_obs),mdata)
    c1 = np.ma.masked_where((mdata<thre_mdl)&(odata>=thre_obs),mdata)
    d1 = np.ma.masked_where((mdata<thre_mdl)&(odata<thre_obs),mdata)
    a = np.ma.count_masked(a1)
    b = np.ma.count_masked(b1)
    c = np.ma.count_masked(c1)
    d = np.ma.count_masked(d1)
   #print "mdata=",mdata, "thre_mdl=",thre_mdl
   #print "odata=",odata, "thre_obs=",thre_obs
   #print "a1=",a1,"a=",a
   #print "b1=",b1,"b=",b
   #print "c1=",c1,"c=",c
   #print "d1=",d1,"d=",d
   
    return a,b,c,d

  def hit_rate(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: hit rate
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub hit_rate: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)

    if a+c!=0:
       hrate = float(a)/float(a+c)
    else:
       hrate = False

    return hrate

  def false_alarm_rate(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: false-alarm rate
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub false_alarm_rate: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)

    if a+b!=0:
       frate = float(b)/float(a+b)
    else:
       frate = False

    return frate

  def odds_ratio(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: odds ratio
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub odds_ratio: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)

    if b*c!=0:
       orate = (float(a)*float(d))/(float(b)*float(c))
    else:
       orate = False

    return orate

  def base_rate(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: base rate
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub basse_rate: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)
    n = a+b+c+d

    if n!=0:
       brate = float(a+c)/float(n)
    else:
       brate = False

    return brate

  def edi(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: EDI (Extremal Dependence Index)
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub edi: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    H = self.hit_rate(mdata,odata,thre_obs,thre_mdl=thre_mdl) # hit_rate
    F = self.false_alarm_rate(mdata,odata,thre_obs,thre_mdl=thre_mdl) # false-alarm ratio
    H = float(H)
    F = float(F)

    if F!=0.0 and H!=0.0 and (log(F) + log(H)) != 0.0:
       edi = (log(F)-log(H))/(log(F)+log(H))
    else:
       edi = False

    return edi

  def sedi(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: SEDI (Symmetric Extremal Dependence Index)
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub sedi: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    H = self.hit_rate(mdata,odata, thre_obs, thre_mdl=thre_mdl) # hit_rate
    F = self.false_alarm_rate(mdata,odata, thre_obs, thre_mdl=thre_mdl) # false-alarm ratio
    H = float(H)
    F = float(F)

    if F!=0.0 and H!=0.0 and F!=1.0 and H!=1.0:
      bunsi = log(F) - log(H) - log(1.0-F) + log(1.0-H)
      bunbo = log(F) + log(H) + log(1.0-F) + log(1.0-H)
    else:
      bunbo = 0.0

   #print "H=",H,log(H), log(1.0-H)
   #print "F=",F,log(F), log(1.0-F)
   #print "bunsi=",bunsi,"bunbo=",bunbo

    if bunbo == 0.0 or F == 0.0 or H == 0.0 or F==1.0 or H == 1.0:
       sedi = False
    else:
       sedi = float(bunsi)/float(bunbo)

    return sedi

  def eds(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: EDS (extreme dependency score)
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub eds: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)
    n = a+b+c+d

    if n!=0 and float(a)/float(n)!=0.0 and log(float(a)/float(n))!=0:
       eds = 2*log(float(a+c)/float(n))/log(float(a)/float(n)) - 1.0
    else:
       eds = False

    return eds

  def seds(self,model,obs,thre_obs,thre_mdl=False,bias_adjust=False):
    """
    input : model # model data
    input :   obs # observed data
    input : thre_obs# threshold for observations
    input : thre_mdl# threshold for model (default = thre_obs)
    input : bias_adjust# bias adjust or not (default = False)
    output: SEDS (Symmetric extreme dependency score)
    reference: Ferro and Stephenson (2011, Wea. Forecasting)
    """
    onum=len(obs)
    mnum=len(model)
    if mnum != onum:
      print("mystat error sub seds: mnum should be equal onum")
      raise

    odata = np.array(obs)
    mdata = np.array(model)

    if bias_adjust:
      mdata = mdata * odata.mean()/mdata.mean()

    if thre_mdl==False:
      thre_mdl=thre_obs

    a,b,c,d = self.cal_contingency_table(mdata, odata, thre_mdl, thre_obs)
    n = a+b+c+d

    if n!=0 and float(a)/float(n)!=0 and log(float(a)/float(n))!=0:
       seds = log((float(a+b)*float(a+c))/(float(n)*float(n)))/log(float(a)/float(n)) - 1.0
    else:
       seds = False

    return seds

  def rmse(self,data,obs,ret=0,normalize=0):
    """
     normalize 1: normalized by observed standard deviation
    """
    undef=-9.99E33
    onum=len(obs)
    mnum=len(data)
    if mnum != onum:
      print("mystat error sub rmse error: mnum should be equal onum")
      raise
    sum=0.0
    for i in range(onum):
      sum+=(float(data[i])-float(obs[i]))**2
    sum=sum/float(onum)
    rmse=math.sqrt(sum)
    if normalize==1:
       ostd = np.array(obs).std(ddof=1)
       if ostd == 0.0:
          rmse = undef
       else:
          rmse = rmse/float(ostd)
    return rmse

  def mse(self,data,obs,ret=0):
    onum=len(obs)
    mnum=len(data)
    if mnum != onum:
      print("mystat error sub rmse error: mnum should be equal onum")
      raise
    sum=0.0
    for i in range(onum):
      sum+=(float(data[i])-float(obs[i]))**2
    mse=sum/float(onum)
    return mse

  def msep(self,data,obs,ret=0,kmax=100):
    onum=len(obs)
    mnum, emax = np.shape(data)
    if mnum != onum:
      print("mystat error sub msep error: mnum should be equal onum")
      raise

    msep=0.0
    for tt, ob in enumerate(obs):
      mdata = data[tt,:]

      #--compute probability
      for kk in range(kmax+1):
        pp = float(len(np.where(mdata==kk)[0]))/float(emax)
  
        msep=msep + pp * (kk - ob)**2
    msep=msep/float(onum)
    return msep

  def msss(self,data,obs,ret=0):
    onum=len(obs)
    mnum=len(data)
    if mnum != onum:
      print("mystat error sub msss error: mnum should be equal onum")
      print("mnum=",mnum, "onum=",onum)
      raise

    msss=1.0
    omean = obs.mean(axis=0)

    bunshi = 0.0
    bunbo = 0.0
    for tt, ob in enumerate(obs):
     #print tt, "data=",data[tt]," obs=",obs[tt]
      bunshi = bunshi + math.pow(obs[tt]-data[tt],2.0)
      bunbo = bunbo + math.pow(obs[tt]-omean,2.0)
    bunshi = 1.0/float(onum)*bunshi
    bunbo = 1.0/float(onum)*bunbo

    msss = 1 - float(bunshi)/float(bunbo)
    return msss

  def msss_masked(self,data,obs,ret=0):
    onum=len(obs)
    mnum=len(data)
    if mnum != onum:
      print("mystat error sub msss error: mnum should be equal onum")
      print("mnum=",mnum, "onum=",onum)
      raise

    xi2=[]
    yi2=[]
    for ii in range(mnum):
      if data.mask[ii]==False and obs.mask[ii]==False:
         xi2.append(data[ii])      
         yi2.append(obs[ii])      

    onum2=len(yi2)
    xi2 = np.ma.array(xi2)
    yi2 = np.ma.array(yi2)

    msss=1.0
   #omean = obs.mean(axis=0)
    omean = yi2.mean(axis=0)

    bunshi = 0.0
    bunbo = 0.0
    for tt, ob in enumerate(yi2):
      bunshi = bunshi + math.pow(yi2[tt]-xi2[tt],2.0)
      bunbo = bunbo + math.pow(yi2[tt]-omean,2.0)
    bunshi = 1.0/float(onum2)*bunshi
    bunbo = 1.0/float(onum2)*bunbo

    msss = 1 - float(bunshi)/float(bunbo)
    return msss

  def rmse_masked(self,data,obs,ret=0):
    onum=len(obs)
    mnum=len(data)
    if mnum != onum:
      print("mystat error sub rmse error: mnum should be equal onum")
      raise
    sum=0.0
    num=0
    for i in range(onum):
      if (np.ma.count_masked(obs)!=0 and obs.mask[i]==True) or (np.ma.count_masked(data)!=0 and data.mask[i]==True):
        continue
      sum+=(float(data[i])-float(obs[i]))**2
      num=num+1
    sum=sum/float(num)
    rmse=math.sqrt(sum)
    return rmse

  def rmse_2d(self,data,obs,ret=0):
    rmse=float(np.ma.sqrt(np.ma.mean((data-obs)**2)))

    return rmse

  def meanerror(self,data,obs):
    meanerror = float(np.ma.mean(data-obs))
    return meanerror

  def taylor_skill_score(self,data,obs):
    stdd = float(data.flatten().std(ddof=1))
    stdo = float(obs.flatten().std(ddof=1))

    r0 = 1.0

    r = float(np.ma.corrcoef(data.flatten(),obs.flatten())[0,1])
    if (stdd != 0.0 and stdo !=0.0):
      s1 = (4 * (1 + r))/((1+r0)*(stdd/stdo + stdo/stdd)**2)
      s2 = (4 * (1 + r)**4)/(((1+r0)**4)*(stdd/stdo + stdo/stdd)**2)
    else:
      s1 = 0.0
      s2 = 0.0
      r = -9.99E33
    return s1,s2,r

  def autocorrelation(self,xi,lag): #自己相関係数の計算 
     n=len(xi)
     h=int(lag)

     #--check arguments
     if int(n) < 3 or n-h < 2 or h < 1:
       print("autocorrelation: invalid argument")

     mean=self.mean2(xi)
     num=0.0
     den=0.0
     for t in range(n-h-1):
        num = num + (xi[t] - mean)*(xi[t+h] - mean)

     den = self.var(xi)*(n-1)

     if den != 0:
       ac=float(num)/float(den)
     else:
       ac=0.0

     return ac

  def PearsonCorrelation(self,xi,yi):
    meanxi=self.mean2(xi)
    meanyi=self.mean2(yi)
    numxi=len(xi)
    numyi=len(yi)
    gamma1=0.0
    gamma2=0.0
    gamma12_0=0.0
    gamma12_h=0.0

    for i in range(numxi):
      gamma1=gamma1+(xi[i]-meanxi)**2/(float(numxi))

    for i in range(numyi):
      gamma2=gamma2+(yi[i]-meanyi)**2/(float(numyi))

    gamma12_0=math.sqrt(gamma1*gamma2)

    for i in range(numxi):
      gamma12_h=gamma12_h + (xi[i]-meanxi)*(yi[i]-meanyi)/float(numxi)

    out_cor=float(gamma12_h)/float(gamma12_0)
    return out_cor

  def correlation_test(self,xi,yi,sig_level):
    #sig_level=0.05 は両側有意水準5%

    #サンプル数
    numxi=len(xi)
    numyi=len(yi)

    if numxi != numyi:
      raise "error in correlation_test: sample numbers are different"
    num=numxi
    #相関係数
    if np.count_nonzero(xi)==0 or np.count_nonzero(yi)==0:
      cor=np.nan
      ista=0
    else:
       cor=self.PearsonCorrelation(xi,yi)
       #--t値の計算
       t0=(abs(cor) * sqrt(float(num-2)))/(sqrt(1.0-cor*cor))
       #--自由度num-1、有意水準sig_levelのt値の計算
       tg=self.tbunpu(num-2,sig_level/2.0) #tbunpuは片側検定のsig_levelを必要なので両側水準を２で割る
       #--統計的有意性の検定（t0とtgの比較)
       if abs(t0) >= abs(tg):
          ista=1  #棄却=相関は有意
       else:
          ista=0  #採択=相関は無意

    return cor,ista

  def correlation_test_pval(self,xi,yi):

    #サンプル数
    numxi=len(xi)
    numyi=len(yi)

    if numxi != numyi:
      raise "error in correlation_test: sample numbers are different"
    num=numxi
    #相関係数
    if np.count_nonzero(xi)==0 or np.count_nonzero(yi)==0:
         cor=np.nan
         pval=np.nan
    else:
         cor,pval=st.pearsonr(xi,yi)

    return cor,pval

# def rank_correlation_test_pval(self,xi,yi):
#   X = scipy.stats.stats.rankdata(xi)
#   Y = scipy.stats.stats.rankdata(yi)

#   cor,pval = self.correlation_test_pval(X,Y)

#   return cor,pval

  def rank_correlation_test_pval(self,xi,yi):
    numxi=len(xi)
    numyi=len(yi)
    if numxi != numyi:
      raise "error in rank_correlation_test: sample numbers are different"

    xi = np.array(xi)
    yi = np.array(yi)
    if np.count_nonzero(xi)==0 or np.count_nonzero(yi)==0:
         cor=np.nan
         pval=np.nan
    else:
         cor,pval = rank_corr_1dim(xi,yi)

    return cor,pval


  def inflation_calibration(self, xi, yi, opt=1):
    # xi: observations (tmax)
    # yi: ensemble simulations (tmax, emax)
    # Johnson, C. and Bowler, N.:2008: On the reliability and calibration of ensemble forecasts, Mon. Wea. Rev, 137, 1717-1720.
    
    tmax,emax = np.shape(yi)
   #print "tmax=",tmax,"emax=",emax

    if np.count_nonzero(yi.mean(axis=1))==0 or np.count_nonzero(xi)==0:
       alpha=0
       beta=0
    else:
       rho = self.PearsonCorrelation(xi, yi.mean(axis=1)) # corr(x, fbar)
      #print "xi=",xi," yi=",yi.mean(axis=1)
      #print "rho=",rho
       
       sigma_x = np.std(xi) # Eq. 2a
       sigma_fvar = np.std(yi.mean(axis=1)) # Eq. 2b
       sigma_t = np.std(yi, axis=1) # Eq.1b
       sigma_e = math.sqrt(np.array(sigma_t*sigma_t).mean()) # Eq. 2d
       
       alpha = rho * (float(sigma_x)/float(sigma_fvar)) # Eq. 7a
       beta  = math.sqrt((1-rho*rho)*(sigma_x*sigma_x)/(sigma_e*sigma_e)) # Eq. 7b
       print("alpha=",alpha, " beta=",beta)

    ndata = np.zeros((tmax,emax))
    for tt in range(tmax):
      for ee in range(emax):
         ftbar = yi[tt,:].mean()
         epsilon = yi[tt,ee] - ftbar
         ndata[tt,ee] = alpha * ftbar + beta * epsilon
    
    if opt ==1:
      return ndata
    elif opt==2:
      return alpha,beta

  def inflation_calibration_from_alpha_beta(self, yi, alpha, beta):
    # xi: observations (tmax)
    # yi: ensemble simulations (tmax, emax)
    # Johnson, C. and Bowler, N.:2008: On the reliability and calibration of ensemble forecasts, Mon. Wea. Rev, 137, 1717-1720.
    
    tmax,emax = np.shape(yi)
   #print "alpha=",alpha, " beta=",beta

    ndata = np.zeros((tmax,emax))
    for tt in range(tmax):
      for ee in range(emax):
         ftbar = yi[tt,:].mean()
         epsilon = yi[tt,ee] - ftbar
         ndata[tt,ee] = alpha * ftbar + beta * epsilon
    
    return ndata
    
# def rank_correlation_test(self,xi,yi,sig_level):
#   X = scipy.stats.stats.rankdata(xi)
#   Y = scipy.stats.stats.rankdata(yi)

#   cor,ista = self.correlation_test(X,Y, sig_level)

#   return cor,ista

  def rank_correlation_test(self,xi,yi,sig_level):
    numxi=len(xi)
    numyi=len(yi)
    if numxi != numyi:
      raise "error in rank_correlation_test: sample numbers are different"

    xi = np.array(xi)
    yi = np.array(yi)
    if np.count_nonzero(xi)==0 or np.count_nonzero(yi)==0:
       cor=np.nan
       ista=0
    else:
      #cor,pval = rank_corr_1dim(xi,yi)
       cor,pval = spearmanr(xi,yi)

       if abs(sig_level) >= abs(pval):
          ista=1  #棄却=相関は有意
       else:
          ista=0  #採択=相関は無意

    return cor,ista

  def partial_corr(self,C):
      """
      Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
      for the remaining variables in C.
      Parameters
      ----------
      C : array-like, shape (n, p)
          Array with the different variables. Each column of C is taken as a variable
      Returns
      -------
      P : array-like, shape (p, p)
          P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
          for the remaining variables in C.
      """
      from scipy import stats, linalg
      
      C = np.asarray(C)
      p = C.shape[1]
      P_corr = np.zeros((p, p), dtype=np.float)
      for i in range(p):
          P_corr[i, i] = 1
          for j in range(i+1, p):
              idx = np.ones(p, dtype=np.bool)
              idx[i] = False
              idx[j] = False
              beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
              beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
  
              res_j = C[:, j] - C[:, idx].dot( beta_i)
              res_i = C[:, i] - C[:, idx].dot(beta_j)
              
              corr = stats.pearsonr(res_i, res_j)[0]
              P_corr[i, j] = corr
              P_corr[j, i] = corr
          
      return P_corr

  def partial_rank_corr(self,C):
      """
      Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
      for the remaining variables in C.
      Parameters
      ----------
      C : array-like, shape (n, p)
          Array with the different variables. Each column of C is taken as a variable
      Returns
      -------
      P : array-like, shape (p, p)
          P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
          for the remaining variables in C.
      """
      from scipy import stats, linalg
      
      C = np.asarray(C)
      p = C.shape[1]
      P_corr = np.zeros((p, p), dtype=np.float)
      for i in range(p):
          P_corr[i, i] = 1
          for j in range(i+1, p):
              idx = np.ones(p, dtype=np.bool)
              idx[i] = False
              idx[j] = False
              beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
              beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
  
              res_j = C[:, j] - C[:, idx].dot( beta_i)
              res_i = C[:, i] - C[:, idx].dot(beta_j)
              
              corr, pval = self.rank_correlation_test_pval(res_i, res_j)
              P_corr[i, j] = corr
              P_corr[j, i] = corr
          
      return P_corr

  def correlation_test_pval_masked(self,xi,yi):

    #サンプル数
    numxi=len(xi)
    numyi=len(yi)

    if numxi != numyi:
      raise "error in correlation_test: sample numbers are different"

    xi2=[]
    yi2=[]
    for ii in range(numxi):
      if xi.mask[ii]==False and yi.mask[ii]==False:
         xi2.append(xi[ii])      
         yi2.append(yi[ii])      

    num=len(xi2)

    #相関係数
    cor,pval=st.pearsonr(xi2,yi2)

    return cor,pval

  def correlation_test_masked(self,xi,yi,sig_level):
    #sig_level=0.05 は両側有意水準5%
    #サンプル数
    numxi=len(xi)
    numyi=len(yi)

    if numxi != numyi:
      raise "error in correlation_test_masked: sample numbers are different"

    xi2=[]
    yi2=[]
    for ii in range(numxi):
      if xi.mask[ii]==False and yi.mask[ii]==False:
         xi2.append(xi[ii])      
         yi2.append(yi[ii])      

    num=len(xi2)
    #相関係数
   #cor=self.PearsonCorrelation(xi2,yi2)
    cor=float(np.ma.corrcoef(xi2,yi2)[0,1])
    #--t値の計算
    t0=(abs(cor) * sqrt(float(num-2)))/(sqrt(1.0-cor*cor))
    #--自由度num-1、有意水準sig_levelのt値の計算
    tg=self.tbunpu(num-2,sig_level/2.0) #tbunpuは片側検定のsig_levelを必要なので両側水準を２で割る
    #--統計的有意性の検定（t0とtgの比較)
    if abs(t0) >= abs(tg):
       ista=1  #棄却=相関は有意
    else:
       ista=0  #採択=相関は無意

    return cor,ista

  def tvalue(self,num,sig_level): #t分布の値 by Kudoh
    #sig_levelは片側 sig_level=0.005 は両側で90%に相当
    pi=3.1415926535897932385E0
    if sig_level < 0.0:
       sign=-1.0
    else:
       sign=1.0

    if (abs(sig_level) < 1.0e-10):
       level=sign * 1.0e-10
    else:
       level=sig_level

    t2=level*level
    x=t2/(t2+num)

    if ((num % 2) != 0 ):
        u=math.sqrt(x*(1.0-x))/pi
        p=1.0-2.0*math.atan2(math.sqrt(1.0-x),math.sqrt(x))/pi
        ia=1
    else:
        u=math.sqrt(x) * (1.0 -x)/2.0
        p=math.sqrt(x)
        ia=2

    if (ia != num):
        for i1 in range(ia,num-2+1,2):
          p=p+2.0 * u/i1
          u=u*(1.0+i1)/i1*(1.0-x)

    value=u/abs(level)
      
    return value

  def tbunpu(self,num,sig_level):
    value=0
    nk=35
    sn=5
    ptable=zeros((sn,), dtype=float)
    ftable=zeros((nk,), dtype=float)
    ttable=zeros((sn,nk), dtype=float)
    ptable[0:sn]=[0.1,0.05,0.025,0.01,0.005] #80%,90%,95%,98%,99%
    ftable[0:nk]=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000]
    ttable[0,0:nk]=[3.0777, 1.8856, 1.6377, 1.5332, 1.4759, 1.4398, 1.4149, 1.3968, 1.3830, 1.3722, 1.3634, 1.3562, 1.3502, 1.3450, 1.3406, 1.3368, 1.3334, 1.3304, 1.3277, 1.3253, 1.3212, 1.3178, 1.3150, 1.3125, 1.3104, 1.3031, 1.2987, 1.2958, 1.2938, 1.2922, 1.2910, 1.2901, 1.2893, 1.2886, 1.2816]
    ttable[1,0:nk]=[6.3138, 2.9200, 2.3534, 2.1318, 2.0150, 1.9432, 1.8946, 1.8595, 1.8331, 1.8125, 1.7959, 1.7823, 1.7709, 1.7613, 1.7531, 1.7459, 1.7396, 1.7341, 1.7291, 1.7247, 1.7171, 1.7109, 1.7056, 1.7011, 1.6973, 1.6839, 1.6759, 1.6706, 1.6669, 1.6641, 1.6620, 1.6602, 1.6588, 1.6577, 1.6449]
    ttable[2,0:nk]=[12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281, 2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0739, 2.0639, 2.0555, 2.0484, 2.0423, 2.0211, 2.0086, 2.0003, 1.9944, 1.9901, 1.9867, 1.9840, 1.9818, 1.9799, 1.9600]
    ttable[3,0:nk]=[31.8205, 6.9646, 4.5407, 3.7470, 3.3649, 3.1427, 2.9980, 2.8965, 2.8214, 2.7638, 2.7181, 2.6810, 2.6503, 2.6245, 2.6025, 2.5835, 2.5669, 2.5524, 2.5395, 2.5280, 2.5083, 2.4922, 2.4786, 2.4671, 2.4573, 2.4233, 2.4033, 2.3901, 2.3808, 2.3739, 2.3685, 2.3642, 2.3607, 2.3578, 2.3263]
    ttable[4,0:nk]=[63.6567, 9.9248, 5.8409, 4.6041, 4.0322, 3.7074, 3.4995, 3.3554, 3.2498, 3.1693, 3.1058, 3.0545, 3.0123, 2.9768, 2.9467, 2.9208, 2.8982, 2.8784, 2.8609, 2.8453, 2.8188, 2.7969, 2.7787, 2.7633, 2.7500, 2.7045, 2.6778, 2.6603, 2.6479, 2.6387, 2.6316, 2.6259, 2.6213, 2.6174, 2.5758]

    #--信頼区間のチェック
    tp=0
    for i in range(sn):
      if sig_level == ptable[i]:
        tp=i
        break

    if int(tp) == 0:
      print("信頼区間sig_levelは0.1,0.05,0.025,0.01,0.005を指定してください")
      raise

    #--t分布表を調べる 
    if num <= 20:
       tn=num -1
    elif num > 120:
       tn=nk-1
    else:
       for i in range(19,nk):
          if num >= ftable[i] and num <= ftable[i+1]:
            tn=i
            break

    value=ttable[tp,tn]  #ｔ分布表の値
    if num > 20 and num < 120:
       dy=float(ttable[tp,tn+1]) - float(ttable[tp,tn])
       dx=float(ftable[tn+1]) - float(ftable[tn])
       value = float(value) + (float(dy)/float(dx))*(float(num) - float(ftable[tn]))

    return value

  def var(self, xi):  #不偏分散
   #o_var=r.var([float(x) for x in xi])
    o_var=np.array(xi).var()
   #return o_var[0]
    return o_var

  def sd(self,xi):
    o_sd=r.sd([float(x) for x in xi])
    return o_sd[0]

  def mean(self,xi):
   #o_mean=r.mean([float(x) for x in xi])
   #data = np.array(xi)
    o_mean = np.array(xi).mean()
   #return o_mean[0]
    return o_mean

  def mean2(self,xi):
    rsum=0.0
    for i in range(len(xi)):
      rsum=float(rsum) + float(xi[i])

    o_mean=float(rsum)/float(len(xi))
    return o_mean
   
  def ttest(self,xi,yi):
    kekka=r.t_test(xi,yi)
    return kekka

  def simple_lm(self,xi,yi):
    xlen=len(xi)
    ylen=len(yi)
    if xlen != ylen or xlen == 0:
            print("lm:error:xlen and ylen mismatch.") 
            print("          xlen=",xlen)
            print("          ylen=",ylen)
            raise
   #xx,yyの作成
    xx=[]
    yy=[]

    for i in range(xlen):
      xx.append(float(xi[i]))
      yy.append(float(yi[i]))

    x_list_R=robjects.FloatVector(xi)
    y_list_R=robjects.FloatVector(yi)
    robjects.globalenv['x_list'] = x_list_R
    robjects.globalenv['y_list'] = y_list_R
    r=robjects.r
    lm=r.lm('y_list ~ x_list')

    slope=lm.rx2('coefficients').rx2('x_list')[0]
    seppen=lm.rx2('coefficients').rx2('(Intercept)')[0]
    spearman_r = r.cor(x_list_R, y_list_R, method='spearman')[0] #Spearman correlation coefficient
   #print "slope",slope
   #print "seppen=",seppen
   #print "spearman_r=",spearman_r

    return  slope,seppen,spearman_r

 #def aov(self,xi,cate1,cate2):
  def aov(self):
    r = robjects.r
    ctl = robjects.FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
    trt = robjects.FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])

    group = r.gl(2, 10, 20, labels = ["Ctl","Trt"])
    return 

  def mk_test(self,x, alpha = 0.05):
    """
    this perform the MK (Mann-Kendall) test to check if the trend is present in 
    data or not
    
    Input:
        x:   a vector of data
        alpha: significance level
    
    Output:
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the sifnificance test
        
    Examples
    --------
      >>> x = np.random.rand(100)
      >>> h,p = mk_test(x,0.05)  # meteo.dat comma delimited
    """
    n = len(x)
    
    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])
    
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    
    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)
    
    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2) 
    
    return h, p

  def linreg(self,X,Y):
    """
     return a,b in solution to y = ax + b such that root mean square distance 
     between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in map(None, X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    a =  (Sxy * N - Sy * Sx)/det
    b =  (Sxx * Sy - Sx * Sxy)/det
    return a,b

  def runningmean(self,X,wind,opt="undef"):
   #in : X(N,) First one dimensional input array
   #     wind(M,) one dimensional input array
   #opt: same : Mode same returns output of length max(M, N)
   #     valid: returns output of length M,N - min(M, N)
      b = np.ones(wind)/float(wind)
      out = np.ma.zeros(len(X)) #initialization
      undef=-9.99E33

      if opt == "undef":
        a_ave = np.convolve(X, b, 'same')
        out[:] = a_ave[:]
        out[0:wind/2] = undef
        out[-wind/2+1:] = undef
        out = np.ma.masked_where(out==undef,out)
      else:
        out= np.convolve(X, b, opt)
        
      return out

  def normalize(self,X,opt=1):
      xmean=X.mean() 
      xvar=X.std() 
      out = (X - xmean)/float(xvar)
      return out

  def last_consecutive_true(self,bool_data, num_of_series):
       """
        return last element which experienced consecutive [num_of_series] True 
        bool_data=[False, True,  True, False, True, True,  True, False, True], num_of_series=3 => 6
        bool_data=[False, True,  True, False, True, True, False, False, True], num_of_series=3 => False
       """
       cnum = np.zeros(len(bool_data))
  
       for ii in range(len(bool_data)):
           if ii==0 and bool_data[ii]:
             cnum[ii]=1
           elif bool_data[ii]:
             cnum[ii]=cnum[ii-1]+1
  
       temp = np.where(cnum>=num_of_series)[0]
       if len(temp) == 0:
         out = False
       else:
         out = np.where(cnum>=num_of_series)[0][-1]
       return out
  
  def first_consecutive_true(self, bool_data, num_of_series):
       """
        return first element which experienced consecutive [num_of_series] True 
        bool_data=[False, True,  True, False, True, True,  True, False, True], num_of_series=3 => 4
        bool_data=[False, True,  True, False, True, True, False, False, True], num_of_series=3 => False
       """
       cnum = np.zeros(len(bool_data))
  
       for ii in range(len(bool_data)-1,-1,-1):
           if ii==len(bool_data)-1 and bool_data[ii]:
             cnum[ii]=1
           elif bool_data[ii]:
             cnum[ii]=cnum[ii+1]+1
  
       temp = np.where(cnum>=num_of_series)[0]
       if len(temp) == 0:
         out = False
       else:
         out = np.where(cnum>=num_of_series)[0][0]
       return out

  def lagcorr(self, x,y,lag=None,verbose=True):
      '''Compute lead-lag correlations between 2 time series.
  
      <x>,<y>: 1-D time series.
      <lag>: lag option, could take different forms of <lag>:
            if 0 or None, compute ordinary correlation and p-value;
            if positive integer, compute lagged correlation with lag
            upto <lag>;
            if negative integer, compute lead correlation with lead
            upto <-lag>;
            if pass in an list or tuple or array of integers, compute 
            lead/lag correlations at different leads/lags.
  
      Note: when talking about lead/lag, uses <y> as a reference.
      Therefore positive lag means <x> lags <y> by <lag>, computation is
      done by shifting <x> to the left hand side by <lag> with respect to
      <y>.
      Similarly negative lag means <x> leads <y> by <lag>, computation is
      done by shifting <x> to the right hand side by <lag> with respect to
      <y>.
  
      Return <result>: a (n*2) array, with 1st column the correlation 
      coefficients, 2nd column correpsonding p values.
  
      Currently only works for 1-D arrays.
      '''
  
      import numpy
      from scipy.stats import pearsonr
  
      if len(x)!=len(y):
          raise('Input variables of different lengths.')
  
      #--------Unify types of <lag>-------------
      if numpy.isscalar(lag):
          if abs(lag)>=len(x):
              raise('Maximum lag equal or larger than array.')
          if lag<0:
              lag=-numpy.arange(abs(lag)+1)
          elif lag==0:
              lag=[0,]
          else:
              lag=numpy.arange(lag+1)    
      elif lag is None:
          lag=[0,]
      else:
          lag=numpy.asarray(lag)
  
      #-------Loop over lags---------------------
      result=[]
      if verbose:
          print('\n#<lagcorr>: Computing lagged-correlations at lags:',lag)
  
      for ii in lag:
          if ii<0:
              result.append(pearsonr(x[:ii],y[-ii:]))
          elif ii==0:
              result.append(pearsonr(x,y))
          elif ii>0:
              result.append(pearsonr(x[ii:],y[:-ii]))
  
      result=numpy.asarray(result)
  
      return result

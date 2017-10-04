import zfits
import math
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
np.set_printoptions(threshold=np.nan)


class photon:
    def __init__(self, calib, init):
        '''
        Input Paramerters:
        calib: Path to calibration data (drs.fits) [String]
        init: Path to data (fits.fz) [String]
        '''
        self.calib = calib
        self.init = init
        self.data=[]
        self.processed=False
        self.binneddata=[]
        self.eventpos=[]
        self.slicepos=[]
        self.int_values=[]

        
    def load_data2(self,out_path,channels,events=50):
        '''
        Use for large data arrays.
        out_path: Path to output .txt file [str]
        channels: The channels one wants to read in [array]
        events: The number of events one wants to read in [int]
        '''
        datap=zfits.FactFitsCalib(calib_path=self.calib,data_path=self.init)
        data = []
        print("Step 1: Convertion...")
        for i, event in tqdm.tqdm(enumerate(datap),total=events):
            if i > events:
                break
            data.append(event['CalibData'].copy())
        print("Step 1 complete")
        print("Step 2: Writing to file...")
        file=open(out_path,"w")
    
        for line_id, dat in tqdm.tqdm(enumerate(data),total=events):    
            file.write("Data of Event "+str(line_id)+"\n"+"\n")
            for channel in channels:
                file.write("Channel "+str(channel)+"\n")
                file.write(str(dat[channel])+"\n"+"\n")       
        file.close()
        print("Step 2 complete")
    
    def load_data(self,events=50):
        '''events: Number of events to read in [Int]'''
        dat=zfits.FactFitsCalib(calib_path=self.calib,data_path=self.init)
        for i, event in enumerate(dat):
            if i > events:
                break
            self.data.append(event['CalibData'].copy())
            
    def show(self,channels=[1],flanks=False):
        '''
        channels: The number of channels one wants to display [Int Array]
        flanks: If True flanks are indicated and binned data is shown
        '''
        if flanks==False:
            for line_id,dat in enumerate(self.data):
                plt.figure(line_id,figsize=(14, 6))
                for channel in channels:
                    plt.plot(dat[channel], '.:', label=str(channel))
                plt.grid()
                plt.xlabel("Time Slice")
                plt.ylabel("Signal Strength")
                plt.legend()
                plt.title("Raw Data, Event "+str(line_id))
                plt.show()
        if flanks==True:
            if self.processed==True:
                for line_id in range(self.binneddata.shape[0]):
                    plt.figure(line_id,figsize=(14, 6))
                    plt.plot(self.binneddata[line_id], '.:')
                    bar=[]
                    for i in range(self.eventpos.shape[0]):
                        if self.eventpos[i]==line_id:
                            if self.slicepos[i]>0:
                                bar=np.append(bar,self.slicepos[i])
                    for xcoord in bar:
                        plt.axvline(x=xcoord,c='r')
                    plt.grid()
                    plt.legend()
                    plt.title("Binned signal with flanks indicated, Event "+str(line_id))
            else:
                print("Error: Please process data first before setting flanks=True !!!")

        
    def process(self,channel,cut=20,binsize=5,derivativelimit=3,gain=5,deadtime=5,maxsearch=3,length=5,limit=2,interpol=16,integrate=6,showevent=-1):
        '''
        channel: The channel one wants to evaluate [Int]
        cut: Number of slices cut of at beginning and end of each timeline [Int]
        binsize: The number of slices put into one bin [Int]
        derivativelimit: Minimal slope a flank needs to have in order to be choosen [Float]
        gain: Minimal hight a flank needs to have in order to be choosen [Float]
        deadtime: Number of bins after a flank detection, where no new flank can be detected [Int]
        maxsearch: Number of bins after a flank that are used to search for a maximum [Int]
        length: Minimal number of bins a signal needs to stay above limit [Int]
        limit: Determines lower limit that a signal has to surpass by (mean value of flank)/limit [Int]
        interpol: Number of bins used for interpolation [Int]
        integrate: Number of bins used for integration [Int]
        showevent: Event chosen to demonstrate integration procedure [Int]. If negative: No event chosen.
        '''
        cleandata=np.zeros([len(self.data),1024-2*cut])
        for line_id, dat in enumerate(self.data):
            cleandata[line_id]=(dat[channel])[cut:-cut]
            
        binnum=int(cleandata.shape[1]/binsize)
        self.binneddata=np.zeros((cleandata.shape[0],binnum))
        for event in range(cleandata.shape[0]):
            for i in range(binnum-1):
                self.binneddata[event,i]=np.average(cleandata[event,i*binsize:(i+1)*binsize])
                
        dcleandata=np.zeros([self.binneddata.shape[0],self.binneddata.shape[1]-1])
        for i in range(self.binneddata.shape[0]):
            for j in range(self.binneddata.shape[1]-1):
                dcleandata[i,j]=self.binneddata[i,j+1]-self.binneddata[i,j]  
                
        dlarge=dcleandata > derivativelimit
        for line in range(dlarge.shape[0]):
            for entry in range(dlarge.shape[1]):
                if dlarge[line,entry]==True:
                    min=self.binneddata[line,entry]
                    max=min
                    for i in range(5):
                        if entry-i==0:
                            break
                        if self.binneddata[line,entry-i] <= min:
                            min=self.binneddata[line,entry-i]
                    for i in range(10):
                        if i+entry==self.binneddata.shape[1]-1:
                            break
                        if self.binneddata[line,entry+i] >= max:
                            max=self.binneddata[line,entry+i]
                    if abs(max-min)<gain:
                        dlarge[line,entry]=False
                
        for line,l in enumerate(dlarge):
            for i in range(0,l.shape[0]-1):
                if i==len(l):
                    break        
                if l[i]==True:                
                    l[i+1:i+deadtime]=False 
        self.eventpos,self.slicepos=np.nonzero(dlarge) 
        
        arrivaltimes=np.zeros(self.slicepos.shape[0],dtype=int)
        for line_id in range(self.binneddata.shape[0]):
            for i in range(self.eventpos.shape[0]):
                if self.eventpos[i]==line_id:
                    check=False
                    min=self.binneddata[line_id,self.slicepos[i]]
                    max=min
                    for j in range(maxsearch):
                        if j+self.slicepos[i]>=self.binneddata.shape[1]-(maxsearch+1):
                            arrivaltimes[i]=-1000
                            check=True
                            break
                        if self.binneddata[line_id,self.slicepos[i]+j] >= max:
                            max=self.binneddata[line_id,self.slicepos[i]+j]
                    if check==False:
                        middle=(max-min)/2+min
                        idx = (np.abs(self.binneddata[line_id,self.slicepos[i]:self.slicepos[i]+maxsearch]-middle)).argmin()
                        arrivaltimes[i]=idx+self.slicepos[i]
                        if np.any(self.binneddata[line_id,arrivaltimes[i]+1:arrivaltimes[i]+length] < middle/limit):
                            arrivaltimes[i]=-1000

        todel=np.zeros(0,dtype=int)
        for i in range(arrivaltimes.shape[0]):
            if arrivaltimes[i] < 0:
                todel=np.append(todel,i)
        arrivaltimes=np.delete(arrivaltimes,todel)
        self.slicepos=np.delete(self.slicepos,todel)
        self.eventpos=np.delete(self.eventpos,todel)
        
        showmin1=0
        showmin2=0
        self.int_values=np.zeros(self.eventpos.shape[0])
        for event in range(self.eventpos.shape[0]):
            if self.slicepos[event]<self.binneddata.shape[1]-interpol:
                min1=self.binneddata[self.eventpos[event],self.slicepos[event]]
                min2=self.binneddata[self.eventpos[event],self.slicepos[event]+interpol]
                step=(min2-min1)/interpol
                integral=0
                for j in range(integrate+arrivaltimes[event]-self.slicepos[event]):
                    integral+=self.binneddata[self.eventpos[event],self.slicepos[event]+j]-(j*step+min1)
                self.int_values[event]=integral
                if showevent>0:
                    if event==showevent:
                        showmin1=min1
                        showmin2=min2
        self.int_values=self.int_values[self.int_values>0]
        
        if showevent > 0 :
            plt.figure(figsize=(14, 6))
            plt.plot(self.binneddata[self.eventpos[showevent]])
            plt.axvline(self.slicepos[showevent],c='r')
            plt.axvline(6+arrivaltimes[showevent],c='r')
            plt.plot([self.slicepos[showevent],self.slicepos[showevent]+16],[showmin1,showmin2],c='y')
        
        self.processed=True
        
    def crosstalk(self,numbin=50,log=True,normed=False,hi=3,fu=2):
        '''
        numbin: Number of bins used in histogram [Int]
        log: If True the histogram is given in log scale
        normed: If True the histogram is normalised to 1
        hi: Derivative limit for search of first minimum
        fu: Derivative limit for search of second minimum
        '''
        if self.processed==True:
            #Fingerplot
            numbin=100
            plt.figure(figsize=(14, 6))
            n, bins, patches = plt.hist(self.int_values, numbin, facecolor='blue', alpha=0.5,log=True,normed=False)
            der=np.zeros(n.shape[0]-1)
            for i in range(n.shape[0]-1):
                der[i]=n[i+1]-n[i]

            max1p=np.argmax(n)
            tet=-1
            i=0
            while tet<=hi: #Arbitrary (choose such that 2nd max is chosen correctly)
                tet=der[max1p+i]
                i+=1
            min1p=max1p+i

            max2p=np.argmax(n[min1p:])+min1p
            tet=-1
            i=0
            while tet<=fu: #Arbitrary (choose such that 2nd max is chosen correctly)
                tet=der[max2p+i]
                i+=1
            min2p=max2p+i

            max1=bins[max1p]
            min1=bins[min1p]
            max2=bins[max2p]
            min2=bins[min2p]

            plt.axvline(bins[max1p])
            plt.axvline(bins[max2p])
            plt.axvline(bins[min1p],c='y')
            plt.axvline(bins[min2p],c='y')

            #Gaussian Fitting
            bin_centres = (bins[:-1] + bins[1:])/2
            first=n[:min1p]
            second=n[min1p:min2p]

            def gauss(x, *p):
                '''Gaussian function used for fitting'''
                A, mu, sigma = p
                return A*np.exp(-(x-mu)**2/(2.*sigma**2))

            # Initial guess of the coefficients as requiered by curve_fit
            p0 = [1., bin_centres[min1p], 1.]

            coeff1, var_matrix1 = curve_fit(gauss, bin_centres[:min1p], first, p0=p0)
            coeff2, var_matrix2 = curve_fit(gauss, bin_centres[min1p:min2p], second, p0=p0)
            # Get the fitted curve
            hist_fit1 = gauss(bin_centres[:min1p], *coeff1)
            hist_fit2 = gauss(bin_centres[min1p:min2p], *coeff2)

            plt.plot(bin_centres[:min1p],hist_fit1,c='r')
            plt.plot(bin_centres[min1p:min2p],hist_fit2,c='k')
            plt.xlabel("p.e.")
            plt.ylabel("# of events")
            norm1,mean1,std1=coeff1
            norm2,mean2,std2=coeff2

            cross_mean2=gauss(mean2, *coeff2)
            cross_mean1=gauss(mean1, *coeff1)
            cross_mean=cross_mean2/cross_mean1

            cross_err=cross_mean*(abs(gauss(mean1+std1,*coeff1)-cross_mean1)/cross_mean1+abs(gauss(mean2+std2,*coeff2)-cross_mean2)/cross_mean2)

            print("1 Photon events: "+str(cross_mean1))
            print("2 Photon events: "+str(cross_mean2))
            print("Crosstalk Probability: "+str(cross_mean*100)+" +/-"+str(cross_err*100)+" % ")
            

        else:
            print("Error: Please process data first before calling crosstalk() !!!")
            
    
        
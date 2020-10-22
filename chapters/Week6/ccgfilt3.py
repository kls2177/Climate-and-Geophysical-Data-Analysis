"""
Class for computing the curve fitting/smoothing technique used by Thoning et al
1989.

This technique uses the following step:

   1 - Fit a function consisting of a polynomial and harmonics to the data
   2 - Smooth the residuals from the function fit with a low-pass filter using
       fft and user defined cutoff value.
   3 - Calculate the inverse fft of the low-pass filter to get smoothed data in
       time domain.
   4 - Determine the smoothed curve of interest by combining the function with
       the filtered data.

The function to be fit to the data is specified in the routines 'fitFunc' and
'harmonics'.
"""

import sys
import datetime

from math import pi, sqrt, atan2
from scipy import optimize
from scipy import stats
from scipy import interpolate
from scipy import fftpack
import numpy
numpy.seterr(over='ignore')


# --------------------------------------------------
# Define the function we are trying to fit
# This is a combination of a polynomial and harmonic function
# --------------------------------------------------
def fitFunc(params, x, numpoly, numharm):
    """ Calculate the function at time x with coefficients given in params.
    This is a combination of a polynomial with numpoly coefficients, and
    a sin/cosine harmonic with numharm coefficients.
    e.g., with numpoly=3 and numharm=2:

        y = a + b*x + c*x^2 + d*sin(2*pi*x) + e*cos(2*pi*x) + f*sin(2*pi*x*2) + g*cos(2*pu*x*2)

    where a = params[0], b = params[1], c = params[2], d = params[3] ...
    """

    # polynomial part
    # we need to reverse the order of the polynomial coefficients for input into polyval
    p = numpy.polyval(params[numpoly-1::-1], x)

    # get harmonic part of function
    s = harmonics(params, x, numpoly, numharm)

    return p+s

# --------------------------------------------------


def harmonics(params, x, numpoly, numharm):

    """ calculate the harmonic part of the function at time x """

    # harmonic part
    pi2 = 2*pi*x
    if numharm > 0:
        # create an array s of correct size by explicitly evaluating first harmonic
        s = params[numpoly]*numpy.sin(pi2) + params[numpoly+1]*numpy.cos(pi2)

        # do additional harmonics (nharm > 1)
        for i in range(1, numharm):
            ix = 2*i + numpoly    # index into params for harmonic coefficients
            s += params[ix]*numpy.sin((i+1)*pi2) + params[ix+1]*numpy.cos((i+1)*pi2)

        n = numpoly + 2*numharm
        s = (1+params[n]*x) * s        # amplitude gain factor

        return s
    else:
        return 0

# --------------------------------------------------
# function to calc the difference between input values and function


def errfunc(p, x, y, numpoly, numharm):
    return y - fitFunc(p, x, numpoly, numharm)

# --------------------------------------------------


class ccgFilter:
    """
    Input Parameters
    ----------
    xp : numpy array
    time values for input data
    yp : numpy array
    dependent values for input data
    shortterm : int
    Short term cutoff value in days for smoothing of data
        Optional. Default is 80
    longterm : int
    Long term cutoff value in days for extracting trend from data
        Optional. Default is 667
    sampleinterval : int
    Interval in days between samples, calculate equally spaced values at this interval
        Optional. Default is calculated from xp
    numpoly : int
    Number of polynomial terms used in function fit - e.g. 3 = quadratic
        Optional.  Default is 3
    numharm : int
    Number of harmonics used in function fit
        Optional.  Default is 4
    timezero : float
    Value where x = 0 in the function coefficients
        Optional.  Default is 0
    gap : float
    When determining equally spaced values for the fft,
    if gap != 0, then gap is the number of days between samples that should
    be filled in with values from the function, rather than linear interpolated.
        Optional.  Default is 0
    debug: boolean
    If true, print out extra information during calculations.
        Optional.  Default is false


    Attributes
    ----------
      Input Data
    xp : numpy array
        Time value for input data
    yp : numpy array
        Dependent values for input data
    np : int
        Number of points in xp, yp
    xinterp : numpy array
        Equally spaced interpolated values from input data

      For the function fit
    numpoly : int
        Number of polynomial terms used in function fit - e.g. 3 = quadratic
    numharm : int
        Number of harmonics used in function fit
    timezero : float
        Value where x = 0 in the function coefficients
    params : numpy array
        Parameters (coefficients) for the function fit
    covar : numpy array
        Covariance values of the parameters
    numpm : int
        Total number of parameters in the function
    resid : numpy array
        Residuals from function fit for times specified in input array xp
    yinterp : numpy array
        Equally spaced interpolated values of the residuals from the functions fit
            for times specified in array xinterp

      For the filter
    sampleinterval : int
        Interval in days between equally spaced points used in the fft
    dinterval : float
        Sample interval in decimal years
    shortterm : int
        Short term cutoff value in days for smoothing of data
    longterm : int
        Long term cutoff value in days for extracting trend from data
    smooth : numpy array
        smoothed results from applying short term cutoff filter to residuals of data from the function.
            Equally spaced at xinterp
    trend : numpy array
        trend results from applying long term cutoff filter to residuals of data from the function.
        Equally spaced at xinterp
    deriv : numpy array
        derivative of function + trend.  Equally spaced at xinterp
    ninterp : int
        number of points in each of xinterp, smooth, trend

      Misc.
    rsd1 : float
            Standard deviation of residuals about function
    rsd2 : float
            Standard deviation of residuals about smooth curve
    debug : boolean
        Flag for showing additional information during computation

    Methods
    -------
      For each of the methods below, the input value x can be a single point, a list, or a numpy array

      getFunctionValue(x)
    Returns the value of the function part of the filter at time x.

      getSmoothValue(x)
    Returns the 'smoothed' data at time x.  This is function + self.smooth

      getTrendValue(x)
    Returns the 'trend' of the data at time x.  This is polynomial part of function + self.trend

      getHarmonicValue(x)
    Returns the value of the harmonic part of the function at time x.

      getPolyValue(x)
    Returns the value of the polynomial part of the function at time x

      getAmplitudes()
    Get seasonal cycle amplitudes
    Returns a list of tuples, each tuple has 6 values (year, total_amplitude, max_date, max_value, min_date, min_value)


      Additional methods:

        getFilterResponse(cutoff)
          Returns the value of the filter for frequencies 0 - 10 cycles/year at given cutoff

        getMonthlyMeans()
          Return of list of tuples containing monthy means from the smoothed curve.
      The value of the curve is computed at every sample interval, then summed up for each
          month and the average computed.

        getTrendCrossingDates()
      Get the dates when the smoothed curve crosses the trend curve.
      That is, when the detrended smooth seasonal cycle crosses 0.

    """

    def __init__(self, xp, yp, shortterm=80, longterm=667, sampleinterval=0, numpolyterms=3, numharmonics=4, timezero=-1, gap=0, debug=False):
        t0 = datetime.datetime.now()
        # save input data as numpy arrays
        self.xp = numpy.array(xp)
        self.yp = numpy.array(yp)
        self.np = len(xp)
        # Calculate the average time interval between data points.
        # Set the sampleinterval variable if not set on the command line.
        if sampleinterval == 0:
            # calculate the average interval between samples that are at least 1 day apart
            sd = 0
            sdiff = 0
            tx = self.xp[0]
            for i in range(1, self.np):
                if self.xp[i]-tx > 0.002739:
                    diff = self.xp[i]-tx
                    sdiff += diff
                    sd += 1
                tx = self.xp[i]

            avginterval = sdiff/sd * 365

            if avginterval > 1:
                self.sampleinterval = round(avginterval, 0)
            else:
                self.sampleinterval = avginterval
            if debug:
                print("changed sampleinterval to {}".format(self.sampleinterval))

        else:
            self.sampleinterval = sampleinterval

        self.dinterval = self.sampleinterval/365.0  # sample interval in decimal years

        # If the data is actually an average over a relatively large time period,
        # such as annual averages, change the number of harmonics to an appropriate value.
        nh = int(365.0/(self.sampleinterval*2))
        if nh < numharmonics:
            self.numharm = nh
            if debug:
                print("changed numharmonics to {}".format(nh))
        else:
            self.numharm = numharmonics

        self.shortterm = shortterm
        self.longterm = longterm
        self.numpoly = numpolyterms
        if timezero < 0:
            self.timezero = int(xp[0])
            if debug:
                print("changed timezero to {}".format(self.timezero))
        else:
            self.timezero = timezero
        self.debug = debug
        self.numpm = self.numpoly + 2*self.numharm

        # apply filter to data
        self._filter_data(gap)

        # compute derivatives of polynomial and long term trend
        self._compute_deriv()

        # standard deviation of residuals about smooth curve
        r = self.yp - self.getSmoothValue(self.xp)
        # rmean = numpy.mean(r)
        self.rsd2 = numpy.std(r, ddof=1)
        if self.debug:
            print("rsd about smooth curve is {}".format(self.rsd2))

        t1 = datetime.datetime.now()
        if self.debug:
            print("Total time elapsed: {}".format(t1-t0))

    # ------------------------------------------------------------

    def _filter_data(self, gap):

        if self.debug:
            print("=== Inside filter_data. ===")
            print("  Number of points = {a}, Sample Interval = {b} days, {c} years".format(a=self.np, b=self.sampleinterval, c=self.dinterval))
            print("  Cutoff 1 = {}, Cutoff 2 = {}".format(self.shortterm, self.longterm))
            print("  Numpoly = {}, Numharm = {}".format(self.numpoly,self.numharm))
            print("  First point = {},{}".format(self.xp[0], self.yp[0]))
            print("  Last point = {},{}".format(self.xp[-1], self.yp[-1]))

        # Remove the timezero value from the x data so that coefficients will be relative to the timezero date
        work = self.xp - self.timezero

        # Fit the function to the data
        pm = [1] * self.numpm        # initial parameter values set to 1
        pm.append(0)
        self.numpm += 1
        self.params, self.covar, info, mesg, ier = optimize.leastsq(errfunc, pm, full_output=1, args=(work, self.yp, self.numpoly, self.numharm))
        if self.debug:
            print("  Finished leastsq")
            for i in range(self.numpm):
                print("    param[{}] = {}".format(i, self.params[i]))

        #  calculate residuals from fit
        resid = self.yp - fitFunc(self.params, work, self.numpoly, self.numharm)
        rmean = numpy.mean(resid)
        self.rsd1 = numpy.std(resid, ddof=1)
        if self.debug:
            print("  Finished residuals")
            print("    rmean = {}, rsd = {}".format(rmean, self.rsd1))

        # fit linear line to ends of residual data
        # subtract this from residuals so ends are ~ near 0
        ca, cb = self._adjustend(work, resid, self.longterm)
        resid -= (ca + cb*work)
        if self.debug:
            print("  Finished adjustend")
            print("    ca = {}, cb = {}".format(ca, cb))
            print("    x[0] = {}, x[{}] = {}".format(work[0], self.np, work[-1]))
            print("    resid[0] = {}, resid[{}] = {}".format(resid[0], self.np, resid[-1]))

        # Interpolate data at evenly spaced intervals (self.sampleinterval)
        self.xinterp, yinterp = self._lin_interp(work, resid, gap)
        self.ninterp = len(self.xinterp)

        # add linear fit and timezero back in to residuals from function
        self.resid = resid + ca + cb*work
        if self.debug:
            print("  Interpolated points.")
            print("    Number of interpolated points: {}".format(self.ninterp))
            print("    xinterp[np-1] = {}, x[0] = {}".format(self.xinterp[-1], self.xinterp[0]))
            print("    yinterp[np-1] = {}, y[0] = {}".format(yinterp[-1], yinterp[0]))

        # do fft on interpolated data
        fft = fftpack.rfft(yinterp)

        # do short term filter
        if self.debug:
            print("  Do short term filter, cutoff = {}".format(self.shortterm))
        a = self._freq_filter(fft, self.dinterval, self.shortterm)
        yfilt = fftpack.irfft(a)
        self.smooth = yfilt + ca + cb*self.xinterp

        # do long term filter
        if self.debug:
            print("  Do long term filter, cutoff = {}".format(self.longterm))
        a = self._freq_filter(fft, self.dinterval, self.longterm)
        yfilt = fftpack.irfft(a)
        self.trend = yfilt + ca + cb*self.xinterp

        # add linear fit and timezero back in to interpolated values
        self.yinterp = yinterp + ca + cb*self.xinterp
        self.xinterp = self.xinterp + self.timezero


    # ------------------------------------------------------------
    def _adjustend(self, x, y, cutoff):
        """ Determine the slope of the data based on just the ends, i.e. 1/4 of the cutoff """

        # check if length of data is too short to adjust
        if x[-1] - x[0] < cutoff/365.0:
            return 0, 0

        # length of data to use is 1/4 of cutoff length
        c = cutoff/365.0/4.0

        z = numpy.where((x <= x[0] + c) | (x >= x[-1] - c))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x[z], y[z])
        return intercept, slope

    #------------------------------------------------------------
    def _lin_interp(self, x, y, gap):
        """ Linear interpolate between input data to get equally spaced values
        at every sample interval.
        """

        # calculate the x values for evenly spaced data at the specified sampling interval
        xi = numpy.arange(x[0], x[-1]+self.dinterval/2, self.dinterval)
        xi[-1] = x[-1]        # make sure last point is equal to last data point

        # if there are multiple y data points at a single x value, then average them
        # to get only 1 y data point for each x
        xx = []
        yy = []
        xt = x[0]
        ys = y[0]
        ns = 1
        for xp, yp in zip(x[1:], y[1:]):
            if xp == xt:
                ys += yp
                ns += 1
            else:
                ya = ys/ns
                xx.append(xt)
                yy.append(ya)
                ys = yp
                ns = 1
            xt = xp
        ya = ys/ns
        xx.append(xt)
        yy.append(ya)

        # calculate interpolation values at each x point
        f = interpolate.interp1d(xx, yy)

        # if a gap setting was not made, use normal linear interpolation
        # to get equally space points.
        # Otherwise, fill in gaps using the function value (0) instead.
        if gap == 0:
            yi = f(xi)

        else:
            n = len(xx)
            ni = len(xi)
            yi = numpy.zeros((ni))
            j = 0
            for i in range(ni):
                while xi[i] >= xx[j]:
                    j += 1
                    if j >= n-1:
                        break

                j -= 1
                if (xx[j+1] - xx[j]) > gap/365.0: # 8*self.dinterval:
                    yi[i] = 0
                else:
                    yi[i] = f(xi[i])


        return xi, yi

    #------------------------------------------------------------
    def _freq_filter(self, fft, dinterv, cutoff):
        """ Apply low-pass filter to fft data.
        Multiply each discrete frequency in fft by a
        low-pass filter function value set by the value 'cutoff'
        input:
            fft - results of fft
            dinterv - sampling interval in years
            cutoff - cutoff value in days
        """

        n2 = len(fft)
        cf = cutoff/365.0    # convert cutoff to years
        cutoff2 = 1.0/cf    # change to cycles/year

        freq = fftpack.rfftfreq(n2, dinterv)    # get array of frequencies
        rw = self._vfilt(freq, cutoff2, 6)    # get filter value at frequencies
        filt = fft*rw                # apply filter values to fft


        return filt

# this is how to do filtering at each frequency
#    filt = numpy.zeros( (len(fft)) )
#    b = 1.0/(n2*dinterv)
#    for i in range(1, n2-1, 2):
#        freq = (i+1)/2.0 * b
#        rw = self._fnfilt(freq, cutoff2, 6)
#        filt[i] = fft[i]*rw
#        filt[i+1] = fft[i+1]*rw
#
#    dmaxfreq = 1.0/(2*dinterv)
#    rw = self._fnfilt(dmaxfreq, cutoff2, 6)
#    filt[-1] = fft[-1]*rw
#    filt[0] = fft[0]
#    def _fnfilt(self, freq, sigma, power):
#    """ Determine filter value at freq """
#
#    z = pow((freq/sigma), power)
#    if z > 20.0:
#        f = 1e-10
#    else:
#        f = 1.0 / pow(2.0, z)
#
#    return f
#
#    return filt

    #------------------------------------------------------------
    def _vfilt(self, freq, sigma, power):
        """ vectorized version of getting filter value at a frequency
        input:
            freq - array of frequencies
            sigma - cutoff value in cycles/year
            power - integer value for f/fc^power
        returns:
            array of filter values at each frequency

        the clip statement insures against underflow, although
        numpy seems to handle it internally anyway
        """
        z = numpy.power((freq/sigma), power)
#        z = numpy.clip(z, 0, 20.0)
        f = 1.0 / numpy.power(2.0, z)
        return f


    #------------------------------------------------------------
    def _compute_deriv(self):
        """ Compute derivative of trend.
        This is the derivative of self.trend + derivative of polynomial part of the function
        """

        # Connect trend data points with spline to get derivative at each point
        tck = interpolate.splrep(self.xinterp, self.trend, s=0.0)
        self.deriv = interpolate.splev(self.xinterp, tck, der=1)

        # compute derivative of polynomial at each interpolated data point
        # we need to reverse order of polynomial coefficients for input into poly1d
        poly = numpy.poly1d(self.params[self.numpoly-1::-1])
        pd = numpy.polyder(poly)
        self.deriv += pd(self.xinterp - self.timezero)

    #------------------------------------------------------------
    def stats(self):

        stats = ""
        if self.np == 0:
                stats += "No data points.  No Statistics available."
                return;

        # GetCalendarDate(self.timezero, &year, &month, &day, &hour, &minute, &second);

        stats += "*****  Filter Statistics.  *****\n"

        stats += "Beginning date:                 %.6f\n" % self.xp[0]
        stats += "Ending date:                    %.6f\n" % self.xp[self.np-1]

        stats += "Number of data points:          %d\n\n" % self.np

        stats += "FUNCTION PARAMETERS\n"
#        stats += "Time = 0 on %d/%d/%d\n"        ,  year, month, day
        stats += "Time = 0 on %f\n" % self.timezero  # year, month, day
        stats += "Number of polynomial terms:     %d\n" % self.numpoly
        stats += "Number of harmonic terms:       %d\n" % self.numharm
        stats += "Total Number of parameters:     %d\n" % self.numpm
        stats += "------------------------------------------------------\n"
        stats += "Parameter          Value          Standard Deviation\n"
        stats += " Polynomial\n"
        for i in range(self.numpm):
            if i == self.numpoly: stats += " Harmonics\n"
            if i == self.numpoly + 2*self.numharm:
                stats += " Amplitude Gain Factor\n"
                stats += "%5d %20.6f %20.6f\n" % (i, self.params[i], sqrt(self.covar[i][i]))

            stats += "------------------------------------------------------\n"
            stats += "Harmonic   Amplitude  Std. Dev.    Phase (degrees)  Std. Dev.\n"
        for i in range(1, self.numharm+1):
            ix = 2*(i-1)+self.numpoly
            a = self.params[ix]
            b = self.params[ix+1]
            c = a*a+b*b
            siga = (a*a*self.covar[ix][ix]+b*b*self.covar[ix+1][ix+1])/c
            sigtheta = (b*b*self.covar[ix][ix]+a*a*self.covar[ix+1][ix+1])/(c*c)
            stats += "%5d %11.2f %10.2f %16.2f %12.2f\n" % (i, sqrt(c), sqrt(siga), atan2(b,a)*180/pi, sqrt(sigtheta)*180.0/pi)

        stats += "------------------------------------------------------\n"
        stats += "Full covariance matrix:\n"
        for i in range(self.numpm):
            for j in range(self.numpm):
                    stats += "%13.4e" % self.covar[i][j]
                    stats += "\n"

        stats += "------------------------------------------------------\n"
#      stats += "Chi squared value of function fit: %f\n",chisq)
        stats += "Residual standard deviation about function: %f\n" % self.rsd1
        stats += "------------------------------------------------------\n"
        stats += "\n"
        stats += "FILTER PARAMETERS\n"
        stats += "Short term self cutoff:       %3d days\n" % self.shortterm
        stats += "Long term self cutoff:        %3d days\n" % self.longterm
        stats += "Sampling interval:            %3g days\n" % self.sampleinterval
        stats += "\n"
#        stats += "Function Standard Deviation:          %8.4f" % sqrt(self.funcvar)
#        stats += "Short Term Filter Standard Deviation: %8.4f" % sqrt(self.varf1)
#        stats += "Long  Term Filter Standard Deviation: %8.4f" % sqrt(self.varf2)
#        stats += "Smoothed curve Standard Deviation:    %8.4f" % sqrt(self.varf1 + self.funcvar)
#        stats += "Trend curve Standard Deviation:       %8.4f" % sqrt(self.varf2 + self.funcvar)
#        stats += "Detrended Cycle Standard Deviation:   %8.4f" % sqrt(self.varf2 + self.varf1 + 2*self.funcvar)
#        stats += "Growth Rate Standard Deviation:       %8.4f" % sqrt(2*(self.varf2 + self.funcvar))
        stats += "\n"

        stats += "Residual standard deviation about smooth curve: %f\n" % self.rsd2

        return stats

    #------------------------------------------------------------
    def getAmplitudes(self):
        """ Get amplitudes of seasonal cycle for each year.
        The amplitude is from the detrended data, which is the
        harmonic part of function + smooth curve - trend curve.
        Find max and min values of this for each year, save the values
        and the dates at which they occur.

        Returns
        --------
        A list of tuples, each tuple has 6 values (year, total_amplitude, max_date, max_value, min_date, min_value)
        """

        # get harmonic part of function at interpolated data points
        ycycle = harmonics(self.params, self.xinterp-self.timezero, self.numpoly, self.numharm)

        # added short term smoothed data
        ycycle = ycycle + self.smooth - self.trend

        # Find max and min values of the seasonal cycle
        tyear = int(self.xinterp[0])
        amps = []
        amax = -9999
        amin = 9999
        for x,y in zip(self.xinterp, ycycle):
            year = int(x)
            if year != tyear:
                t = (tyear, amax - amin, dmax, amax, dmin, amin)
                amps.append(t)
                amax = -9999
                amin = 9999
                dmax = 0
                dmin = 0
                tyear = year
            if y > amax:
                amax = y
                dmax = x
            if y < amin:
                amin = y
                dmin = x

        return amps

    #------------------------------------------------------------
    def getFunctionValue(self, x):
        """ Determine the value of the function at time x.
        x can be either a single point or a list
        """

        return fitFunc(self.params, x-self.timezero, self.numpoly, self.numharm)

    #------------------------------------------------------------
    def getSmoothValue(self, x):
        """ Return the 'smoothed' data at time x
        This is the function plus the smoothed residuals.
        """

        ysmooth = self.getFunctionValue(self.xinterp)
        ysmooth = ysmooth + self.smooth

        f = interpolate.interp1d(self.xinterp, ysmooth, bounds_error=False)
        yi = f(x)

        return yi

    #------------------------------------------------------------
    def getTrendValue(self, x):
        """ Return the 'trend' of the data at time x
        This is the polynomial part of the function plus the trend of the residuals.
        i.e., poly plus the long term filter of the residuals
        """

        ytrend = self.getPolyValue(self.xinterp)
        ytrend = ytrend + self.trend

        f = interpolate.interp1d(self.xinterp, ytrend)
        yi = f(x)

        return yi

    #------------------------------------------------------------
    def getPolyValue(self, x):
        """ Get the values of the polynomial part of the function time x

        Returns
        -------
        A numpy 1d array with the polynomial values at the given x
        """

        xa = numpy.array(x)

        p = numpy.polyval(self.params[self.numpoly-1::-1], xa-self.timezero)

        return p

    #------------------------------------------------------------
    def getHarmonicValue(self, x):
        """ Get the values of the harmonic part of the function time x

        Returns
        -------
        A numpy 1d array with the harmonic values at the given x
        """

        xa = numpy.array(x)

        # get harmonic part of function at x
        y = harmonics(self.params, xa-self.timezero, self.numpoly, self.numharm)

        return y

    #------------------------------------------------------------
    def getGrowthRateValue(self, x):
        """ Get the values of the derivative of the trend

        Returns
        -------
        A numpy 1d array with the growth rate values at the given x
        """

        f = interpolate.interp1d(self.xinterp, self.deriv)
        yi = f(x)

        return yi

    #------------------------------------------------------------
    def getFilterResponse(self, cutoff):
        """ Get the filter response for a range of frequencies.
        Input
        -----
        cutoff - cutoff value in days for the filter

        Returns
        -------
        Two 1d numpy arrays, length 1000, with the frequency and the corresponding filter response
        for the given cutoff.

        Range of frequencies is 0 to 2*cutoff frequency, in 1000 steps
        """

        fmax = (365.0/float(cutoff)) * 2
        fstep = fmax/1000.
        cf = cutoff/365.0
        cutoff2 = 1.0/cf
        freq = numpy.linspace(0, fmax, 1000)
        rw = self._vfilt(freq, cutoff2, 6)    # get filter value at frequencies

        return freq, rw

    #------------------------------------------------------------
    def getMonthlyMeans(self):
        """ Get monthly mean values from the smoothed curve
        Note: first and last months could be incomplete

        Returns
        --------
        A list of tuples, each tuple has 5 values (year, month, value, std. deviation, n)
        """

        ysmooth = self.getSmoothValue(self.xinterp)

        a = []
        data = []
        tyear = 0
        tmonth = 0
        for x, y in zip(self.xinterp, ysmooth):
            (year, month, day, hour, minute, second) = self.calendarDate(x)

            if year != tyear or month != tmonth:
                if len(a):
                    mean = numpy.mean(a)
                    std = numpy.std(a, ddof=1)
                    data.append((tyear, tmonth, mean, std, len(a)))
                    a = []

            a.append(y)
            tyear = year
            tmonth = month

        mean = numpy.mean(a)
        std = numpy.std(a, ddof=1)
        data.append((year, month, mean, std, len(a)))

        return data

    #------------------------------------------------------------
    def getTrendCrossingDates(self):
        """ Get the dates when the smoothed curve crosses the trend curve.
        That is, when the detrended smooth seasonal cycle crosses 0.
        """

        # get harmonic part of function at interpolated data points
        ycycle = harmonics(self.params, self.xinterp-self.timezero, self.numpoly, self.numharm)

        # added short term smoothed data
        ycycle = ycycle + self.smooth - self.trend

        tcup = []
        tcdown = []
        tyear = int(self.xinterp[0])
        ty = ycycle[0]
        for x,y in zip(self.xinterp, ycycle):

            if ty < 0.0 and y >= 0.0:
                tcup.append(x)
            if ty > 0.0 and y <= 0.0:
                tcdown.append(x)

            ty = y


        return (tcup, tcdown)



    ###################################################
    # Don't use these routines for resolution less than 1 second.
    def calendarDate(self, decyear):

        dyr =  int(decyear)
        fyr = decyear - dyr

        if dyr % 4 == 0:
            nsec = fyr * (366*86400)
        else:
            nsec  = fyr * (365*86400)

        nsec = round(nsec, 0)

        ndays = int(nsec / 86400)
        doy  = ndays + 1

        if doy > 366:
            dyr = dyr + 1
            doy = 1
        month, day = self.to_mmdd(dyr, doy)

        nsecs = round(nsec - (ndays*86400), 0)
        hour  = int(nsecs / 3600)
        minute = int((nsecs - (hour *3600)) / 60)
        seconds = int(round(nsecs - (hour * 3600.0) - (minute * 60.0), 0))

        return dyr,month,day,hour,minute,seconds


    ###################################################
    def to_mmdd(self, year, doy):

        if year % 4 == 0:
            mona = [31,29,31,30,31,30,31,31,30,31,30,31]
        else:
            mona  = [31,28,31,30,31,30,31,31,30,31,30,31]

        idoy =  doy
        imon = 0
        while idoy - mona[imon] > 0:
            idoy  = idoy - mona[imon]
            imon += 1

        month = imon + 1
        return int(month), int(idoy)

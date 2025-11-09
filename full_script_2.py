#IMPORTING MODULES

import numpy as np
import pandas as pd
import os
from astropy.timeseries import LombScargle
import lightkurve as lk
import matplotlib.pyplot as plt
import shutil
from astropy.utils.data import _get_download_cache_loc


#Getting TIC ID strings (these will have an index as they are in a panda series)

tic_ids = pd.read_csv("for_tim_student.csv")["TICID"]

#forming into a simple python list

id_list = tic_ids.tolist()


#GRAPHICAL ANALYSIS **************

#search function

def selective_search(tic_id):
    search_result = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')

    if not search_result:
        print(f"⚠️ No TESS light curves found for TIC {tic_id}.")
        return None
    
    try:
        selec_search = search_result[search_result.author == 'SPOC']
        exptimes = selec_search.table['exptime']
        min_index = exptimes.argmin()
        parsed_search = selec_search[min_index]
    except:
        print("No Spoc result found")
        exptimes = search_result.table['exptime']
        min_index = exptimes.argmin()
        parsed_search = search_result[min_index]
        
    return parsed_search

#downloading data and removing bad data points function

def dwnlwd(parsed_search):
    data = parsed_search.download(quality_bitmask='default').remove_nans()
    return data

#rough plot

def quick_look(data):
    data.plot()
    plt.show()

#Filtering Data to remove oscillations unlikely to be caused by pulsations

def extracter(data):
    #.remove_outliers()
    
    time, flux = data.time.value, data.flux.value
    flux /= np.median(flux)
    time -= time[0]

    return time, flux

def filterer(time,flux):
    mask = (flux > 0.998) & (flux < 1.002)
    time = time[mask]
    flux = flux[mask]

    return time, flux

#lightcurve plotter function

def intensity_plot(time,flux,save, tic_id):
    plt.figure(figsize = (14,8))
    plt.scatter(time, flux, s = 0.5, c='black')
    plt.xlabel("Time Elapsed (days)")
    plt.ylabel("Normalised flux (dimensionless)")
    plt.title(f"Normalised lightcurve for TIC ID:{tic_id}")
    if save == "T":
        plt.savefig(f"lightcurve_of_{tic_id}.png")
    plt.close()

#Fourier Transform function
def calc_lomb_scargle(t,y):
    oversample = 10 # can be adjusted
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)
    fmin = df
    fmax = 1000 # set max freq in c/d

    freq = np.arange(fmin, fmax, df / oversample)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    fct = np.sqrt(4./len(t))
    amp = np.sqrt(sc) * fct * 1e6
    return freq, amp # freq in cycles per day and amp in ppm

#fourier transform plotter function
def fourier_plot_1(freq,amp,save, tic_id):
    plt.figure(figsize = (14, 8))
    plt.plot(freq, amp, c ='black')
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude (ppm)")
    plt.xlim(0,60)
    plt.title(f"Fourier Transform for Intensity Oscillations of TIC ID:{tic_id}")
    if save == "T":
        plt.savefig(f"fourier1_of_{tic_id}.png")
    plt.close()

#Close up (if needed)
def fourier_plot_2(freq,amp,save, tic_id):
    plt.figure(figsize = (14, 8))
    plt.plot(freq, amp, c ='black')
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude (ppm)")
    plt.xlim(0,20)
    plt.title(f"Fourier Transform for Intensity Oscillations of TIC ID:{tic_id}")
    if save == "T":
        plt.savefig(f"fourier2_of_{tic_id}.png")
    plt.close()


#Compiling Function

def star_analysis(tic_id,mask,save):

    parsed_search_result = selective_search(tic_id)

    if parsed_search_result == None:
        return
    
    data = dwnlwd(parsed_search_result)

    #quick_look(data)

    time, flux = extracter(data)

    if mask == "T":
        time, flux = filterer(time,flux)

    intensity_plot(time,flux,save, tic_id)

    freq, amp = calc_lomb_scargle(time, flux)

    fourier_plot_1(freq,amp,save, tic_id)

    fourier_plot_2(freq,amp,save, tic_id)


#SUMMARY GENERATOR **************

raw_table = pd.read_csv("tess_sector_91_92.csv",skiprows=[0, 1],names=["update_date","main_id","TICID","gaiadr3_id","CCD","Tmag","RA","Dec","sector","count","sp_type","sp_qual","plx_value","V","B","G","otype","nbref","rvz_radvel","rvz_redshift","gaiadr3_plx","gaiadr3phot_g_mean_mag","gaiadr3_bp_rp","abs_mag_rough"])

def finder(search_column, return_column, tic_id):
    try:
        # Find the index where the search_column equals the search_number
        index_of_match = raw_table[raw_table[search_column] == tic_id].index[0]
        # Use .loc to get the value at that specific index and column
        return raw_table.loc[index_of_match, return_column]
    except IndexError:
        # This handles the case where no match is found
        return None

#getting name
def id_r(tic_id):
    return finder("TICID","main_id",tic_id)

#getting spectral type
def spec_r(tic_id):
    return finder("TICID","sp_type",tic_id)

#getting object type
def otype_r(tic_id):
    return finder("TICID","otype",tic_id)

#Distance (parsecs, input parallax is in mas)
def dist(tic_id):
    try:
        return (1/float(finder("TICID","gaiadr3_plx",tic_id)))*1e3
    except:
        return None

#photo g mean mag (apparent)
def apar_mag_r(tic_id):
    try:
        return float(finder("TICID","gaiadr3phot_g_mean_mag",tic_id))
    except:
        return None

#Absolute Mag
def abs_mag_r(tic_id):
    try:
        apar = float(finder("TICID","gaiadr3phot_g_mean_mag",tic_id))
        dist = (1/float(finder("TICID","gaiadr3_plx",tic_id)))*1e3
        return apar - 5*np.log10(dist/10)
    except:
        return None

#bp_rp (gaia mag)
def bp_rp_r(tic_id):
    try:
        return float(finder("TICID","gaiadr3_bp_rp",tic_id))
    except:
        return None

#RA and DEC
def ra_dec(tic_id):
    try:
        return (finder("TICID","RA",tic_id),finder("TICID","Dec",tic_id))
    except:
        return None

#refernces in other literature
#UNDER CONSTRUCTION


#Summary Variables
def sum_gen(tic_id):
    return id_r(tic_id),spec_r(tic_id),otype_r(tic_id),dist(tic_id),apar_mag_r(tic_id),abs_mag_r(tic_id),bp_rp_r(tic_id),ra_dec(tic_id)

#PDF GENERATOR***********

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

from reportlab.lib.styles import getSampleStyleSheet


#creating information

def info_creator(tic_id):
    star_analysis(tic_id,"F","T")
    name, spec_type, object_type, distance, apparent_mag, absolute_mag, bp_rp, ra_dec = sum_gen(tic_id)
    return name, spec_type, object_type, distance, apparent_mag, absolute_mag, bp_rp, ra_dec

#pdf generator function
def pdf_creator(tic_id):
    name, spec_type, object_type, distance, apparent_mag, absolute_mag, bp_rp, ra_dec = info_creator(tic_id)

    pdf_name = f"{tic_id} Summary Sheet"
    doc = SimpleDocTemplate(f"sum_sheets/{pdf_name}.pdf", leftMargin=20,rightMargin=20, topMargin=20, bottomMargin=20)

    styles = getSampleStyleSheet()
    story = []

    # Use conditional expressions to format values only if they are not None
    formatted_distance = f"{distance:.2f}" if distance is not None else "N/A"
    formatted_apparent_mag = f"{apparent_mag:.2f}" if apparent_mag is not None else "N/A"
    formatted_absolute_mag = f"{absolute_mag:.2f}" if absolute_mag is not None else "N/A"
    formatted_bp_rp = f"{bp_rp:.2f}" if bp_rp is not None else "N/A"
    formatted_coords = f"{ra_dec[0]}, {ra_dec[1]}" if all(coord is not None for coord in ra_dec) else "N/A"

    text = f"""
<b>Summary on chosen star designated by TICID:</b> {tic_id}<br/>
<b>Common Name:</b> {name}<br/>
<b>Spectral Type:</b> {spec_type}<br/>
<b>Object Type:</b> {object_type}<br/>
<b>Distance:</b> {formatted_distance} (pc)<br/>
<b>Equitorial Coordinates:</b> {formatted_coords} (deg)<br/>
<b>Apparent GAIA Magnitude:</b> {formatted_apparent_mag}<br/>
<b>Absolute GAIA Magnitude:</b> {formatted_absolute_mag}<br/>
<b>BP-RP Value (Gaia):</b> {formatted_bp_rp}
"""
    story.append(Paragraph(text, styles["Normal"]))
    story.append(Spacer(1, 20))

    image_files_to_add = [f"lightcurve_of_{tic_id}.png", f"fourier1_of_{tic_id}.png", f"fourier2_of_{tic_id}.png"]
    
    for fname in image_files_to_add:
        if os.path.exists(fname):
            try:
                img = Image(fname)
                img._restrictSize(doc.width, doc.height)
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"Failed to add image {fname} to PDF: {e}")
    
    doc.build(story)
    
    # After the PDF is built, delete the image files
    for fname in image_files_to_add:
        if os.path.exists(fname):
            os.remove(fname)

    
#TESTING *************
#pdf_creator(322928423)
#pdf_creator(204191276)
#star_analysis(46095850)

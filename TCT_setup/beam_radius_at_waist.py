import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as opt
from math import pi, cos, sin, sqrt, log
import pandas as pd
import re
from scipy.optimize import curve_fit


##----------------------READ ME-------------------#
# This script runs through a series of images taken before and after the focus of a gaussian beam and makes a fit to each. 
# Using the radius at varius positions, it then fits the radii to find the radius at the focal point.
#Just make sure the files of the image are named as follows:(something)_dmm_(somethingelse)[type].bmp where type is either a,b or c
##----------------------READ ME-------------------#
path = r'C:\Users\josev\OneDrive - Universidade de Lisboa\coisas externas\GoLP\LGAD_charac\Beam_size_measurements_delayed_v3'
output_path =  r'C:\Users\josev\OneDrive - Universidade de Lisboa\coisas externas\GoLP\LGAD_charac\Result_plots_2'
d = "delayed" #switch between delayed and not delayed (just for lgad characterization) - use this to differentiate between which folder the files came from
cntr = [670.,330.] #rough initial guess for all fits [820,580] for delayed & unaltered is all over the place - make a general guess, if fit fails, code will ask for new input in terminal


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def parse_filename(name):
    # extract position
    match = re.search(r"_([\d.]+)mm_", name)
    position = float(match.group(1)) if match else None

    # extract type (normal, b, c)
    if name.endswith("b.bmp"):
        ftype = "b"
    elif name.endswith("ado.bmp"):
        ftype = "c"
    else:
        ftype = "a"

    return ftype, position

def fwhm_of_z(z, F0, zR, z0):
    return F0 * np.sqrt(1 + ((z - z0) / zR)**2)


df = pd.DataFrame(columns=["Type","Position", "Radius_FWHM", "Radius_uncertainty","z_err"])


for entry in os.scandir(path):
    if entry.is_file():
        ftype, position = parse_filename(entry.name)
        ##---------View Image----------##
        img = rgb2gray(mpimg.imread(entry.path))
        # imgplot = plt.imshow(img)
        # plt.title(entry.name)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()


        ##---------Fit Gaussian Elipse----------##

        h, w = img.shape
        data = img.ravel()


        x = np.linspace(0, w, w)
        y = np.linspace(0, h, h)
        x, y = np.meshgrid(x, y)




        # automatic guesses for the rest
        center = cntr
        amp_guess = img.max() - img.min()
        offset_guess = img.min()
        sigma_x_guess = img.shape[1] / 100
        sigma_y_guess = img.shape[0] / 100
        theta_guess = 10

        initial_guess = (
            amp_guess,
            center[0],
            center[1],
            sigma_x_guess,
            sigma_y_guess,
            theta_guess,
            offset_guess
        )


        # --- robust fit loop ---
        while True:
            try:
                popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)
                break  # success

            except Exception as e:
                print("\nFit failed:", e)
                plt.imshow(img)
                plt.title("Fit failed — choose new starting parameters")
                plt.show()

                ans = input("Retry with new starting parameters? (y/n): ").strip().lower()
                if ans != "y":
                    print("Skipping this file.")
                    popt = None
                    break

                center = input("Enter new center coordinates: x,y ")
                center = [float(x) for x in center.split(",")]
                sigma_x_guess = float(input("sigma_x: "))
                sigma_y_guess = float(input("sigma_y: "))
                initial_guess = (200, center[0], center[1], sigma_x_guess, sigma_y_guess, 10, 10)


        if popt is None:
            continue  # skip to next file

        

        # create new data with these parameters
        data_fitted = twoD_Gaussian((x, y), *popt)

        print("done")

        print(popt)


        ##---------View Fit ----------##


        u=popt[1]       #x-position of the center
        v=popt[2]       #y-position of the center
        a=abs(popt[3]*sqrt(2 * log(2)))       #radius on the x-axis
        b=abs(popt[4]*sqrt(2 * log(2)))       #radius on the y-axis
        t_rot=-popt[5]   #rotation angle

        t = np.linspace(0, 2*pi, 100)
        Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
            #u,v removed to keep the same center location
        R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
            #2-D rotation matrix

        Ell_rot = np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

        # plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
        # plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange' )    #rotated ellipse
        # plt.grid(color='lightgray',linestyle='--')
        # plt.show(block=False)
        # plt.pause(2)
        # plt.close()


        # --- REVIEW LOOP: accept / retry / skip ---
        while True:
            FWHM_x_pix_provisional = 2 * np.sqrt(2 * np.log(2)) * abs(popt[3])
            FWHM_y_pix_provisional = 2 * np.sqrt(2 * np.log(2)) * abs(popt[4])

            pixel_to_um = 5.3  # micrometers
            FWHM_x_um_provisional = FWHM_x_pix_provisional * pixel_to_um
            FWHM_y_um_provisional = FWHM_y_pix_provisional * pixel_to_um

            print("Provisional Equivalent FWHM (um):",sqrt(FWHM_y_um_provisional*FWHM_x_um_provisional))

            # Show current fit
            plt.imshow(img)
            plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')
            plt.title(f"Accept (a), Retry (r), Skip (s) or Zoom and Retry (zr)? \n {entry.name}")
            plt.show()

            ans = input("Accept (a), Retry (r), or Skip (s), Zoom and retry (zr)? ").strip().lower()

            # ---------------- ACCEPT ----------------
            if ans == "a":
                # leave the loop and append to dataframe
                break

            # ---------------- SKIP ----------------
            elif ans == "s":
                print("Fit skipped.")
                popt = None
                break   # exit review loop; outer loop will skip file

            # ---------------- RETRY ----------------
            elif ans == "r":
                print("Retrying with new initial parameters.")

                # Show image again for choosing new center
                plt.imshow(img)
                plt.title("Choose new starting parameters")
                plt.show()

                center = input("Enter new center coordinates (x,y): ")
                center = [float(x) for x in center.split(",")]

                sigma_x_guess = float(input("sigma_x: "))
                sigma_y_guess = float(input("sigma_y: "))
                theta_guess = float(input("theta (radians): "))

                amp_guess = img.max() - img.min()
                offset_guess = img.min()

                initial_guess = (
                    amp_guess,
                    center[0],
                    center[1],
                    sigma_x_guess,
                    sigma_y_guess,
                    theta_guess,
                    offset_guess
                )

                # Try the refit
                try:
                    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)
                except Exception as e:
                    print("Retry failed:", e)
                    # stay inside the review loop and ask again
                    continue

                # --- Recompute ellipse for the updated fit ---
                u = popt[1]
                v = popt[2]
                a = abs(popt[3] *  sqrt(2 * log(2)))
                b = abs(popt[4] *  sqrt(2 * log(2)))
                t_rot = -popt[5]

                t = np.linspace(0, 2 * pi, 100)
                Ell = np.array([a * np.cos(t), b * np.sin(t)])
                R_rot = np.array([[cos(t_rot), -sin(t_rot)],
                                [sin(t_rot),  cos(t_rot)]])
                Ell_rot = R_rot @ Ell

                # Now loop back to show updated fit and ask again
                continue

            # ---------------- RETRY ----------------
            elif ans == "zr":
                print("Retrying with new initial parameters and a zoomed window.")

                plt.imshow(img)
                plt.title("Choose window and new starting parameters")
                plt.show()
                            
                wndw = input("Enter new window bounds (x1,x2,y1,y2): ")
                wndw = [int(x) for x in wndw.split(",")]
                            
                # --- CROP IMAGE ---
                img = rgb2gray(mpimg.imread(entry.path)[wndw[2]:wndw[3], wndw[0]:wndw[1]])

                # --- REBUILD GRID AND DATA FOR CROPPED IMAGE ---
                h, w = img.shape
                data = img.ravel()

                x = np.linspace(0, w, w)
                y = np.linspace(0, h, h)
                x, y = np.meshgrid(x, y)

                # Now your center must be in *cropped* coordinates (which you are doing)
                center = input("Enter new center coordinates (x,y): ")
                center = [float(x) for x in center.split(",")]

                sigma_x_guess = float(input("sigma_x: "))
                sigma_y_guess = float(input("sigma_y: "))
                theta_guess = float(input("theta (radians): "))

                amp_guess = img.max() - img.min()
                offset_guess = img.min()

                initial_guess = (
                    amp_guess,
                    center[0],
                    center[1],
                    sigma_x_guess,
                    sigma_y_guess,
                    theta_guess,
                    offset_guess
                )

                try:
                    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)
                except Exception as e:
                    print("Retry failed:", e)
                    continue


                # --- Recompute ellipse for the updated fit ---
                u = popt[1]
                v = popt[2]
                a = abs(popt[3] *  sqrt(2 * log(2)))
                b = abs(popt[4] *  sqrt(2 * log(2)))
                t_rot = -popt[5]

                t = np.linspace(0, 2 * pi, 100)
                Ell = np.array([a * np.cos(t), b * np.sin(t)])
                R_rot = np.array([[cos(t_rot), -sin(t_rot)],
                                [sin(t_rot),  cos(t_rot)]])
                Ell_rot = R_rot @ Ell

                # Now loop back to show updated fit and ask again
                continue

            # ---------------- INVALID INPUT ----------------
            else:
                print("Invalid input. Please enter a, r, or s.")
                continue

        # --- OUTSIDE REVIEW LOOP ---
        if popt is None:
            continue  # skip this file entirely

        ##---------Calculate Radius----------##

        print("FWHM radius (pixels): ",round(a,2),round(b,2))

        FWHM_x_pix = 2 * np.sqrt(2 * np.log(2)) * abs(popt[3])
        FWHM_y_pix = 2 * np.sqrt(2 * np.log(2)) * abs(popt[4])

        pixel_to_um = 5.3  # micrometers
        FWHM_x_um = FWHM_x_pix * pixel_to_um
        FWHM_y_um = FWHM_y_pix * pixel_to_um

        radius = sqrt(FWHM_x_um*FWHM_y_um)

        # uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        sigma_x_err = perr[3]
        sigma_y_err = perr[4]

        # constant
        k = 2 * np.sqrt(2 * np.log(2))

        # FWHM uncertainties (in micrometers)
        FWHM_x_err = k * sigma_x_err * pixel_to_um
        FWHM_y_err = k * sigma_y_err * pixel_to_um

        # radius uncertainty
        rel_err_sq = (FWHM_x_err / FWHM_x_um)**2 + (FWHM_y_err / FWHM_y_um)**2
        err_radius = 0.5 * radius * np.sqrt(rel_err_sq)

        print("Just Appended: \n FWHM (x, y) = ", FWHM_x_um, "+/-", FWHM_x_err, "µm,", FWHM_y_um,"+/-", FWHM_y_err, "µm\n", "Equivalent FWHM = ", radius, "+/-", err_radius, "µm" )


        df.loc[len(df)] = {
            "Type": ftype,
            "Position": position,
            "Radius_FWHM": radius,
            "Radius_uncertainty": err_radius,
            "z_err": 0}

        
print(df)
types = df["Type"].unique()

for t in types:

    df_t = df[df["Type"] == t].sort_values("Position")
    z = df_t["Position"].values
    w = df_t["Radius_FWHM"].values
    w_err = df_t["Radius_uncertainty"].values
    z_err = df_t["z_err"].values

    # initial guesses
    w0_guess = np.min(w)
    z0_guess = z[np.argmin(w)]
    zR_guess = 5  # mm, just a reasonable starting point

    popt, pcov = curve_fit(
        fwhm_of_z,
        z,
        w,
        p0=[w0_guess, zR_guess, z0_guess],
        sigma=w_err,          # weight by radius uncertainty
        absolute_sigma=True   # ensures correct covariance scaling
    )

    perr = np.sqrt(np.diag(pcov))
    w0, zR, z0 = popt
    w0_err, zR_err, z0_err = perr

    print(f"\nType {t}:")
    print(f" fwhm0 = {w0:.3f} ± {w0_err:.3f} µm")
    print(f"  zR = {zR:.3f} ± {zR_err:.3f} mm")
    print(f"  z0 = {z0:.3f} ± {z0_err:.3f} mm")

    # Plot
    z_fit = np.linspace(min(z), max(z), 300)
    w_fit = fwhm_of_z(z_fit, *popt)

    plt.errorbar(z, w, xerr=z_err, yerr=w_err, fmt='o', label=f"Data ({t})")
    plt.plot(z_fit, w_fit, label=f"Fit ({t})")
    plt.xlabel("z (mm)")
    plt.ylabel("fwhm(z) (µm)")
    plt.legend()
    plt.title(f"Beam radius vs z for type {t} ({d})")
    plt.savefig(os.path.join(output_path, f"fwhm(z)_type{t}_{d}"), dpi=300, bbox_inches="tight")
    plt.show()

import numpy as np


def ANOPP_OASPL(self, azimuth, polar, dz, dist, spd):

    #Flight parameters
    T_0 = 288.15
    rho_0 = 1.225
    R = 287.05
    g = 9.80665
    l = -0.0065
    gamma = 1.4 
    mu = 1.84E-5 #ambient dynamic viscosity [kg/(ms)]
    theta = polar
    phi = azimuth
    r = dist
    pe02 = (2E-5)**2 #reference value

    #Aircraft parameters
    A_w  = None #wing area [m^2]
    b_w = None #wing span [m]
    A_f = None #flap area [m^2]
    b_f = None #flap span [m]
    df = None #flap deflection agle #[rad]
    n_MLG = None #number of wheels per boggie (MLG) [-]
    d_MLG = None #diameter of MLG [m]
    d_NLG = None #diameter of NLG [m]   


    #Parameters for geometry function
    K_CW = 4.646E-5 #K constant for trailing edge clean wing [-]
    K_SL = 4.646E-5 #K constant for leading edge slats [-]
    K_FL = 2.787E-4 #K constant for trailing edge slats [-]
    K_LDG_2 = 4.349E-4 #K constant for landing gear with 2 wheels [-]
    K_LDG_4 = 3.414E-4 #K constant for landing gear with 4 wheels [-]
    a_CW = 5 #a constant for clean wing [-]
    a_SL = 5 #a constant for LE slats [-]
    a_FL = 6 #a constant for TE flaps [-]
    a_LDG = 6 #a constant for landing gear (both MLG and NLG) [-]

    #Frequencies for 1/3 octave band
    band_numbers = np.arange(1, 44) #band numbers
    f_n = 10**(band_numbers / 10) #centre frequencies
    f_U = 2**(1/6) * f_n #upper bound
    f_L = f_n / (2**(1/6)) #lower bound
    delta_f = f_U - f_L #bandwidth

    #Ambient conditions
    temp = T_0 + l * dz
    c = np.sqrt(gamma * R * temp)
    M = spd / c
    rho = rho_0 * (temp / T_0)**(-g / (R * l) - 1)  


    #clean wing calculations
    G_cw = 0.37 * (A_w/b_w**2) * ((rho * M * c * A_w)/(mu * b_w))**(-0.2)
    L_cw = G_cw * b_w
    S_cw = (f_n * L_cw * (1 - M * np.cos(theta))) / (M * c)
    F_cw = 0.613 * (10 * S_cw)**4 * ((10 * S_cw)**1.5 + 0.5)**(-4)
    D_cw = 4 * (np.cos(phi))**2 * (np.cos(theta/2))**2
    P_cw = (K_CW * M**a_CW * G_cw * (rho * c**3 * b_w**2 ))
    p_e_2_cw = (rho * c * P_cw * D_cw * F_cw) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)


    #slats calculations
    G_sl = 0.37 * (A_w/b_w**2) * ((rho * M * c * A_w)/(mu * b_w))**(-0.2)
    L_sl = G_sl * b_w
    S_sl = (f_n * L_sl * (1 - M * np.cos(theta))) / (M * c)
    F_sl = 0.613 * (10 * S_sl)**4 * ((10 * S_sl)**1.5 + 0.5)**(-4) + 0.613 * (2.19 * S_sl)**4 * ((2.19 * S_sl)**1.5 + 0.5)**(-4)
    D_sl = 4 * (np.cos(phi))**2 * (np.cos(theta/2))**2
    P_sl = (K_SL * M**a_SL * G_sl * (rho * c**3 * b_w**2 ))
    p_e_2_sl = (rho * c * P_sl * D_sl * F_sl) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)
        

    #flaps calculations
    G_fl = A_f/b_w**2 * (np.sin(df))**2
    L_fl = A_f / b_f
    S_fl = (f_n * L_fl * (1 - M * np.cos(theta))) / (M * self.c)
    F_fl = np.empty(S_fl.shape)
    F_fl[S_fl < 2] = 0.0480 * S_fl[S_fl < 2]
    F_fl[(2 <= S_fl) & (S_fl <= 20)] = 0.1406 * (S_fl[(2 <= S_fl) & (S_fl <= 20)])**(-0.55)
    F_fl[S_fl > 20] = 216.49 * (S_fl[S_fl > 20])**(-3)
    D_fl = 3 * (np.sin(df) * np.cos(theta) + np.cos(df) * np.sin(theta) * np.cos(phi))**2
    P_fl = (K_FL * M**a_FL * G_fl * (rho * c**3 * b_w**2 ))
    p_e_2_fl = (rho * c * P_fl * D_fl * F_fl) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

    #MLG calculations
    G_mlg = n_MLG * (d_MLG/b_w)**2
    L_mlg = d_MLG
    S_mlg = (f_n * L_mlg * (1 - M * np.cos(theta))) / (M * c)
    #F_mlg = n_MLG * 0.0577 * S_mlg**2 * (0.25 * S_mlg**2 + 1)**(-1.5) #for MLG with 4 wheels
    F_mlg = n_MLG * 13.59 * S_mlg**2*(S_mlg**2 + 12.5)**(-2.25) #for MLG with 2 wheels
    D_mlg = 3/2 * (np.sin(theta))**2
    P_mlg = (K_LDG_4 * M**a_LDG * G_mlg * (rho * c**3 * b_w**2 ))
    p_e_2_mlg = (rho * c * P_mlg * D_mlg * F_mlg) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

    #NLG calculations
    G_nlg = 2 * (d_NLG/b_w)**2
    L_nlg = d_NLG
    S_nlg = (f_n * L_nlg * (1 - M * np.cos(theta))) / (M * c)
    F_nlg = 2 * 13.59 * S_nlg**2*(S_nlg**2 + 12.5)**(-2.25)
    D_nlg = 3/2 * (np.sin(theta))**2
    P_nlg = (K_LDG_2 * M**a_LDG * G_nlg * (rho * c**3 * b_w**2 ))
    p_e_2_nlg = (rho * c * P_nlg * D_nlg * F_nlg) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

    #OSPL calculations
    PBL_wing = 10 * np.log10(p_e_2_cw/pe02) #- 10 * np.log10(delta_f) #PSL - wing
    PBL_slats = 10 * np.log10(p_e_2_sl/pe02) #- 10 * np.log10(delta_f) #PSL - slats
    PBL_flap = 10 * np.log10(p_e_2_fl/pe02) #- 10 * np.log10(delta_f) #PSL - flaps
    PBL_MLG = 10 * np.log10(p_e_2_mlg/pe02) #- 10 * np.log10(delta_f) #PSL - main landing gear
    PBL_NLG = 10 * np.log10(p_e_2_nlg/pe02) #- 10 * np.log10(delta_f) #PSL - nose landing gear

    PBL_total= 10 * np.log10(10**(PBL_wing/10) + 10**(PBL_slats/10) + 10**(PBL_flap/10) + 10**(PBL_MLG/10) + 10**(PBL_NLG/10)) #PSL - total

    #A-weighting function
    dL_A = -145.528 + 98.262 * np.log10(f_n) - 19.509 * (np.log10(f_n))**2 + 0.975 * (np.log10(f_n))**3
    PBL_total_A = PBL_total + dL_A

    #OASPL (or L_A) calculation
    sum = 0
    for i in range(len(PBL_total_A)):
        sum += 10 ** (PBL_total_A[i] / 10) 

    OASPL = 10 * np.log10(sum)
    #OASPL is the only metric which can be displayed in real time. To account for the effect of noise duration, SEL must becalculated in post.




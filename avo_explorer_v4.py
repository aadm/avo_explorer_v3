# avo_explorer_v4
# -----------------
# aadm 2019, 2023, 2024
#
# streamlit port of the clunky v2 jupyter notebook+docker
# found at https://github.com/aadm/avo_explorer
# 
# to run locally: 
# $ streamlit run avo_explorer_v3.py
#
# to run webapp:
#
# https://xxx.streamlit.app/
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import streamlit as st


def pr(vp, vs):
    '''
    Calculate Poisson's ratio. (aadm 2017)
    '''
    return (vp**2-2*vs**2) / (2*(vp**2-vs**2))


def hilterman(vp1, vs1, rho1, vp2, vs2, rho2, theta,
         angcontrib=False):
    '''
    Calculate P-wave reflectivity with Verm & Hilterman's equation. (aadm 2024)

    Hilterman (2001), Seismic Amplitude Interpretation,
    SEG Distinguished Instructor Short Course n.4 (p. 3-35)
    '''
    a = np.deg2rad(theta)
    pr1 = pr(vp1, vs1)
    pr2 = pr(vp2, vs2)
    dvp = vp2 - vp1
    drho = rho2 - rho1
    dpr = pr2 - pr1

    # calculate average properties
    avp = np.mean([vp1, vp2], axis=0)
    avs = np.mean([vs1, vs2], axis=0)
    arho = np.mean([rho1, rho2], axis=0)
    apr = np.mean([pr1, pr2], axis=0)
 
    R0 = 0.5 * (dvp/avp + drho/arho) # normal incidence reflection coefficient

    C = 4 * avs**2/avp**2 * np.sin(a)**2
    term1 = R0 * (1 - 4 * avs**2/avp**2 * np.sin(a)**2)
    term2 = dpr/(1 - apr)**2 * np.sin(a)**2
    term3 = 0.5 * dvp/avp * (np.tan(a)**2 - 4 * avs**2/avp**2 * np.sin(a)**2)
    R = term1 + term2 + term3

    if angcontrib:
        return R, term1, term2, term3
    else:
        return R


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta,
           approx=True, terms=False, angcontrib=False):
    '''
    Calculate P-wave reflectivity with Shuey's equation. (aadm 2024)

    Mavko et al. (2009), The Rock Physics Handbook, 2nd ed.
      Cambridge University Press (p.102)
    Castagna (1993), AVO Analysis - Tutorial and Review in
      "Offset-Dependent Reflectivity: Theory and Practice of AVO Analysis",
      ed. J. P. Castagna and M. Backus, SEG Investigations in Geophysics n.8 (p.9)
    '''
    a = np.deg2rad(theta)
    pr1 = pr(vp1, vs1)
    pr2 = pr(vp2, vs2)
    dvp = vp2 - vp1
    drho = rho2 - rho1
    dpr = pr2 - pr1

    # calculate average properties
    avp = np.mean([vp1, vp2], axis=0)
    arho = np.mean([rho1, rho2], axis=0)
    apr = np.mean([pr1, pr2], axis=0)

    R0 = 0.5*(dvp/avp + drho/arho) # normal incidence reflection coefficient

    B = (dvp/avp) / (dvp/avp + drho/arho)
    A0 = B - 2*(1 + B) * (1 - 2*apr)/(1 - apr) 
    G = A0*R0 + dpr / (1 - apr)**2 # gradient: intermediate angles

    F = 0.5 * dvp/avp # F: dominant approaching critical angles

    # if angles _and_ velocities are array reshape
    if isinstance(theta, np.ndarray) & isinstance(vp1, np.ndarray):
        R0 = R0.reshape(-1, 1)
        G = G.reshape(-1, 1)
        F = F.reshape(-1, 1)

    R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2 - np.sin(a)**2)

    if approx:
        R = R0 + G*np.sin(a)**2
    else:
        R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2 - np.sin(a)**2)

    if terms:
        if angcontrib:
            return R, R0, G*np.sin(a)**2, F*(np.tan(a)**2 - np.sin(a)**2)
        else:
            return R, R0, G
    else:
        return R    


def get_avo_classes(hybrid_class_4=True):
    '''
    Returns reference AVO classes definition from 
    from Hilterman, 2001, Seismic Amplitude Interpretation,
    SEG-EAGE Distinguished Instructor Short Course.
    Class 4 from Castagna, J. P., and H. W. Swan, 1997,
    Principles of AVO crossplotting, The Leading Edge.
    '''
    tmp_shale = np.array([[3094, 1515, 2.40, 0],
                          [2643, 1167, 2.29, 0],
                          [2192, 818, 2.16, 0],
                          [3240, 1620, 2.34, 0]])
    tmp_sandg = np.array([[4050, 2526, 2.21, .2],
                          [2781, 1665, 2.08, .25],
                          [1542, 901, 1.88, .33],
                          [1650, 1090, 2.07, .163]])
    tmp_sandb = np.array([[4115, 2453, 2.32, .2],
                          [3048, 1595, 2.23, .25],
                          [2134, 860, 2.11, .33],
                          [2590, 1060, 2.21, .163]])
    avocl = ['CLASS1', 'CLASS2', 'CLASS3', 'CLASS4']
    logs = ['VP', 'VS', 'RHO', 'PHI']
    shale = pd.DataFrame(tmp_shale, columns=logs, index=avocl)
    sandg = pd.DataFrame(tmp_sandg, columns=logs, index=avocl)
    sandb = pd.DataFrame(tmp_sandb, columns=logs, index=avocl)
    if hybrid_class_4:
        sandb.loc['CLASS4'] = sandb.loc['CLASS3']
        sandg.loc['CLASS4'] = sandg.loc['CLASS3']
    return shale, sandb, sandg




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# initialize app

st.set_page_config(page_title='AVO Explorer', layout="centered")

# with st.sidebar:
#     st.title(':grey[AVO Explorer v4]')
#     st.write(
#         '''Porting of my old
#         [AVO Explorer notebook](https://github.com/aadm/avo_explorer).

#         (aadm 2019, 2023, 2024)''')

st.title(':grey[AVO Explorer v4]')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get elastic properties for default avo classes
sh, ssb, ssg = get_avo_classes()
avocl = sh.index.to_list()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: input elastic properties for shale and sand 

opt_vp = dict(min_value=1500., max_value=6000., step=10., format='%.0f')
opt_vs = dict(min_value=700., max_value=4000., step=10., format='%.0f')
opt_rho = dict(min_value=1.5, max_value=3.5, step=0.05, format='%.2f')
ep_ss = np.full(3, np.nan)
ep_sh = np.full(3, np.nan)
elastic_properties_widgets = st.columns(2, gap='medium')
with elastic_properties_widgets[0]:
    st.subheader('Shale (above)')
    ep_sh[0] = st.number_input('Vp', value=2200., **opt_vp)
    ep_sh[1] = st.number_input('Vs', value=820.,  **opt_vs)
    ep_sh[2] = st.number_input('rho', value=2.2, **opt_rho)
with elastic_properties_widgets[1]:
    st.subheader('Sand (below)')
    ep_ss[0] = st.number_input('Vp', value=1550., **opt_vp)
    ep_ss[1] = st.number_input('Vs', value=900.,  **opt_vs)
    ep_ss[2] = st.number_input('rho', value=1.9, **opt_rho)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: select angle range

aa = np.arange(0,91)
angle_range = st.select_slider(
    'Angle range', options=aa, value=[0.0, 30.0])

angles = np.arange(angle_range[0], angle_range[1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: misc options (reflectivity equation, AVO reference chart plot)

st.divider()

# ig = st.toggle('Plot AVO Intercept-Gradient')
ig = False


# akir = st.toggle('Use Aki-Richards reflectivity equation _(default: Shuey 2-term)_')

refl_eq = st.selectbox('Reflectivity equation', ['Shuey', 'Hilterman'])

avoref = st.radio('Select AVO reference', ['None', 'Brine Sand', 'Gas Sand'], index=2)

avocl = st.radio('Select AVO Class', avocl)

df_b = pd.DataFrame(angles, columns = ['angles'])
df_g = pd.DataFrame(angles, columns = ['angles'])
logs = ['VP', 'VS', 'RHO']
fluids = ['brine', 'gas']

for cl in avocl:
    avorefsh = sh.loc[cl, logs]
    avorefssb = ssb.loc[cl, logs]
    avorefssg = ssg.loc[cl, logs]
    if refl_eq == 'Shuey':
        df_b[cl] = shuey(*avorefsh, *avorefssb, angles, approx=False)
        df_g[cl] = shuey(*avorefsh, *avorefssg, angles, approx=False)
    else:
        df_b[cl] = hilterman(*avorefsh, *avorefssb, angles)
        df_g[cl] = hilterman(*avorefsh, *avorefssg, angles)


# calculate reflectivity from user input
df = pd.DataFrame(angles, columns = ['angles'])

if refl_eq == 'Shuey':
    df_b[cl] = shuey(*ep_sh, *ep_ss, angles, approx=False, terms=True, angcontrib=True)
else:
    df_b[cl] = hilterman(*ep_sh, *ep_ss, angles, angcontrib=True)
    angcontrib=True
    if akir:
        df['Reflectivity'] = akirichards(*ep_sh, *ep_ss, angles)
    else:
        df['Reflectivity'] = shuey(*ep_sh, *ep_ss, angles)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# make plot


def plot_rc(data, stack, histoplot):  
    f, ax = plt.subplots()
    # ax.axhline(R0, ls='--', color='0.5', label='NI')
    ax.plot(ang, rcs, ls='-', lw=4, color='k',  alpha=0.5, label='[Shuey] RC')
    ax.plot(ang, R0+t1s, ls='-', lw=2, color='r',  alpha=0.5, label='[Shuey] Near')
    ax.plot(ang, t2s, ls='-', lw=2, color='b',  alpha=0.5, label='[Shuey] Far')

    ax.plot(ang, rch, ls='--', lw=4, color='k',  alpha=0.5, label='[Hilterman] RC')
    ax.plot(ang, t1h, ls='--', lw=2, color='r',  alpha=0.5, label='[Hilterman] Near')
    ax.plot(ang, t2h, ls='--', lw=2, color='g',  alpha=0.5, label='[Hilterman] Mid')
    ax.plot(ang, t3h, ls='--', lw=2, color='b',  alpha=0.5, label='[Hilterman] Far')
    ax.legend()

    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.legend(fontsize='small')    
    ax.set_xlim(x0, x1)
    return fig          

st.divider()


rcs, R0, t1s, t2s = squit.avo.shuey(vp1, vs1, rho1, vp2, vs2, rho2, ang, approx=False, terms=True, angcontrib=True)
rch, t1h, t2h, t3h = squit.avo.hilterman(vp1, vs1, rho1, vp2, vs2, rho2, ang, angcontrib=True)


fig = plot_rc()

st.pyplot(fig=fig, use_container_width=True)


# avo_explorer_v3
# -----------------
# aadm 2019, 2023
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
import altair as alt

def get_avo_classes(hybrid_class_4=True):
    '''
    Returns reference AVO classes definition
    as 3 Pandas DataFrames.

    Parameters
    ----------
    hybrid_class_4 : bool, optional
        If True, returns alternative class 4 using Castagna's
        shale parameters coupled with Hilterman's class 3 sands.
        Default: True

    Returns
    -------
    shale : DataFrame
    sand_brine : DataFrame
    sand_gas : DataFrame

    Notes
    -----
    Classes 1, 2, 3 from Hilterman, 2001,
    Seismic Amplitude Interpretation,
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


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta, approx=True, terms=False):
    '''
    Calculate P-wave reflectivity with Shuey's equation.
    (aadm 2016, 2020)

    Parameters
    ----------
    vp1, vs1, rho1 : float or array_like
        P-, S-wave velocity (m/s) and density (g/cm3) of upper medium.
    vp2, vs2, rho2 : float or array_like
        P-, S-wave velocity (m/s) and density (g/cm3) of lower medium.
    theta : int or array_like
        Angle of incidence (degree).
    approx : bool, optional
        If True returns approximate 2-terms form. Default: True
    terms :  bool, optional
        If True returns reflectivity, intercept and gradient.
        Default: False.

    Returns
    -------
    R : float or array_like
        Reflectivity at angle theta.
    R0, G : float
        Intercept and gradient, only output if terms is True.

    Notes
    -----
    If input properties are arrays with length n
    and angles are also arrays with length m,
    the function returns a (n, m) array.

    References
    ----------
    Avseth et al. (2005), Quantitative Seismic Interpretation,
    Cambridge University Press (p.182)
    '''
    a = np.radians(theta)
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp = np.mean([vp1, vp2], axis=0)
    vs = np.mean([vs1, vs2], axis=0)
    rho = np.mean([rho1, rho2], axis=0)
    R0 = 0.5*(dvp/vp + drho/rho)
    G = 0.5*(dvp/vp) - 2*(vs**2/vp**2)*(drho/rho+2*(dvs/vs))
    F = 0.5*(dvp/vp)
    # # if angles is an array
    # if a.size > 1:
    #     R0 = R0.reshape(-1, 1)
    #     G = G.reshape(-1, 1)
    #     F = F.reshape(-1, 1)
    if approx:
        R = R0 + G*np.sin(a)**2
    else:
        R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2-np.sin(a)**2)
    if terms:
        return R, R0, G
    else:
        return R


def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    '''
    Calculates P-wave reflectivity with Aki-Richards equation.
    (aadm 2017)

    Parameters
    ----------
    vp1, vs1, rho1 : float or array_like
        P-, S-wave velocity (m/s) and density (g/cm3) of upper medium.
    vp2, vs2, rho2 : float or array_like
        P-, S-wave velocity (m/s) and density (g/cm3) of lower medium.
    theta : int or array_like
        Angle of incidence (degree).

    Returns
    -------
    R : float or array_like
        Reflectivity at angle theta.

    References
    ----------
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.182)
    '''
    a = np.radians(theta)
    p = np.sin(a)/vp1
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp  = np.mean([vp1,vp2])
    vs  = np.mean([vs1,vs2])
    rho = np.mean([rho1,rho2])
    A = 0.5*(1-4*p**2*vs**2)*drho/rho
    B = 1/(2*np.cos(a)**2) * dvp/vp
    C = 4*p**2*vs**2*dvs/vs
    R = A + B - C
    return R


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# initialize app

st.set_page_config(page_title='AVO Explorer', layout="centered")

st.title(':grey[AVO Explorer v3]')
st.write('')

st.write(
    '''
    Porting of my old
    [AVO Explorer notebook](https://github.com/aadm/avo_explorer).

    :grey[(aadm 2019,2023)]
    ''')
st.divider()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: input elastic properties for shale and sand 

opt_vp = dict(min_value=1500., max_value=6000., step=10., format='%.0f')
opt_vs = dict(min_value=700., max_value=4000., step=10., format='%.0f')
opt_rho = dict(min_value=1.5, max_value=3.5, step=0.05, format='%.2f')
ep_ss = np.full(3, np.nan)
ep_sh = np.full(3, np.nan)
elastic_properties_widgets = st.columns(2, gap='medium')
with elastic_properties_widgets[0]:
    st.header('Shale (above)')
    ep_sh[0] = st.number_input('Vp', value=2200., **opt_vp)
    ep_sh[1] = st.number_input('Vs', value=820.,  **opt_vs)
    ep_sh[2] = st.number_input('rho', value=2.2, **opt_rho)
with elastic_properties_widgets[1]:
    st.header('Sand (below)')
    ep_ss[0] = st.number_input('Vp', value=1550., **opt_vp)
    ep_ss[1] = st.number_input('Vs', value=900.,  **opt_vs)
    ep_ss[2] = st.number_input('rho', value=1.9, **opt_rho)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: select angle range

# angle_range = st.select_slider(
#     'Angle range', np.arange(0,50),
#     value=[0.0, 90.0])

aa = np.arange(0,91)
angle_range = st.select_slider(
    'Angle range', options=aa, value=[0.0, 30.0])

angles = np.arange(angle_range[0], angle_range[1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: misc options (reflectivity equation, AVO reference chart plot)

st.divider()

# ig = st.toggle('Plot AVO Intercept-Gradient')
ig = False


akir = st.toggle('Use Aki-Richards reflectivity equation _(default: Shuey 2-term)_')

plot_avoref = st.radio(
    'Select AVO reference',
    ['None', 'Brine Sand', 'Gas Sand'],
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build reference AVO classes
# get elastic properties for default avo classes
sh, ssb, ssg = get_avo_classes()

avocl = sh.index.to_list()
df_b = pd.DataFrame(angles, columns = ['angles'])
df_g = pd.DataFrame(angles, columns = ['angles'])
logs = ['VP', 'VS', 'RHO']
fluids = ['brine', 'gas']

for cl in avocl:
    avorefsh = sh.loc[cl, logs]
    avorefssb = ssb.loc[cl, logs]
    avorefssg = ssg.loc[cl, logs]
    if ig:
        _, Ib, Gb = shuey(*avorefsh, *avorefssb, angles, terms=True)
        _, Ig, Gg = shuey(*avorefsh, *avorefssg, angles, terms=True)
    else:
        if akir:
            df_b[cl] = akirichards(*avorefsh, *avorefssb, angles)
            df_g[cl] = akirichards(*avorefsh, *avorefssg, angles)
        else:
            df_b[cl] = shuey(*avorefsh, *avorefssb, angles)
            df_g[cl] = shuey(*avorefsh, *avorefssg, angles)

if not ig:
    df_b = df_b.melt('angles', var_name='AVO Class', value_name='Reflectivity')
    df_g = df_g.melt('angles', var_name='AVO Class', value_name='Reflectivity')

# calculate reflectivity from user input
df = pd.DataFrame(angles, columns = ['angles'])
if ig:
    _, Iu, Gu = shuey(*ep_sh, *ep_ss, angles, terms=True)
else:
    if akir:
        df['Reflectivity'] = akirichards(*ep_sh, *ep_ss, angles)
    else:
        df['Reflectivity'] = shuey(*ep_sh, *ep_ss, angles)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# make plot

st.divider()

# colrs = alt.Scale(range=['brown', 'olive', 'red', 'magenta'])


if ig:
    source = pd.DataFrame([[Iu, Gu]], columns=['I', 'G'])
    c0 = alt.Chart(source).mark_circle(
        color='black',
        size=100,
    ).encode(
        x=alt.X('I:Q', scale=alt.Scale(domain=[-.5, .5])),
        y=alt.Y('G:Q', scale=alt.Scale(domain=[-.5, .5]))
        )
else:
    c0 = alt.Chart(df).mark_line(
        color='black',
        strokeWidth=4,
    ).encode(
        x='angles:Q',
        y='Reflectivity:Q')

    cgas = alt.Chart(df_g).mark_line(opacity=0.5).encode(
        x='angles:Q',
        y='Reflectivity:Q',
        color=alt.Color('AVO Class:N'))
        # color=alt.Color('AVO Class:N', scale=colrs))

    cbri = alt.Chart(df_b).mark_line(
        opacity=0.5,
        strokeWidth=2,
        strokeDash=[8,4]
    ).encode(
        x='angles:Q',
        y='Reflectivity:Q',
        color=alt.Color('AVO Class:N'))
        # color=alt.Color('AVO Class:N', scale=colrs))


if plot_avoref == 'Brine Sand':
    chart = c0+cbri
elif plot_avoref == 'Gas Sand':
    chart = c0+cgas
else:
    chart = c0

st.altair_chart(chart, use_container_width=True, theme="streamlit")


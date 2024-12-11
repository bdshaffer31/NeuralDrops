
def disjoining_pressure (params,h):
    h_star = params.h_max/100
    n = 3
    m = 2
    dis_press = -params.sigma*np.square(params.theta_e)*(n-1)*(m-1)/(n-m)/(2*h_star)*((h_star/params.h_max)**n-(h_star/params.h_max)**m)
    return dis_press
import numpy
import pandas
import scipy.stats

def compute_b(s_x, S_xprime_arr, noise_var, gp):
    """Compute vector a and b in h(a,b)
    :param s_x: (s,x)
    :param S_xprime_arr: (S,x') for all x' in all_x
    :param noise_var: noise variance of (x,s)
    :param cov: covariance object
    :return: (a,b)
    """
    total_pts = numpy.vstack((s_x.reshape((1,-1)), S_xprime_arr))
    b = numpy.zeros(len(S_xprime_arr))
    for i in range(len(S_xprime_arr)):
        b[i] = gp.compute_variance_of_points(total_pts[[0, i+1], :])[0, 1] # covariance between (s,x) and (S,x')
    b /= numpy.sqrt(noise_var + gp.compute_variance_of_points(s_x.reshape((1,-1)))[0, 0])
    return b


def compute_c_A(a_in, b_in):
    """Algorithm 1 in Frazier 2009 paper"""
    M = len(a_in)
    # Use the same subscripts as in Algorithm 1, therefore, a[0] and b[0] are dummy values with no meaning
    a = numpy.concatenate(([numpy.inf], a_in))
    b = numpy.concatenate(([numpy.inf], b_in))
    c = numpy.zeros(M+1)
    c[0] = -numpy.inf
    c[1] = numpy.inf
    A = [1]
    for i in range(1, M):
        c[i+1] = numpy.inf
        while True:
            j = A[-1]
            c[j] = (a[j] - a[i+1]) / (b[i+1] - b[j])
            if len(A) != 1 and c[j] <= c[A[-2]]:
                del A[-1]
            else:
                break
        A.append(i+1)
    return c, A


def compute_kg(a, b, cost, cutoff=10.0):
    """Algorithm 2 in Frazier 2009 paper"""
    df = pandas.DataFrame({'a':a, 'b':b})
    sorted_df = df.sort_values(by=['b', 'a'])
    sorted_df['drop_idx'] = numpy.zeros(len(sorted_df))
    sorted_index = sorted_df.index
    # sorted_df.index = xrange(len(sorted_df))
    for i in xrange(len(sorted_index)-1):
        if sorted_df.ix[sorted_index[i], 'b'] == sorted_df.ix[sorted_index[i+1], 'b']:
            sorted_df.ix[sorted_index[i], 'drop_idx'] = 1
    truncated_df = sorted_df.ix[sorted_df['drop_idx']==0, ['a', 'b']]
    new_a = truncated_df['a'].values
    new_b = truncated_df['b'].values
    index_keep = truncated_df.index.values
    c, A = compute_c_A(new_a, new_b)
    if len(A) <= 1:
        return 0.0, numpy.array([])
    final_b = numpy.array([new_b[idx-1] for idx in A])
    final_index_keep = numpy.array([index_keep[idx-1] for idx in A])
    final_c = numpy.array([c[idx] for idx in A])
    # compute log h() using numerically stable method
    d = numpy.log(final_b[1:]-final_b[:-1]) - 0.5*numpy.log(2.*numpy.pi) - 0.5*numpy.power(final_c[:-1],2.0)
    abs_final_c = numpy.absolute(final_c[:-1])
    for i in xrange(len(d)):
        if abs_final_c[i] > cutoff:
            d[i] += numpy.log1p( -final_c[i] * final_c[i] / (final_c[i]*final_c[i]+1) )
        else:
            d[i] += numpy.log1p(-abs_final_c[i] * scipy.stats.norm.cdf(-abs_final_c[i]) / scipy.stats.norm.pdf(abs_final_c[i]))
    kg = numpy.exp(d).sum() / cost
    return kg, final_index_keep, abs_final_c


def compute_kg_given_x_prime(s, x, all_S_x, noise_var, cost, gp):
    '''Return KG/cost for point x at s'''
    a = gp.compute_mean_of_points(all_S_x)
    s_x = numpy.concatenate(([s], x))
    b = compute_b(s_x, all_S_x, noise_var, gp)

    # compute_kg(a, b, cost) would throw a valueerror if b is zero everywhere
    # (for zero covariance, e.g., due to bad hyperparameters or distant points)
    if numpy.all(numpy.abs(b) <= 1e-10):
        print 'Warning: compute_kg_given_x_prime: all bs are zero for IS ' + str(s)
        # Often caused by bad hyperparameters. If it occurs for all IS and many sets of initial points, increase
        # the number of initial points or enforce stricter bounds on length scales via the hyperprior.
        kg = 0.0
    else:
        kg, index_keep, abs_c = compute_kg(a, b, cost)
    # print(s, noise_var, cost, kg)
    return kg
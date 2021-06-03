import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()
    
    r = torch.zeros(X.shape)
    c = torch.clone(init_c)
    print(X)
    for ix in range(n_iters):
        
        sum1 = torch.zeros(1,2)
        sum2 = torch.zeros(1,2)
        x1 = torch.zeros(0)
        x2 = torch.zeros(0)

        for ix in range(X.shape[1]):

            if(torch.norm((X[:,ix] - c[:,0]),2)) < (torch.norm((X[:,ix] - c[:,1]),2)):
                r[:,ix] = torch.tensor([1,0])
                sum1 = sum1 + (X[:,ix])
                x1 = torch.cat((x1, X[:,ix]))
            else:
                r[:,ix] = torch.tensor([0,1])
                sum2 = sum2 + (X[:,ix])
                x2 = torch.cat((x2, X[:,ix]))
        sums = torch.sum(r, dim=1)

        c[:,0] = sum1 / sums[0]
        c[:,1] = sum2 / sums[1]
        c1 = torch.reshape(c[:,0],(2,1))
        c2 = torch.reshape(c[:,1],(2,1))
        
        x1 = torch.reshape(x1,(-1,2))
        x2 = torch.reshape(x2,(-1,2))
        x1 = torch.transpose(x1,1,0)
        x2 = torch.transpose(x2,1,0)

        hw4_utils.vis_cluster(c1,x1,c2,x2)
    return c